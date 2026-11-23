# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU block-table slot-mapping patch.

vLLM v1's ``BlockTable.compute_slot_mapping`` launches a Triton kernel
(``_compute_slot_mapping_kernel``). On LPU the kernel fails at compile time
(evas-triton-opt ``InsertDeallocOp.cpp`` "liveness within multiple blocks" —
a compile-time failure, not a launch failure). This patch replaces
``compute_slot_mapping`` with a pure-PyTorch
implementation that reproduces the kernel's logic exactly (including
context-parallel block interleaving), so no Triton kernel runs.

Only applied on LPU.
"""

import torch

import vllm.v1.worker.block_table as _bt_mod
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

_PATCHED = False
_ORIGINAL_COMPUTE_SLOT_MAPPING = None


def _compute_slot_mapping_pytorch(
    self,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Pure-PyTorch port of ``_compute_slot_mapping_kernel``.

    Computes ``slot_mapping[offset] = block_number * block_size +
    local_block_offset`` for every token, padded to ``max_num_batched_tokens``
    with ``PAD_SLOT_ID`` for CUDA-graph compatibility.
    """
    num_tokens = positions.shape[0]

    slot_mapping = self.slot_mapping.gpu
    # Work on the device where positions/block_table live (LPU).
    device = slot_mapping.device

    # Synchronize the device before reading buffers back. vLLM writes
    # query_start_loc / positions / block_table via CpuGpuBuffer.copy_to_gpu
    # (gpu.copy_(cpu, non_blocking=True)) — an ASYNC H2D copy. Under TP>1 on
    # LPU (double-die, multiple forked workers) reading those buffers back
    # without a sync returns torn data (e.g. positions [0,1,2,0,4] that reads
    # correctly as [0,1,2,3,4] after a sync). Single-worker TP1 is fast enough
    # that the copy happens to complete; TP4 exposes the race. A device-level
    # synchronize (no event, so no evError 2181) drains the pending copies.
    # (See memory project-fl-evas-slotmap-async-copy-race.)
    try:
        torch.lpu.synchronize()
    except Exception:
        pass

    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    cp_interleave = self.cp_kv_cache_interleave_size
    block_size = self.block_size
    virtual_block_size = block_size * total_cp_world_size

    block_table = self.block_table.gpu  # [max_num_reqs, max_num_blocks_per_req]

    # Start from a fully-padded slot mapping, then fill the real tokens.
    # (Equivalent to the kernel's last program which pads [num_tokens,
    # max_num_tokens) with PAD_ID.)
    slot_mapping.fill_(PAD_SLOT_ID)

    if num_tokens == 0:
        return

    # query_start_loc may arrive as a device tensor (the FL runner passes its
    # ``.gpu`` CpuGpuBuffer slice).  The full sync above has already drained the
    # async H2D, so a single batched ``.cpu()`` read is correct — and replaces
    # moving qsl to the device only to pull per-request scalars back with
    # ``.item()`` (2×num_reqs blocking D2H sync round-trips per step, plus the
    # redundant H2D).  We loop over requests (num_reqs is small) and vectorize
    # over each request's token range, mirroring the Triton kernel's
    # per-program structure.  This avoids torch.searchsorted, which the LPU
    # backend does not support for int64.
    qsl_cpu = query_start_loc.cpu()
    # LPU only supports integer remainder/floordiv on int32 (not int64), so do
    # all integer arithmetic below in int32. block_table itself is int32.
    pos_dev = positions.to(device=device, dtype=torch.int32)

    for req_idx in range(num_reqs):
        start_idx = int(qsl_cpu[req_idx])
        end_idx = int(qsl_cpu[req_idx + 1])
        if end_idx <= start_idx:
            continue

        pos = pos_dev[start_idx:end_idx]
        block_indices = pos // virtual_block_size
        # block_table[req_idx] is the row; gather block numbers per token.
        row = block_table[req_idx]
        block_numbers = row[block_indices].to(torch.int32)

        virtual_block_offsets = pos - block_indices * virtual_block_size

        if total_cp_world_size == 1 and cp_interleave == 1:
            local_block_offsets = virtual_block_offsets
            is_local = torch.ones_like(virtual_block_offsets, dtype=torch.bool)
        else:
            is_local = (
                (virtual_block_offsets // cp_interleave) % total_cp_world_size
                == total_cp_rank
            )
            local_block_offsets = (
                virtual_block_offsets // (total_cp_world_size * cp_interleave)
            ) * cp_interleave + (virtual_block_offsets % cp_interleave)

        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = torch.where(
            is_local, slot_ids, torch.full_like(slot_ids, PAD_SLOT_ID)
        )

        slot_mapping[start_idx:end_idx] = slot_ids.to(slot_mapping.dtype)


def _apply_block_table_dirty_flag() -> None:
    """Skip the per-step block-table CPU->GPU copy when nothing changed.

    vLLM MRV1 calls ``commit_block_table(num_reqs)`` at the start of every
    ``_prepare_inputs`` (gpu_model_runner.py:1796) to overlap the block-table
    copy with the following CPU prefill work. On LPU every copy/fence is
    expensive, and the block table only actually changes when a request gains /
    loses a KV block or a request row is added/removed.

    We track a dirty flag on the inner ``BlockTable`` and make
    ``commit_block_table`` a no-op when clean. The flag is set by **every**
    CPU-side mutation (``append_row``/``add_row``/``clear_row``/``move_row``/
    ``swap_row``/``clear``) — including ``clear_row``, so the
    ``remove_request()`` sync path (gpu_model_runner.py:5454, which relies on
    cleared rows being visible to the attention-metadata builder) still copies
    the moment a row is cleared. Patching the inner ``BlockTable`` covers all
    mutations, since ``MultiGroupBlockTable`` delegates to it.
    """
    cls = _bt_mod.BlockTable
    if getattr(cls, "_evas_block_table_dirty_patched", False):
        return

    _MUTATORS = (
        "append_row",
        "add_row",
        "clear_row",
        "move_row",
        "swap_row",
        "clear",
    )
    for name in _MUTATORS:
        original = getattr(cls, name)

        def _set_dirty(self, *args, __orig=original, **kwargs):
            ret = __orig(self, *args, **kwargs)
            self._evas_block_table_dirty = True
            return ret

        setattr(cls, name, _set_dirty)

    original_commit = cls.commit_block_table

    def _commit_if_dirty(self, num_reqs: int):
        if not getattr(self, "_evas_block_table_dirty", False):
            # Nothing changed since the last commit; the GPU buffer already
            # reflects the CPU table for [0, num_reqs). Skip the copy.
            return
        original_commit(self, num_reqs)
        self._evas_block_table_dirty = False

    cls.commit_block_table = _commit_if_dirty
    cls._evas_block_table_dirty_patched = True


def _compute_slot_mappings_024_pytorch(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
) -> torch.Tensor:
    """Pure-PyTorch port of ``BlockTables.compute_slot_mappings``.

    The active worker block-table class is ``BlockTables``
    (``vllm/v1/worker/gpu/block_table.py``) and the kernel is multi-group
    (``slot_mappings`` is ``[num_kv_cache_groups, max_tokens]``, one block
    table per group).  The
    Triton ``_compute_slot_mappings_kernel`` does not run on LPU (multi-block
    liveness compile failure), leaving every slot at ``PAD_SLOT_ID`` -- which
    makes the DSV4 SWA cache never get written (quant-insert sees all -1
    slots).  This port mirrors the kernel exactly, including per-group block
    tables and CP interleaving.
    """
    num_reqs = idx_mapping.shape[0]
    num_groups = self.num_kv_cache_groups
    dev = self.slot_mappings.device
    # Drain async H2D copies (idx_mapping / query_start_loc / positions arrive
    # via CpuGpuBuffer.copy_to_gpu, non_blocking=True).
    try:
        torch.lpu.synchronize()
    except Exception:
        pass
    self.slot_mappings.fill_(PAD_SLOT_ID)
    if positions.shape[0] == 0:
        return self.slot_mappings[:, :num_tokens_padded]

    # Single batched D2H for the per-request scalars (sync above already
    # drained the async H2D) — see the same idiom in
    # ``_compute_slot_mapping_pytorch``: no device round-trip, no per-scalar
    # ``.item()`` reads.
    qsl_cpu = query_start_loc.cpu()
    idx_cpu = idx_mapping.cpu()
    pos_dev = positions.to(device=dev, dtype=torch.int32)

    for g in range(num_groups):
        bs = self.block_sizes[g]
        vbs = bs * self.cp_size
        block_table = self.block_tables[g].gpu  # [max_reqs, max_blocks] int32
        row_slots = self.slot_mappings[g]
        for b in range(num_reqs):
            req = int(idx_cpu[b])
            s = int(qsl_cpu[b])
            e = int(qsl_cpu[b + 1])
            if e <= s:
                continue
            p = pos_dev[s:e]
            block_indices = p // vbs
            block_offsets = p - block_indices * vbs
            block_numbers = block_table[
                req, block_indices.to(torch.int64)
            ].to(torch.int32)
            if self.cp_size == 1 and self.cp_interleave == 1:
                local_offsets = block_offsets
                is_local = torch.ones_like(block_offsets, dtype=torch.bool)
            else:
                is_local = (
                    (block_offsets // self.cp_interleave) % self.cp_size
                    == self.cp_rank
                )
                local_offsets = (
                    block_offsets // (self.cp_interleave * self.cp_size)
                ) * self.cp_interleave + (block_offsets % self.cp_interleave)
            slots = block_numbers * bs + local_offsets
            slots = torch.where(
                is_local, slots, torch.full_like(slots, PAD_SLOT_ID)
            )
            row_slots[s:e] = slots.to(row_slots.dtype)
    return self.slot_mappings[:, :num_tokens_padded]


def apply_lpu_block_table_patch() -> None:
    """Replace BlockTable.compute_slot_mapping Triton kernels with PyTorch on LPU.

    The active class is ``vllm.v1.worker.gpu.block_table.BlockTables``
    (``compute_slot_mappings``, multi-group); the single-group
    ``vllm.v1.worker.block_table.BlockTable.compute_slot_mapping`` still exists
    as a compat shim.  Both are rebound so the Triton kernels never run on LPU.
    Also installs the dirty-flag skip on ``commit_block_table`` so the per-step
    block-table copy is elided when no row has changed.
    """
    global _PATCHED
    if _PATCHED:
        return
    global _ORIGINAL_COMPUTE_SLOT_MAPPING
    _ORIGINAL_COMPUTE_SLOT_MAPPING = _bt_mod.BlockTable.compute_slot_mapping
    _bt_mod.BlockTable.compute_slot_mapping = _compute_slot_mapping_pytorch
    try:
        import vllm.v1.worker.gpu.block_table as _bt024_mod
        _bt024_mod.BlockTables.compute_slot_mappings = (
            _compute_slot_mappings_024_pytorch
        )
    except Exception:
        pass
    _apply_block_table_dirty_flag()
    _PATCHED = True

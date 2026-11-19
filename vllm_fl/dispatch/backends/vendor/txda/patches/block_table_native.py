# Copyright (c) 2026 BAAI. All rights reserved.

"""Native replacements for the V2 block-table / slot-mapping kernels on TXDA.

``vllm/v1/worker/gpu/block_table.py`` addresses the per-group block-table
tensors through ``_load_ptr``: the kernels ``tl.load`` a ``data_ptr()`` stored
in a ``uint64`` tensor and cast the value to a pointer. The TXDA Triton
backend miscompiles that pattern — the load reads address 0 — so:

* ``_gather_block_tables_kernel`` writes zeros into every gathered block
  table, and
* ``_compute_slot_mappings_kernel`` resolves every token to KV slot 0.

With all tokens written to one slot the attention output is garbage and the
logits come out NaN (vLLM aborts with ``NaNs detected in logits``). This is
the V2 analogue of the block-table translation bug fixed for the V1 runner.

The replacements keep the same contract but index the Python-side tensors
directly, which TXDA executes correctly (the buffers themselves are plain
``[max_num_reqs, max_num_blocks]`` int32 CUDA tensors).

Removal condition: delete this module once the TXDA Triton backend supports
casting loaded integers to pointers.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_installed = False


def gather_block_tables(
    num_kv_cache_groups,
    input_block_tables,
    block_tables,
    num_blocks,
    idx_mapping,
    num_reqs_padded,
    out=None,
) -> tuple:
    """Pure-torch ``_gather_block_tables_kernel``.

    Destination rows are *batch* rows; source rows are request-state slots
    (``idx_mapping``). The two differ whenever slots are not handed out in
    batch order, so the source and destination indices must be distinct.
    """
    if num_kv_cache_groups == 0:
        return ()
    if out is None:
        out = tuple(input_block_tables)
    num_reqs = idx_mapping.shape[0]
    slots = idx_mapping[:num_reqs].to(torch.int64)
    batch_rows = torch.arange(num_reqs, device=slots.device)
    valid = slots >= 0
    if not bool(valid.all()):
        slots = slots[valid]
        batch_rows = batch_rows[valid]
    for group_id in range(num_kv_cache_groups):
        src = block_tables[group_id]
        dst = out[group_id]
        # Padded rows are zeroed, matching the kernel.
        dst[:num_reqs_padded].zero_()
        if slots.numel() == 0:
            continue
        counts = num_blocks[group_id, slots].to(torch.int64)
        width = min(int(counts.max().item()), dst.shape[1])
        if width == 0:
            continue
        columns = torch.arange(width, device=dst.device)
        entries = src[slots][:, :width]
        # Exactly ``num_blocks`` ids per row: entries past a request's own
        # block count hold stale ids whose blocks may already belong to
        # another request.
        live = columns[None, :] < counts[:, None]
        dst[batch_rows[:, None], columns[None, :]] = torch.where(
            live, entries, torch.zeros_like(entries)
        )
    return tuple(bt[:num_reqs_padded] for bt in out)


def compute_slot_mappings(
    num_kv_cache_groups,
    slot_mappings,
    block_sizes,
    slot_mapping_enabled,
    block_tables,
    idx_mapping,
    query_start_loc,
    positions,
    num_tokens_padded,
    pad_slot_id,
    out=None,
):
    """Pure-torch ``_compute_slot_mappings_kernel`` (CP == 1 path)."""
    if num_kv_cache_groups == 0:
        return (slot_mappings if out is None else out)[:, :num_tokens_padded]
    result = slot_mappings if out is None else out
    result[:, :num_tokens_padded].fill_(pad_slot_id)
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return result[:, :num_tokens_padded]

    slots = idx_mapping[:num_reqs].to(torch.int64)
    lengths = (
        query_start_loc[1 : num_reqs + 1].to(torch.int64)
        - query_start_loc[:num_reqs].to(torch.int64)
    )
    total = int(lengths.sum().item())
    if total == 0:
        return result[:, :num_tokens_padded]
    token_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=idx_mapping.device), lengths
    )
    req_slots = slots[token_ids]
    pos = positions[:total].to(torch.int64)
    real = req_slots >= 0
    all_real = bool(real.all())
    for group_id in range(num_kv_cache_groups):
        if not slot_mapping_enabled[group_id]:
            continue
        block_size = int(block_sizes[group_id])
        block_indices = torch.div(pos, block_size, rounding_mode="floor")
        block_offsets = pos % block_size
        table = block_tables[group_id]
        block_numbers = torch.zeros_like(block_indices)
        if all_real:
            block_numbers = table[req_slots, block_indices].to(torch.int64)
        elif bool(real.any()):
            block_numbers[real] = table[
                req_slots[real], block_indices[real]
            ].to(torch.int64)
        slot_ids = block_numbers * block_size + block_offsets
        if not all_real:
            slot_ids = torch.where(
                real, slot_ids, torch.full_like(slot_ids, pad_slot_id)
            )
        result[group_id, :total] = slot_ids
    return result[:, :num_tokens_padded]


def apply_staged_write(
    gpu: torch.Tensor,
    indices: list[int],
    starts: list[int],
    contents: list[int | float],
    cu_lens: list[int],
) -> None:
    """Pure-torch ``StagedWriteTensor.apply_write``.

    The kernel flattened the target and wrote ``index * stride(0) + start``,
    so a 1-D tensor addresses element ``index`` and a 2-D block table
    addresses ``gpu[index, start:start+n]``.
    """
    flat = gpu.reshape(-1)
    stride0 = gpu.stride(0) if gpu.dim() else 1
    prev = 0
    for index, start, cu in zip(indices, starts, cu_lens):
        values = contents[prev:cu]
        prev = cu
        if not values:
            continue
        offset = index * stride0 + start
        flat[offset : offset + len(values)] = flat.new_tensor(values)


def install_block_table_native() -> None:
    """Replace the V2 block-table/slot-mapping kernels with native indexing."""
    global _installed
    if _installed:
        return
    _installed = True

    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu import block_table as bt_mod

    cls = bt_mod.BlockTables
    if getattr(cls, "_fl_txda_native_block_tables", False):
        return

    def _gather(self, idx_mapping, num_reqs_padded, out=None, out_ptrs=None):
        return gather_block_tables(
            self.num_kv_cache_groups,
            self.input_block_tables,
            [bt.gpu for bt in self.block_tables],
            self.num_blocks.gpu,
            idx_mapping,
            num_reqs_padded,
            out=out,
        )

    def _slot_mappings(
        self, idx_mapping, query_start_loc, positions, num_tokens_padded, out=None
    ):
        return compute_slot_mappings(
            self.num_kv_cache_groups,
            self.slot_mappings,
            self.block_sizes,
            self._slot_mapping_enabled,
            [bt.gpu for bt in self.block_tables],
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded,
            PAD_SLOT_ID,
            out=out,
        )

    def _apply_write(self, *args, **kwargs):
        if not self._staged_write_indices:
            return
        apply_staged_write(
            self.gpu,
            self._staged_write_indices,
            self._staged_write_starts,
            self._staged_write_contents,
            self._staged_write_cu_lens,
        )
        self.clear_staged_writes()

    def _fused_apply(self, tensors, output_ptrs, output_strides):
        # Writing each tensor natively is equivalent to the fused kernel: every
        # staged entry targets its own row of its own tensor.
        for t in tensors:
            t.apply_write()

    cls.gather_block_tables = _gather
    cls.compute_slot_mappings = _slot_mappings
    cls._fl_txda_native_block_tables = True

    # ``StagedWriteTensor.apply_write`` / ``FusedStagedWriter.apply`` have the
    # same defect: the staged block ids never reach the block table, so every
    # request keeps an all-zero block table and every token maps to slot 0.
    from vllm.v1.worker.gpu import buffer_utils as bu

    bu.StagedWriteTensor.apply_write = _apply_write
    bu.FusedStagedWriter.apply = _fused_apply

    logger.info("Installed native TXDA block-table gather / slot mapping")

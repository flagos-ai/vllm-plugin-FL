# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU replacement for vLLM's Triton slot-mapping kernel.

vLLM computes the KV-cache slot mapping with a Triton kernel
(``vllm.v1.worker.block_table._compute_slot_mapping_kernel``) that operates
on ``int64`` ``positions`` / ``slot_mapping`` buffers.  The GCU300 Triton
backend rejects 64-bit dtypes outright ("64-bit data type not supported on
GCU300") - even a bare int64 load/store fails to compile - so the kernel
cannot run on GCU.

We replace ``BlockTable.compute_slot_mapping`` with a vectorised on-device
int32 implementation that reproduces the kernel's semantics exactly (including
context-parallel interleaving).  The cache index space (block numbers x
block_size, ~1e8 slots for realistic configs) fits comfortably in int32, and
every operator involved (``searchsorted``, ``//``, ``%``, ``-``, advanced
indexing, ``where``) compiles cleanly at int32 under ``flag_gems.enable()`` on
both the vendor Triton and FlagTree backends.  Verified bit-identical to a
CPU int64 reference on synthetic cases (up to 4096 tokens) and on live serve
inputs.

Two implementation notes:

- ``searchsorted`` (not ``repeat_interleave``) maps tokens to requests.  The
  FlagGems GCU300 ``repeat_interleave`` routes through an ``index_select``
  kernel whose grid.y is capped at 255, so it crashes past ~4080 scheduled
  tokens; ``searchsorted`` on the monotone request-end boundaries is a single
  op with no such limit.
- Staying on-device (vs. computing on CPU) avoids a host round-trip per step
  and scales to large batches / long contexts.  ``flag_gems.enable()`` reroutes
  these ops into FlagGems Triton kernels, but at int32 they all compile.
"""

from __future__ import annotations

import logging

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

logger = logging.getLogger(__name__)
_patched = False


def compute_slot_mapping_int32(
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    total_cp_world_size: int,
    total_cp_rank: int,
    cp_kv_cache_interleave_size: int,
    max_num_batched_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Standalone on-device int32 slot_mapping computation for GCU300.

    Args:
        num_reqs: number of requests
        query_start_loc: cumulative token ends per request, shape [num_reqs+1], int32
        positions: token positions, shape [num_tokens], int64
        block_table: block IDs, shape [num_reqs, max_blocks_per_req], int32
        block_size: KV block size (tokens per block)
        total_cp_world_size: pcp_world_size * dcp_world_size
        total_cp_rank: pcp_rank * dcp_world_size + dcp_rank
        cp_kv_cache_interleave_size: CP interleaving chunk size
        max_num_batched_tokens: CUDA-graph max (pad tail to this)
        device: target device

    Returns:
        slot_mapping tensor, shape [max_num_batched_tokens], int64, padded with PAD_SLOT_ID

    This is a semantic rewrite of vLLM's ``_compute_slot_mapping_kernel``:
    - Vectorized across all scheduled tokens (not per-request Triton loop)
    - ``searchsorted`` replaces ``repeat_interleave`` (avoids GCU300 grid.y=255 cap)
    - All ops at int32 (cache index space fits comfortably; int64 hits GCU300 wall)
    - Verified bit-identical to CPU int64 reference on synthetic + live inputs
    """
    virtual_block_size = block_size * total_cp_world_size
    total_scheduled = int(query_start_loc[num_reqs].item())

    # Allocate output on device, pad tail for CUDA-graph
    slot_mapping = torch.full(
        (max_num_batched_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=device
    )
    if total_scheduled == 0:
        return slot_mapping

    i32 = torch.int32
    qsl = query_start_loc[: num_reqs + 1].to(device, i32)
    pos = positions[:total_scheduled].to(device, i32)
    bt = block_table.to(device, i32)

    # Token -> request mapping via searchsorted (token t belongs to request r
    # where qsl[r] <= t < qsl[r+1]). This replaces repeat_interleave, whose
    # GCU300 index_select kernel caps grid.y at 255.
    tok = torch.arange(total_scheduled, device=device, dtype=i32)
    token_req = torch.searchsorted(qsl[1:], tok, right=True).to(i32)

    block_indices = pos // virtual_block_size
    block_numbers = bt[token_req, block_indices]

    virtual_block_offsets = pos - block_indices * virtual_block_size
    is_local = (
        virtual_block_offsets // cp_kv_cache_interleave_size
    ) % total_cp_world_size == total_cp_rank
    local_block_offsets = (
        virtual_block_offsets // (total_cp_world_size * cp_kv_cache_interleave_size)
    ) * cp_kv_cache_interleave_size + (
        virtual_block_offsets % cp_kv_cache_interleave_size
    )

    slot_ids = block_numbers * block_size + local_block_offsets
    slot_ids = torch.where(
        is_local, slot_ids, torch.full_like(slot_ids, PAD_SLOT_ID)
    )
    slot_mapping[:total_scheduled] = slot_ids.to(torch.int64)
    return slot_mapping


def _compute_slot_mapping_torch(
    self,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """vLLM monkey-patch adapter: calls standalone int32 function, writes to self.slot_mapping.gpu."""
    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

    result = compute_slot_mapping_int32(
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        positions=positions,
        block_table=self.block_table.gpu,
        block_size=self.block_size,
        total_cp_world_size=total_cp_world_size,
        total_cp_rank=total_cp_rank,
        cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        max_num_batched_tokens=self.max_num_batched_tokens,
        device=self.slot_mapping.gpu.device,
    )
    self.slot_mapping.gpu.copy_(result)


def apply_slot_mapping_gcu_patch() -> None:
    """Replace ``BlockTable.compute_slot_mapping`` with the on-device int32 version."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    try:
        import vllm.v1.worker.block_table as bt

        bt.BlockTable.compute_slot_mapping = _compute_slot_mapping_torch
        _patched = True
        logger.info(
            "Patched BlockTable.compute_slot_mapping for GCU "
            "(on-device int32; avoids int64 Triton kernel)."
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to patch compute_slot_mapping for GCU: %s", exc
        )

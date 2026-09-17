# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""GCU300 slot_mapping patch.

vLLM's native ``BlockTable.compute_slot_mapping`` launches
``_compute_slot_mapping_kernel``, a triton kernel that operates on int64
``positions`` / ``slot_mapping`` and has hardcoded ``.to(tl.int64)`` casts.
GCU300's triton_gcu compiler rejects 64-bit IR, so the kernel fails to compile.

This module replaces ``BlockTable.compute_slot_mapping`` with a pure-torch,
fully **on-device int32** reimplementation. The cache-slot index space
(num_blocks * block_size, ~1e8 in realistic configs) fits comfortably in int32
(~2.1e9), so every intermediate stays int32 and never touches the int64 wall.

Semantics mirror the native kernel exactly (see vllm/v1/worker/block_table.py):
per request r covering tokens [start, end):
    block_idx        = pos // (block_size * TOTAL_CP_WORLD_SIZE)
    vblock_offset    = pos - block_idx * (block_size * TOTAL_CP_WORLD_SIZE)
    is_local         = (vblock_offset // INTERLEAVE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
    local_offset     = (vblock_offset // (TOTAL_CP_WORLD_SIZE * INTERLEAVE)) * INTERLEAVE
                       + (vblock_offset % INTERLEAVE)
    slot             = block_table[r, block_idx] * block_size + local_offset
    slot             = where(is_local, slot, PAD_ID)
Padding tokens [num_tokens, max_num_batched_tokens) are set to PAD_ID.

``torch.searchsorted`` maps each token to its request index (instead of
``repeat_interleave``, which on GCU300 flag_gems routes through an index_select
kernel with a grid.y=255 limit that breaks above ~4080 tokens).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _compute_slot_mapping_int32(self, num_reqs, query_start_loc, positions):
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    device = self.slot_mapping.gpu.device
    num_tokens = positions.shape[0]
    max_num = self.max_num_batched_tokens
    block_size = self.block_size
    cp_world = self.pcp_world_size * self.dcp_world_size
    cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    interleave = self.cp_kv_cache_interleave_size

    slot_out = self.slot_mapping.gpu  # int64 buffer [max_num_batched_tokens]

    # No scheduled tokens: pad everything and return.
    if num_tokens == 0:
        slot_out[:max_num].fill_(PAD_SLOT_ID)
        return

    # qsl[:num_reqs+1] are request-token boundaries (int32 in vLLM).
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)

    # Map each token index -> its request via searchsorted on the end boundaries.
    # token t belongs to request r where qsl[r] <= t < qsl[r+1].
    tok = torch.arange(num_tokens, device=device, dtype=torch.int32)
    # right=True on qsl[1:] gives the count of ends <= t ... use qsl[1:] as ends.
    req_idx = torch.searchsorted(qsl[1:], tok, right=True).to(torch.int32)
    req_idx = torch.clamp(req_idx, max=num_reqs - 1)

    pos = positions[:num_tokens].to(torch.int32)

    vblock = block_size * cp_world
    block_idx = pos // vblock
    vblock_offset = pos - block_idx * vblock

    # block_table.gpu is int32 [max_num_reqs, max_num_blocks_per_req]
    bt = self.block_table.gpu
    block_numbers = bt[req_idx, block_idx].to(torch.int32)

    if cp_world == 1:
        local_offset = vblock_offset
        is_local = torch.ones_like(pos, dtype=torch.bool)
    else:
        is_local = (
            (vblock_offset // interleave) % cp_world
        ) == cp_rank
        local_offset = (
            vblock_offset // (cp_world * interleave)
        ) * interleave + (vblock_offset % interleave)

    slot_ids = block_numbers * block_size + local_offset
    slot_ids = torch.where(
        is_local, slot_ids, torch.full_like(slot_ids, PAD_SLOT_ID)
    )

    # Write results (cast to the int64 buffer dtype at the boundary) + pad tail.
    slot_out[:num_tokens] = slot_ids.to(slot_out.dtype)
    if max_num > num_tokens:
        slot_out[num_tokens:max_num].fill_(PAD_SLOT_ID)


def apply_slot_mapping_gcu_patch() -> None:
    """Replace BlockTable.compute_slot_mapping with the on-device int32 version."""
    from vllm.v1.worker.block_table import BlockTable

    if BlockTable.compute_slot_mapping is _compute_slot_mapping_int32:
        return

    BlockTable.compute_slot_mapping = _compute_slot_mapping_int32
    logger.info("GCU: patched BlockTable.compute_slot_mapping (on-device int32)")

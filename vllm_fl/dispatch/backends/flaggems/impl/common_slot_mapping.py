# Copyright (c) 2026 BAAI. All rights reserved.
"""Framework adapter for FlagGems common slot-mapping metadata kernels."""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


def compute_common_slot_mapping_flaggems(
    block_table: Any,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    """Unpack vLLM cache groups for the FlagGems tensor-level API."""
    from flag_gems.fused import compute_common_slot_mapping

    for index, table in enumerate(block_table.block_tables):
        total_cp_world_size = table.pcp_world_size * table.dcp_world_size
        total_cp_rank = table.pcp_rank * table.dcp_world_size + table.dcp_rank
        compute_common_slot_mapping(
            table.block_table.gpu,
            table.slot_mapping.gpu,
            num_reqs,
            query_start_loc,
            positions,
            seq_lens,
            num_computed_tokens,
            max_num_batched_tokens=table.max_num_batched_tokens,
            block_size=table.block_size,
            total_cp_world_size=total_cp_world_size,
            total_cp_rank=total_cp_rank,
            cp_kv_cache_interleave_size=table.cp_kv_cache_interleave_size,
            null_block_id=NULL_BLOCK_ID,
            pad_slot_id=PAD_SLOT_ID,
            update_num_computed_tokens=index == 0,
        )

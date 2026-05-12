# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems fused moe operator implementations.
"""

from typing import Optional

import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def moe_align_block_size_flaggems(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flag_gems import moe_align_block_size_triton

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    # TODO(lms): ignore_invalid_experts not effective now
    # moe_align_block_size has optimize version to filtered out
    # all invalid experts directly when counting the number of experts
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def topk_softmax_flaggems(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    from flag_gems import topk_softmax

    try:
        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
    except:
        topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices


def moe_sum_flaggems(inp, out):
    from flag_gems import moe_sum

    moe_sum(inp, out)

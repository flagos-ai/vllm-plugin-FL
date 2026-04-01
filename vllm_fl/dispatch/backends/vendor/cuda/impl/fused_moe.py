# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA rotary embedding operator implementations.
"""

import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def moe_align_block_size_cuda(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm._custom_ops import moe_align_block_size

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

    moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def topk_softmax_cuda(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    from vllm._custom_ops import topk_softmax

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )
    return topk_weights, topk_indices


def moe_sum_cuda(inp, out):
    from vllm._custom_ops import moe_sum

    moe_sum(inp, out)

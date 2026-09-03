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


### TODO(lms): support rocm
def fused_topk_bias_flaggems(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FlagGems implementation of fused_topk_bias.

    Uses FlagGems fused kernels:
    - topk_softmax for softmax scoring path
    - topk_softplus_sqrt for sqrtsoftplus scoring path
    Falls back to PyTorch for sigmoid (no fused FlagGems kernel yet).
    """
    M = hidden_states.size(0)
    assert M == gating_output.size(0), "Number of tokens mismatch"
    device = hidden_states.device
    ids_dtype = torch.int32 if indices_type is None else indices_type

    def _alloc_topk_buffers():
        w = torch.empty(M, topk, dtype=torch.float32, device=device)
        ids = torch.empty(M, topk, dtype=ids_dtype, device=device)
        tei = torch.empty(M, topk, dtype=torch.int32, device=device)
        return w, ids, tei

    if scoring_func == "softmax":
        topk_weights, topk_ids, token_expert_indices = _alloc_topk_buffers()

        topk_weights, topk_ids = topk_softmax_flaggems(
            topk_weights, topk_ids, token_expert_indices,
            gating_output, renormalize,
        )

        if e_score_correction_bias is not None:
            # topk_softmax doesn't support bias natively, re-select with bias
            scores = gating_output.softmax(dim=-1)
            scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
            topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)[1].to(ids_dtype)
            topk_weights = scores.gather(1, topk_ids.to(torch.int64)).to(torch.float32)
            if renormalize:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        if routed_scaling_factor != 1.0:
            topk_weights *= routed_scaling_factor
        return topk_weights, topk_ids

    elif scoring_func == "sqrtsoftplus":
        from flag_gems import topk_softplus_sqrt

        topk_weights, topk_ids, token_expert_indices = _alloc_topk_buffers()

        topk_softplus_sqrt(
            topk_weights, topk_ids, token_expert_indices,
            gating_output, renormalize, routed_scaling_factor,
            e_score_correction_bias, input_tokens, hash_indices_table,
        )
        return topk_weights, topk_ids

    elif scoring_func == "sigmoid":
        # No fused FlagGems kernel for sigmoid yet, use PyTorch fallback
        scores = gating_output.sigmoid()
        n_routed_experts = gating_output.shape[-1]

        if e_score_correction_bias is not None:
            scores_for_choice = scores.view(
                -1, n_routed_experts
            ) + e_score_correction_bias.unsqueeze(0)
        else:
            scores_for_choice = scores.view(-1, n_routed_experts)

        if hash_indices_table is not None:
            topk_indices = hash_indices_table[input_tokens]
        else:
            topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1)[1]

        topk_weights = scores.gather(1, topk_indices).to(torch.float32)
        if renormalize:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        if routed_scaling_factor != 1.0:
            topk_weights *= routed_scaling_factor
        return topk_weights, topk_indices.to(ids_dtype)

    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


def invoke_fused_moe_triton_kernel_flaggems(
    A,
    B,
    C,
    A_scale,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    config,
    compute_type,
    use_fp8_w8a8,
    use_int8_w8a8,
    use_int8_w8a16,
    use_int4_w4a16,
    per_channel_quant,
    block_shape=None,
    B_bias=None,
):
    from flag_gems import invoke_fused_moe_triton_kernel

    invoke_fused_moe_triton_kernel(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=block_shape,
        B_bias=B_bias,
    )


def grouped_topk_flaggems(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    from flag_gems import grouped_topk

    return grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def moe_sum_flaggems(inp, out):
    from flag_gems import moe_sum

    moe_sum(inp, out)

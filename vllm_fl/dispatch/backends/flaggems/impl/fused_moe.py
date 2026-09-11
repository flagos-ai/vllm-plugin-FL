# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems fused moe operator implementations.
"""

from typing import Optional

import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def fused_marlin_moe_flaggems(*args, **kwargs):
    """Call FlagGems' vLLM-compatible fused Marlin MoE implementation."""
    # FlagGems PR #5437 follows the vLLM API except for the newer GLM-4
    # activation coefficients. Falling back is safer than silently dropping a
    # non-default coefficient for models that use them.
    gemm1_alpha = kwargs.pop("gemm1_alpha", 1.0)
    gemm1_beta = kwargs.pop("gemm1_beta", 0.0)
    if gemm1_alpha != 1.0 or gemm1_beta != 0.0:
        raise NotImplementedError(
            "FlagGems fused_marlin_moe does not support gemm1_alpha/beta"
        )

    # FlagGems PR #5437 copied vLLM 0.20's small sequential scalar-type IDs.
    # vLLM 0.24 encodes the scalar type properties into a stable integer, so
    # translate the three formats implemented by that PR at the adapter edge.
    from vllm.scalar_type import scalar_types

    quant_type_id = kwargs.get("quant_type_id")
    quant_type_map = {
        scalar_types.uint4b8.id: 0,
        scalar_types.uint8b128.id: 1,
        scalar_types.float4_e2m1f.id: 6,
    }
    if quant_type_id in quant_type_map:
        kwargs["quant_type_id"] = quant_type_map[quant_type_id]

    from flag_gems.fused.fused_marlin_moe import fused_marlin_moe

    return fused_marlin_moe(*args, **kwargs)


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

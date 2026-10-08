# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems fused moe operator implementations.
"""

import flag_gems
import torch
from flag_gems import (
    grouped_topk as _grouped_topk,
    invoke_fused_moe_triton_kernel as _invoke_fused_moe_triton_kernel,
    moe_align_block_size_triton as _moe_align_block_size_triton,
    moe_sum as _moe_sum,
    topk_softmax as _topk_softmax,
    topk_softplus_sqrt as _topk_softplus_sqrt,
)
from flag_gems.pt2.fused_moe import (
    moe_sum as _pt2_moe_sum,
    topk_softmax as _pt2_topk_softmax,
)
from flag_gems.pt2.moe_routing import (
    grouped_topk as _pt2_grouped_topk,
    uses_common_moe_routing_kernels as _uses_common_moe_routing_kernels,
)

from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

# These contracts capture the common/NVIDIA kernel objects.  Other vendors
# may replace either public FlagGems export, so retain their original path
# until an equivalent vendor-specific PT2 contract is validated.
_USE_TRANSPARENT_MOE_PRIMITIVES = flag_gems.vendor_name == "nvidia"
_USE_TRANSPARENT_MOE_ROUTING = (
    flag_gems.vendor_name == "nvidia"
    and _uses_common_moe_routing_kernels(
        _grouped_topk,
        _topk_softplus_sqrt,
    )
)


def moe_align_block_size_flaggems(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    _moe_align_block_size_triton(
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
    # The FlagGems API accepts ``renormalize`` directly; a frozen compiled
    # path cannot switch implementations after an exception, and Dynamo cannot
    # soundly trace try/retry control flow.  Both eager and compile call the
    # same kernel with the same arguments.
    if _USE_TRANSPARENT_MOE_PRIMITIVES:
        _pt2_topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
    else:
        _topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
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
    _invoke_fused_moe_triton_kernel(
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
    grouped_topk_impl = (
        _pt2_grouped_topk if _USE_TRANSPARENT_MOE_ROUTING else _grouped_topk
    )
    return grouped_topk_impl(
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
    if _USE_TRANSPARENT_MOE_PRIMITIVES:
        _pt2_moe_sum(inp, out)
    else:
        _moe_sum(inp, out)

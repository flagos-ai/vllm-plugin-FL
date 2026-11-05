# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems fused moe operator implementations.
"""

import torch
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def moe_align_block_size_flaggems(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk_ids.device.type == "gcu":
        # FlagGems-vllm's generic four-stage kernel uses atomic RMW operations
        # unsupported by the pinned GCU compiler. Keep the official FlagGems
        # GCU implementation until the new provider has a native implementation
        # that passes the align value/padding and graph replay regressions.
        from flag_gems import moe_align_block_size_triton
    else:
        from flaggems_vllm import moe_align_block_size_triton

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
    from flaggems_vllm import topk_softmax

    # FlagGems' fused renormalization corrupts the first token on Ascend when
    # the routed token count exceeds 48. The non-renormalizing kernel is
    # numerically correct, so normalize its selected weights with PyTorch.
    normalize_after_kernel = renormalize and gating_output.device.type == "npu"
    kernel_renormalize = renormalize and not normalize_after_kernel
    try:
        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            kernel_renormalize,
        )
    except TypeError:
        topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
        normalize_after_kernel = renormalize
    if normalize_after_kernel:
        denominator = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
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
    from flaggems_vllm import invoke_fused_moe_triton_kernel

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
    from flaggems_vllm import grouped_topk

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
    if inp.device.type == "gcu" and inp.is_contiguous() and out.is_contiguous():
        # FlagGems-vllm's MT kernel puts the token count on grid.y, which can
        # exceed GCU's 255-block limit. Keep its input contract before using
        # the official GCU kernel; retain the stride-aware provider below.
        # Remove this branch once the pinned provider supports GCU launch
        # limits and passes large-token value and graph-replay checks.
        from flaggems_vllm.ops.moe_sum import _check_moe_sum_inputs

        _check_moe_sum_inputs(inp, out)
        from flag_gems import moe_sum

        moe_sum(inp, out)
        return

    try:
        from flaggems_vllm import moe_sum

        moe_sum(inp, out)
        return
    except Exception:
        # MetaX C550: flaggems-vllm ships no working moe_sum kernel for MACA,
        # so dispatch reports "All implementations for op='moe_sum' have
        # failed". The upstream flag_gems kernel is the validated path on this
        # vendor; keep the failure visible everywhere else.
        import flag_gems

        if str(getattr(flag_gems, "vendor_name", "")).lower() != "metax":
            raise
        logger.debug(
            "flaggems_vllm.moe_sum failed on MetaX; falling back to flag_gems",
            exc_info=True,
        )

    from flag_gems import moe_sum as flag_gems_moe_sum

    flag_gems_moe_sum(inp, out)

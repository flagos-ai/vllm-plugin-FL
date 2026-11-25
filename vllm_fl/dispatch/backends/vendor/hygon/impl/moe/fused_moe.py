# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon MoE operator fallbacks backed by vLLM native ops."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def moe_align_block_size_hygon(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from ..native_ops.moe_align_block_size import moe_align_block_size_hygon_out

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
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

    moe_align_block_size_hygon_out(
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


def moe_sum_hygon(inp, out):
    from ..native_ops.moe_sum import moe_sum_hygon_out

    moe_sum_hygon_out(inp, out)


def topk_softmax_hygon(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
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


def invoke_fused_moe_triton_kernel_hygon(
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
    from .bf16_moe_gemm2 import (
        HygonBf16MoeGemm2Config,
        invoke_hygon_bf16_moe_gemm2,
        supports_hygon_bf16_moe_gemm2,
    )

    if supports_hygon_bf16_moe_gemm2(
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        top_k,
        config,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        B_bias,
    ):
        logger.info_once("Using Hygon BF16 optimized MoE GEMM2 for 1 <= M <= 128")
        invoke_hygon_bf16_moe_gemm2(
            A,
            B,
            C,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=mul_routed_weight,
            config=HygonBf16MoeGemm2Config.for_small_decode(config),
        )
        return

    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
        invoke_fused_moe_triton_kernel_flaggems,
    )

    flaggems_config = {
        key: value for key, value in config.items() if key != "workers_per_cu"
    }
    invoke_fused_moe_triton_kernel_flaggems(
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
        flaggems_config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=block_shape,
        B_bias=B_bias,
    )


def grouped_topk_hygon(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    from vllm._custom_ops import grouped_topk

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

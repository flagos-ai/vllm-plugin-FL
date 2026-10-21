# SPDX-License-Identifier: Apache-2.0
"""Exact SwiGLU+INT8 activation fusion for vLLM's Marlin MoE path."""

from __future__ import annotations

from collections.abc import Callable

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.scalar_type import ScalarType, scalar_types

from vllm_fl.ops.deepseek_v4_silu_int8 import (
    silu_mul_dynamic_int8,
)

_ORIGINAL_INTERNAL = marlin_moe._fused_marlin_moe


def _is_plain_activation_callback(activation_func: Callable[..., None]) -> bool:
    """Accept vLLM's free function and its non-LoRA modular bound method."""
    if activation_func is apply_moe_activation:
        return True
    return (
        getattr(activation_func, "__self__", None) is not None
        and getattr(activation_func, "__name__", "") == "activation"
    )


def _fused_marlin_moe_int8_activation(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    num_topk: int,
    quant_type: ScalarType,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None,
    block_size_m: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    activation: MoEActivation = MoEActivation.SILU,
    activation_func: Callable[..., None] = apply_moe_activation,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    is_k_full: bool = True,
    clamp_limit: float | None = None,
    gemm1_alpha: float = 1.0,
    gemm1_beta: float = 0.0,
) -> torch.Tensor:
    use_fused_activation = (
        input_dtype == torch.int8
        and activation == MoEActivation.SILU
        and clamp_limit is None
        and gemm1_alpha == 1.0
        and gemm1_beta == 0.0
    )
    if not use_fused_activation:
        return _ORIGINAL_INTERNAL(
            hidden_states,
            w1,
            w2,
            bias1,
            bias2,
            w1_scale,
            w2_scale,
            topk_weights,
            num_topk,
            quant_type,
            apply_router_weight_on_input,
            expert_map,
            block_size_m,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            activation,
            activation_func,
            input_global_scale1,
            input_global_scale2,
            global_scale1,
            global_scale2,
            g_idx1,
            g_idx2,
            sort_indices1,
            sort_indices2,
            w1_zeros,
            w2_zeros,
            workspace,
            intermediate_cache13,
            intermediate_cache2,
            output,
            input_dtype,
            is_k_full,
            clamp_limit,
            gemm1_alpha,
            gemm1_beta,
        )

    assert hidden_states.ndim == 2
    m, k = hidden_states.size()
    n = marlin_moe.marlin_moe_intermediate_size(w1, w2)
    num_shards = 2
    if workspace is None:
        workspace = marlin_moe.marlin_make_workspace_new(hidden_states.device, 4)
    if intermediate_cache13 is None:
        intermediate_cache13 = torch.empty(
            (m * num_topk * max(num_shards * n, k),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    if intermediate_cache2 is None:
        intermediate_cache2 = torch.empty(
            (m * num_topk, n),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    cache1 = marlin_moe._resize_cache(
        intermediate_cache13, (m * num_topk, num_shards * n)
    )
    cache3 = marlin_moe._resize_cache(intermediate_cache13, (m * num_topk, k))

    gate_up_input, scales1 = marlin_moe.marlin_quant_input(hidden_states, input_dtype)
    if input_global_scale1 is not None:
        scales1 = scales1 * input_global_scale1

    cache1 = ops.moe_wna16_marlin_gemm(
        gate_up_input,
        cache1,
        w1,
        bias1,
        w1_scale,
        scales1,
        global_scale1,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=num_topk,
        mul_topk_weights=apply_router_weight_on_input,
        b_q_type=quant_type,
        size_m=m,
        size_n=num_shards * n,
        size_k=k,
        is_k_full=is_k_full,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    quantized_activation, scales2 = silu_mul_dynamic_int8(
        cache1.view(-1, num_shards * n)
    )
    if input_global_scale2 is not None:
        scales2 = scales2 * input_global_scale2

    if output is None:
        output = cache3
    if expert_map is not None:
        output.zero_()

    output = ops.moe_wna16_marlin_gemm(
        quantized_activation,
        output,
        w2,
        bias2,
        w2_scale,
        scales2,
        global_scale2,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=not apply_router_weight_on_input,
        b_q_type=quant_type,
        size_m=m * num_topk,
        size_n=k,
        size_k=n,
        is_k_full=is_k_full,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    return output


def fused_marlin_moe_with_int8_activation(
    native: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: MoEActivation = MoEActivation.SILU,
    activation_func: Callable[..., None] = apply_moe_activation,
    moe_sum: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    expert_map: torch.Tensor | None = None,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    is_k_full: bool = True,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    clamp_limit: float | None = None,
    gemm1_alpha: float = 1.0,
    gemm1_beta: float = 0.0,
) -> torch.Tensor:
    """Use the fused activation only for its exact supported Marlin case.

    This mirrors vLLM's public ``fused_marlin_moe`` wrapper, but directly
    selects the optimized internal implementation.  In particular, it avoids
    replacing ``marlin_moe._fused_marlin_moe`` process-wide while a request is
    running.
    """
    use_fused_activation = (
        input_dtype == torch.int8
        and activation == MoEActivation.SILU
        and _is_plain_activation_callback(activation_func)
        and clamp_limit is None
        and gemm1_alpha == 1.0
        and gemm1_beta == 0.0
    )
    if not use_fused_activation:
        return native(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            bias1=bias1,
            bias2=bias2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=quant_type_id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            activation=activation,
            activation_func=activation_func,
            moe_sum=moe_sum,
            expert_map=expert_map,
            input_global_scale1=input_global_scale1,
            input_global_scale2=input_global_scale2,
            global_scale1=global_scale1,
            global_scale2=global_scale2,
            g_idx1=g_idx1,
            g_idx2=g_idx2,
            sort_indices1=sort_indices1,
            sort_indices2=sort_indices2,
            w1_zeros=w1_zeros,
            w2_zeros=w2_zeros,
            workspace=workspace,
            intermediate_cache13=intermediate_cache13,
            intermediate_cache2=intermediate_cache2,
            is_k_full=is_k_full,
            output=output,
            input_dtype=input_dtype,
            clamp_limit=clamp_limit,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
        )

    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type in (
        scalar_types.uint4,
        scalar_types.uint8b128,
        scalar_types.uint4b8,
        scalar_types.float8_e4m3fn,
        scalar_types.float4_e2m1f,
    )
    num_bits = (
        4
        if quant_type
        in (
            scalar_types.uint4,
            scalar_types.uint4b8,
            scalar_types.float4_e2m1f,
        )
        else 8
    )
    m, k = hidden_states.shape
    num_experts = w1.shape[0]
    topk = topk_ids.shape[1]
    assert w1.shape[1] * 16 == k
    assert w2.shape[2] // (num_bits // 2) == k
    assert hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert topk_weights.dtype == torch.float32

    for block_size_m in (8, 16, 32, 48, 64):
        if m * topk / num_experts / block_size_m < 0.9:
            break
    block_size_m = max(block_size_m, 16)
    if global_num_experts == -1:
        global_num_experts = num_experts
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        marlin_moe.moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )
    )

    moe_output = _fused_marlin_moe_int8_activation(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        bias1=bias1,
        bias2=bias2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        num_topk=topk,
        quant_type=quant_type,
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=expert_map,
        block_size_m=block_size_m,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        activation=activation,
        activation_func=activation_func,
        input_global_scale1=input_global_scale1,
        input_global_scale2=input_global_scale2,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=None,
        input_dtype=input_dtype,
        is_k_full=is_k_full,
        clamp_limit=clamp_limit,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
    ).view(-1, topk, k)

    if output is None:
        output = torch.empty_like(hidden_states)
    if moe_sum is None:
        return torch.sum(moe_output, dim=1, out=output)
    return moe_sum(moe_output, output)

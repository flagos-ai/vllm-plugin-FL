# SPDX-License-Identifier: Apache-2.0
"""Bridge vLLM's direct calls for the Flash W8A8 model."""

from functools import cache, wraps
from importlib import import_module

import torch

from vllm import _custom_ops
from vllm.model_executor.layers.fused_moe.experts import triton_moe

from .native_callers import install as install_native_calls
from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import use_flaggems_op


@cache
def install():
    """Install once per W8A8 worker, before weights or graphs are created."""
    install_native_calls()
    functional_moe = import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    native_quant = _custom_ops.scaled_int8_quant
    native_expert = triton_moe.invoke_fused_moe_triton_kernel
    quantize = CachedOp("dynamic_per_token_quant_int8")
    expert_gemm = CachedOp("invoke_fused_moe_triton_kernel")

    @wraps(native_quant)
    def quant(input, scale=None, azp=None, symmetric=True):
        if (
            symmetric
            and scale is None
            and azp is None
            and input.is_cuda
            and input.dtype == torch.bfloat16
            and input.ndim == 2
            and input.is_contiguous()
            and input.shape[-1] > 0
        ):
            values, scales = quantize(input)
            return values, scales, None
        return native_quant(input, scale, azp, symmetric)

    @wraps(native_expert)
    def expert(
        a,
        b,
        c,
        a_scale,
        b_scale,
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
        implementation = native_expert
        if (
            use_int8_w8a8
            and per_channel_quant
            and block_shape is None
            and a.dtype == b.dtype == torch.int8
            and b.ndim == 3
            and b.shape[0] == 64
            and b.shape[1:] in ((3072, 1536), (1536, 1536))
        ):
            implementation = expert_gemm
        return implementation(
            a,
            b,
            c,
            a_scale,
            b_scale,
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

    if use_flaggems_op("dynamic_per_token_quant_int8"):
        _custom_ops.scaled_int8_quant = quant
    if use_flaggems_op("invoke_fused_moe_triton_kernel"):
        triton_moe.invoke_fused_moe_triton_kernel = expert
        functional_moe.invoke_fused_moe_triton_kernel = expert

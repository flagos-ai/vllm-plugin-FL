# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend operator registrations.

This module registers all VENDOR (CUDA) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all CUDA (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .cuda import CudaBackend

    backend = CudaBackend()
    is_avail = backend.is_available

    impls = [
        # DeepSeek-V4
        OpImpl(
            op_name="deepseek_v4_inv_rope_quant_int8",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.deepseek_v4_inv_rope_quant_int8,
                is_avail,
            ),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        *[
            OpImpl(
                op_name=f"deepseek_v4_{op_name}",
                impl_id="vendor.cuda",
                kind=BackendImplKind.VENDOR,
                fn=_bind_is_available(
                    getattr(backend, f"deepseek_v4_{op_name}"),
                    is_avail,
                ),
                vendor="cuda",
                priority=BackendPriority.VENDOR,
            )
            for op_name in (
                "inv_rope_quant_fp8",
                "int8_scaled_mm",
                "mhc_pre",
                "mhc_fused_post_pre",
                "mhc_post",
                "hc_head",
            )
        ],
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gelu_and_mul, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # MoE align
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_align_block_size, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # MoE sum
        OpImpl(
            op_name="moe_sum",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_sum, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # topk softmax
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # invoke fused moe triton kernel
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.invoke_fused_moe_triton_kernel, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # grouped topk
        OpImpl(
            op_name="grouped_topk",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.grouped_topk, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

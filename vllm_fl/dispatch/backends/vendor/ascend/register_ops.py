# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend operator registrations.

This module registers all VENDOR (Ascend) implementations.
"""

from __future__ import annotations

import functools

import torch

from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def _grouped_moe_experts(*args, **kwargs):
    # Import the NPU implementation only after the Ascend provider is chosen.
    from .impl.grouped_moe import grouped_experts

    return grouped_experts(*args, **kwargs)


def _w8a8_moe_experts(*, w1_bias, w2_bias, **kwargs):
    """Choose the native NPU kernel by the checkpoint's bias contract."""
    from vllm_fl.quantization.w8a8.moe_experts import (
        _ascend_w8a8_grouped_experts,
        _native_w8a8_fused_experts,
    )

    if kwargs.pop("activation") != "silu":
        raise NotImplementedError("Ascend W8A8 MoE supports SwiGLU only")
    kwargs.pop("global_num_experts")
    grouped_biases_supported = kwargs["hidden_states"].dtype in {
        torch.bfloat16,
        torch.float16,
    } and all(bias is None or bias.dtype == torch.int32 for bias in (w1_bias, w2_bias))
    implementation = (
        _ascend_w8a8_grouped_experts
        if grouped_biases_supported
        else _native_w8a8_fused_experts
    )
    return implementation(w1_bias=w1_bias, w2_bias=w2_bias, **kwargs)


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all Ascend (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .ascend import AscendBackend

    backend = AscendBackend()
    is_avail = backend.is_available

    impls = [
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="grouped_moe_experts",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(_grouped_moe_experts, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="w8a8_moe_experts",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(_w8a8_moe_experts, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

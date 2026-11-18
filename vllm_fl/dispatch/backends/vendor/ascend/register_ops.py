# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend operator registrations.

This module registers all VENDOR (Ascend) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all Ascend (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .ascend import AscendBackend
    from .impl.capabilities import has_npu_operator
    from .impl.rotary import rotary_embedding_torch_npu
    from .impl.vision_attention import (
        scaled_dot_product_attention_ascend,
        supports_scaled_dot_product_attention,
    )

    backend = AscendBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
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
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.ascend.torch_npu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                rotary_embedding_torch_npu,
                lambda: is_avail()
                and has_npu_operator("npu_rotary_mul", "rotary_mode"),
            ),
            vendor="ascend",
            priority=BackendPriority.VENDOR - 1,
        ),
        OpImpl(
            op_name="scaled_dot_product_attention",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                scaled_dot_product_attention_ascend,
                lambda: is_avail()
                and has_npu_operator(
                    "npu_fused_infer_attention_score",
                    "pse_shift",
                    "num_heads",
                    "num_key_value_heads",
                    "input_layout",
                    "scale",
                ),
            ),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
            supports=supports_scaled_dot_product_attention,
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
    ]

    for op_name in ("gdn_decode", "gdn_prefill"):
        impls.append(OpImpl(
            op_name=op_name,
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(getattr(backend, op_name), is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ))
    impls.append(OpImpl(
        op_name="fused_experts",
        impl_id="vendor.ascend",
        kind=BackendImplKind.VENDOR,
        fn=_bind_is_available(backend.fused_experts, is_avail),
        vendor="ascend",
        priority=BackendPriority.VENDOR,
    ))
    registry.register_many(impls)

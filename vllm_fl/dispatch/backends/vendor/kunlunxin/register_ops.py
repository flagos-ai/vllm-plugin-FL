# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin backend operator registrations.

This module registers all VENDOR (Kunlunxin) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all Kunlunxin (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .kunlunxin import KunlunxinBackend

    backend = KunlunxinBackend()
    is_avail = backend.is_available

    impls = [
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # FLA: Chunk Gated Delta Rule
        OpImpl(
            op_name="chunk_gated_delta_rule_fwd",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.chunk_gated_delta_rule_fwd, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # FLA: Fused Recurrent Gated Delta Rule
        OpImpl(
            op_name="fused_recurrent_gated_delta_rule_fwd",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_recurrent_gated_delta_rule_fwd, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # Causal Conv1d Forward
        OpImpl(
            op_name="causal_conv1d_fn",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.causal_conv1d_fn, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # Causal Conv1d Update
        OpImpl(
            op_name="causal_conv1d_update",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.causal_conv1d_update, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
        # Fused GDN Gating
        OpImpl(
            op_name="fused_gdn_gating",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_gdn_gating, is_avail),
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

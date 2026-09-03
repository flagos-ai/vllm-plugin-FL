# Copyright (c) 2026 BAAI. All rights reserved.

"""
GCU backend operator registrations.

This module registers all VENDOR (Enflame / GCU) implementations so the
dispatch policy's ``vendor:gcu`` entries in gcu.yaml resolve to the real
GCUBackend methods. Without it, the platform-level ``default.flagos``
implementations win for every op the yaml does not pin to ``vendor:gcu``
(flag_gems' TritonAttentionBackend for ``attention_backend`` among them —
correct on the newer torch_gcu 2.11/tops1.10.6 stack, garbage on
2.10/tops1.9.10, which is exactly the split observed 2026-09-03).
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
    """Register all GCU (VENDOR) operator implementations."""
    from .gcu import GCUBackend

    backend = GCUBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.gcu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="gcu",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.gcu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="gcu",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.gcu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="gcu",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.gcu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="gcu",
            priority=BackendPriority.VENDOR,
        ),
    ]
    registry.register_many(impls)

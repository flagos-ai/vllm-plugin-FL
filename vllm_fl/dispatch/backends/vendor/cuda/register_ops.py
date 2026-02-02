# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend operator registrations.

This module registers all VENDOR (CUDA) implementations.
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
    Register all CUDA (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .cuda import CudaBackend

    backend = CudaBackend()
    is_avail = backend.is_available

    impls = [
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rmsnorm",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rmsnorm, is_avail),
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
    ]

    registry.register_many(impls)

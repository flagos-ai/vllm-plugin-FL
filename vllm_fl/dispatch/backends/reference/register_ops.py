# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
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
    Register all PyTorch (REFERENCE) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .reference import ReferenceBackend
    from .impl.activation import silu_and_mul_torch
    from .impl.normalization import rms_norm_torch
    from .impl.rotary import rotary_embedding_torch

    backend = ReferenceBackend()
    is_avail = backend.is_available

    impls = [
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(silu_and_mul_torch, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(rms_norm_torch, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(rotary_embedding_torch, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Attention Backend (no instance binding needed)
        OpImpl(
            op_name="attention_backend",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
    ]

    registry.register_many(impls)

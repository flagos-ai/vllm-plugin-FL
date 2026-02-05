# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend operator registrations.

This module registers all DEFAULT (FlagGems) implementations.
Only impls for which use_flaggems_op(op_name) is True are passed to the registry.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
from vllm_fl.utils import use_flaggems_op


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all FlagGems (DEFAULT) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .flaggems import FlagGemsBackend

    backend = FlagGemsBackend()
    is_avail = backend.is_available

    impls = [
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
    ]

    filtered = [impl for impl in impls if use_flaggems_op(impl.op_name)]
    registry.register_many(filtered)

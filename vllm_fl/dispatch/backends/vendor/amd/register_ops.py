# Copyright (c) 2026 BAAI. All rights reserved.

"""
AMD backend operator registrations.

This module registers all VENDOR (AMD) implementations. FlagGems supplies the
element-wise/normalization operators on AMD, so the only vendor op registered
here is the attention-backend selection.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available for OpImpl.is_available() checks."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """Register all AMD (VENDOR) operator implementations."""
    from .amd import AMDBackend

    backend = AMDBackend()
    is_avail = backend.is_available

    impls = [
        # Attention backend selection (Triton attention via FlagTree).
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.amd",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="amd",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

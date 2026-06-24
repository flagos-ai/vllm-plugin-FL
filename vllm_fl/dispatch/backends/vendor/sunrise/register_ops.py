# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise backend operator registrations.

This module registers all VENDOR (Sunrise) implementations.
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
    Register all Sunrise (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .sunrise import SunriseBackend

    backend = SunriseBackend()
    is_avail = backend.is_available


    impls = [
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.sunrise",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="sunrise",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

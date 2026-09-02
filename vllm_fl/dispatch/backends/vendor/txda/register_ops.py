# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (tsingmicro) backend operator registrations.

This module registers the VENDOR (txda) implementations for the dispatch
system. Only the attention backend is registered: tsingmicro TX devices use
the flag_gems implementations for fused ops (silu_and_mul, rms_norm,
rotary_embedding), so no vendor overrides are needed for those.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register the txda (tsingmicro) VENDOR operator implementations.

    Args:
        registry: Registry to register into
    """
    from .txda import TxdaBackend

    backend = TxdaBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="txda",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

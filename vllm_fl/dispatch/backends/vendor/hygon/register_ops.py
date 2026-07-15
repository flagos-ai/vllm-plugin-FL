# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon backend operator registrations."""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available for OpImpl.is_available()."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register Hygon vendor operator implementations.

    Args:
        registry: Registry to register into.
    """
    from .hygon import HygonBackend

    backend = HygonBackend()
    is_avail = backend.is_available

    impls: list[OpImpl] = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gelu_and_mul, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_align_block_size, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="moe_sum",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_sum, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.invoke_fused_moe_triton_kernel, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="grouped_topk",
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.grouped_topk, is_avail),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)


__all__ = [
    "BackendImplKind",
    "BackendPriority",
    "OpImpl",
    "_bind_is_available",
    "register_builtins",
]

"""Register Hygon vendor implementations with the dispatch registry."""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    from .hygon import HygonBackend
    from .patch import apply_hygon_patches

    backend = HygonBackend()
    if backend.is_available():
        apply_hygon_patches()
    available = backend.is_available
    methods = {
        "rms_norm": backend.rms_norm,
        "rotary_embedding": backend.rotary_embedding,
        "attention_backend": backend.attention_backend,
    }
    registry.register_many(
        OpImpl(
            op_name=name,
            impl_id="vendor.hygon",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(method, available),
            vendor="hygon",
            priority=BackendPriority.VENDOR,
        )
        for name, method in methods.items()
    )

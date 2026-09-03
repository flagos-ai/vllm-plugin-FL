# Copyright (c) 2026 BAAI. All rights reserved.

"""
GCU backend operator registrations.
"""

from __future__ import annotations

import functools
import logging

import torch

from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

logger = logging.getLogger(__name__)


def _bind_is_available(fn, is_available_fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def _register_c_ops_privateuse1_fallbacks() -> None:
    """Register _C op fallback impls for missing dispatch keys.

    vllm's _C_ops_registry only registers the pure-torch fallback for the
    "CUDA" dispatch key.  On GCU the logits tensor is on PrivateUse1, so we
    need a concrete-key registration for PrivateUse1.

    Note: "CompositeImplicitAutograd" does NOT work here — it is only
    suitable for pure query ops (like cutlass_scaled_mm_supports_fp8),
    not for mutation ops like apply_repetition_penalties_ (Tensor!
    schema with in-place mutation).
    """
    try:
        import vllm._C  # noqa: F401
        return  # real _C extension is present — nothing to do
    except (ImportError, OSError):
        pass

    try:
        lib = torch.library.Library("_C", "FRAGMENT")
    except Exception:
        logger.exception("Failed to open _C library fragment for fallbacks")
        return

    from vllm_fl.ops._C_ops_registry import _apply_repetition_penalties_impl

    lib.impl(
        "apply_repetition_penalties_",
        _apply_repetition_penalties_impl,
        "PrivateUse1",
    )
    logger.info("Registered _C::apply_repetition_penalties_ for PrivateUse1")

    # Keep the Library object alive — if it is garbage-collected, all
    # implementations registered through it are removed from the dispatch
    # table.  This mirrors register_op_schemas._lib in _C_ops_registry.py.
    _register_c_ops_privateuse1_fallbacks._lib = lib


def register_builtins(registry: OpRegistry) -> None:
    from .gcu import GCUBackend

    backend = GCUBackend()
    is_avail = backend.is_available

    # Register _C op fallbacks for PrivateUse1 dispatch key so that
    # apply_repetition_penalties_ (and similar ops) work on GCU.
    _register_c_ops_privateuse1_fallbacks()

    impls = [
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
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.gcu",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="gcu",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

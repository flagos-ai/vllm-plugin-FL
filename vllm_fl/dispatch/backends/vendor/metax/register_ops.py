# Copyright (c) 2026 BAAI. All rights reserved.

"""
Maca backend operator registrations.

This module registers all VENDOR (Metax) implementations.
"""

from __future__ import annotations

import functools
import importlib

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _native_ops_available(hardware_available, *qualified_names: str) -> bool:
    """A schema-only empty wheel does not provide a vendor implementation."""
    if not hardware_available():
        return False
    # Keep the implementation package lazy: it also imports vLLM activation
    # layers and cannot be loaded while vLLM is discovering platform plugins.
    from .impl.attention.ops._kernel_utils import has_cuda_or_composite_kernel

    return all(has_cuda_or_composite_kernel(name) for name in qualified_names)


def _precompiled_moe_available(hardware_available) -> bool:
    if not hardware_available():
        return False
    try:
        module = importlib.import_module("mcoplib.triton_fused_moe")
    except Exception:
        # The package is optional in official FlagOS images. A missing or
        # ABI-incompatible native extension must leave FlagGems selectable.
        return False
    return callable(getattr(module, "fused_moe_triton_kernel", None))


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
    from . import patches  # noqa: F401 — apply MetaX kernel patches at backend load
    from .metax import MacaBackend

    backend = MacaBackend()
    is_avail = backend.is_available

    impls = [
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.silu_and_mul,
                functools.partial(_native_ops_available, is_avail, "_C::silu_and_mul"),
            ),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.rms_norm,
                functools.partial(
                    _native_ops_available,
                    is_avail,
                    "_C::rms_norm",
                    "_C::fused_add_rms_norm",
                ),
            ),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.rotary_embedding,
                functools.partial(
                    _native_ops_available, is_avail, "_C::rotary_embedding"
                ),
            ),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_align_block_size, is_avail),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        # topk softmax
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.topk_softmax,
                functools.partial(
                    _native_ops_available, is_avail, "_moe_C::topk_softmax"
                ),
            ),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        # invoke fused moe triton kernel
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.invoke_fused_moe_triton_kernel,
                functools.partial(_precompiled_moe_available, is_avail),
            ),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

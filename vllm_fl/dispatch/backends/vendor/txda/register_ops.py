# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX backend operator registrations.

This module registers all VENDOR (METAX) implementations.
"""

from __future__ import annotations

import functools
import logging

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

logger = logging.getLogger(__name__)


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all METAX (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .txda import TxdaBackend

    backend = TxdaBackend()
    is_avail = backend.is_available

    impls = [
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="txda",
            priority=BackendPriority.VENDOR,
        ),
        # Fused MoE Triton Kernel — delegates to vLLM upstream
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.invoke_fused_moe_triton_kernel, is_avail),
            vendor="txda",
            priority=BackendPriority.VENDOR,
        ),
        # RMS Norm — stub (raises NotImplementedError), so dispatch falls
        # through to the next backend (reference) per tsingmicro.yaml config.
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="txda",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

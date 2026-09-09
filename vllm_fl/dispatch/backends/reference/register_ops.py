# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


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

    backend = ReferenceBackend()
    is_avail = backend.is_available

    # ReferenceBackend implements only part of the dispatch surface on some
    # vLLM 0.24 builds. One absent optional MoE helper must not prevent all
    # available PyTorch fallbacks from registering.
    op_names = (
        "dynamic_per_token_quant_int8",
        "silu_and_mul",
        "gelu_and_mul",
        "rms_norm",
        "rotary_embedding",
        "attention_backend",
        "moe_align_block_size",
        "moe_sum",
        "topk_softmax",
        "invoke_fused_moe_triton_kernel",
        "grouped_topk",
    )
    impls = []
    for op_name in op_names:
        fn = getattr(backend, op_name, None)
        if fn is None:
            continue
        impls.append(
            OpImpl(
                op_name=op_name,
                impl_id="reference.torch",
                kind=BackendImplKind.REFERENCE,
                fn=_bind_is_available(fn, is_avail),
                vendor=None,
                priority=BackendPriority.REFERENCE,
            )
        )

    registry.register_many(impls)

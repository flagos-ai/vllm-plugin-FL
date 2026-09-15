# Copyright (c) 2026 BAAI. All rights reserved.

"""
Thead (PPU) backend operator registrations.

This module registers VENDOR (thead) implementations for the dispatch system.
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
    Register all thead (PPU) VENDOR operator implementations.

    At registration time we also load the flash_attn_3 wheel so that
    TheadFlashAttentionBackend can call FA3 ops.

    Args:
        registry: Registry to register into
    """
    from .thead import TheadBackend

    backend = TheadBackend()
    is_avail = backend.is_available
    cache_ops_available = backend.native_cache_ops_are_available
    dynamic_quant_available = backend.native_dynamic_quant_is_available
    moe_ops_available = backend.native_moe_ops_are_available

    impls = [
        OpImpl(
            op_name="mla_prefill",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.mla_prefill, backend.mla_prefill_is_available
            ),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="bf16_indexer_cache_write",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.bf16_indexer_cache_write, cache_ops_available
            ),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="dynamic_per_token_quant_int8",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.dynamic_per_token_quant_int8, dynamic_quant_available
            ),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_align_block_size, moe_ops_available),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="moe_sum",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_sum, moe_ops_available),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk_softmax, moe_ops_available),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="grouped_topk",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.grouped_topk, moe_ops_available),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

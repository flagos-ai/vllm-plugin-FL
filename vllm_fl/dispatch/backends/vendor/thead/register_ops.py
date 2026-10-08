# Copyright (c) 2026 BAAI. All rights reserved.

"""Thead (PPU) backend operator registrations."""

from __future__ import annotations

import functools
from functools import lru_cache
from importlib import import_module

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available for OpImpl."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def _lazy_attr(resolve_owner, name: str, is_available):
    resolved_fn = None

    def resolve():
        nonlocal resolved_fn
        if resolved_fn is None:
            resolved_fn = getattr(resolve_owner(), name)
        return resolved_fn

    def invoke(*args, **kwargs):
        # Keep graph capture on the import-free, pre-resolved path.
        fn = resolved_fn
        if fn is None:
            fn = resolve()
        return fn(*args, **kwargs)

    invoke._is_available = is_available
    invoke._prepare = resolve
    return invoke


def _vendor_impl(op_name, fn):
    return OpImpl(
        op_name=op_name,
        impl_id="vendor.thead",
        kind=BackendImplKind.VENDOR,
        fn=fn,
        vendor="thead",
        priority=BackendPriority.VENDOR,
    )


def register_builtins(registry) -> None:
    """Register Thead PPU implementations."""
    from .thead import TheadBackend

    backend = TheadBackend()
    is_avail = backend.is_available
    if is_avail():
        from vllm_fl.patches.thead_kda_prefill import (
            apply_thead_kda_prefill_metadata_patch,
        )

        apply_thead_kda_prefill_metadata_patch()
    impls = [
        _vendor_impl(
            "mla_prefill",
            _bind_is_available(backend.mla_prefill, backend.mla_prefill_is_available),
        ),
        _vendor_impl(
            "bf16_indexer_cache_write",
            _bind_is_available(
                backend.bf16_indexer_cache_write,
                backend.native_cache_ops_are_available,
            ),
        ),
        _vendor_impl(
            "dynamic_per_token_quant_int8",
            _bind_is_available(
                backend.dynamic_per_token_quant_int8,
                backend.native_dynamic_quant_is_available,
            ),
        ),
        _vendor_impl(
            "moe_align_block_size",
            _bind_is_available(
                backend.moe_align_block_size, backend.native_moe_ops_are_available
            ),
        ),
        _vendor_impl(
            "moe_sum",
            _bind_is_available(backend.moe_sum, backend.native_moe_ops_are_available),
        ),
        _vendor_impl(
            "topk_softmax",
            _bind_is_available(
                backend.topk_softmax, backend.native_moe_ops_are_available
            ),
        ),
        _vendor_impl(
            "grouped_topk",
            _bind_is_available(
                backend.grouped_topk, backend.native_moe_ops_are_available
            ),
        ),
        _vendor_impl(
            "attention_backend",
            _bind_is_available(backend.attention_backend, is_avail),
        ),
    ]

    @lru_cache(None)
    def resolve_indexer():
        cls = getattr(
            import_module(
                "vllm_fl.dispatch.backends.vendor.thead.impl.sparse_indexer"
            ),
            "IndexerOps",
        )
        return cls()

    def indexer_fn(name):
        return _lazy_attr(resolve_indexer, name, is_avail)

    @lru_cache(None)
    def resolve_flaggems_vllm():
        return import_module("flaggems_vllm")

    @lru_cache(None)
    def resolve_causal_conv1d():
        return resolve_flaggems_vllm()

    @lru_cache(None)
    def causal_conv1d_is_available():
        if not is_avail():
            return False
        try:
            implementation = import_module(
                "flaggems_vllm.runtime.backend._thead.ops.causal_conv1d"
            )
            return callable(
                getattr(resolve_causal_conv1d(), "causal_conv1d_fn")
            ) and implementation.is_available()
        except (AttributeError, ImportError, OSError):
            return False

    impls.append(
        _vendor_impl(
            "causal_conv1d_fn",
            _lazy_attr(
                resolve_causal_conv1d,
                "causal_conv1d_fn",
                causal_conv1d_is_available,
            ),
        )
    )

    @lru_cache(None)
    def resolve_chunk_kda():
        return resolve_flaggems_vllm()

    @lru_cache(None)
    def chunk_kda_is_available():
        if not is_avail():
            return False
        try:
            implementation = import_module(
                "flaggems_vllm.runtime.backend._thead.ops.chunk_kda"
            )
            return callable(
                getattr(resolve_chunk_kda(), "chunk_kda_with_safe_gate")
            ) and implementation.is_available()
        except (AttributeError, ImportError, OSError):
            return False

    impls.append(
        _vendor_impl(
            "chunk_kda_with_safe_gate",
            _lazy_attr(
                resolve_chunk_kda,
                "chunk_kda_with_safe_gate",
                chunk_kda_is_available,
            ),
        )
    )

    def capabilities():
        import torch

        return {
            "cache_dtype": torch.bfloat16,
            "query_dtype": torch.bfloat16,
            "compressed_page_size": 32,
        }

    capabilities._is_available = is_avail
    impls.append(_vendor_impl("sparse_indexer_capabilities", capabilities))
    for name in (
        "validate_environment",
        "prepare_query",
        "indexer_k_quant_and_cache",
    ):
        impls.append(_vendor_impl("sparse_indexer_" + name, indexer_fn(name)))
    registry.register_many(impls)

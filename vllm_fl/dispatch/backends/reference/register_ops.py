# Copyright (c) 2026 BAAI. All rights reserved.

"""Reference backend operator registrations."""

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


def _lazy_fn(module: str, name: str, is_available):
    """Import an optional capability only when dispatch selects it."""

    @lru_cache(None)
    def resolve():
        return getattr(import_module(module), name)

    def invoke(*args, **kwargs):
        return resolve()(*args, **kwargs)

    invoke._is_available = is_available
    invoke._prepare = resolve
    return invoke


def _reference_impl(op_name, module, symbol, is_available):
    return OpImpl(
        op_name=op_name,
        impl_id="reference.torch",
        kind=BackendImplKind.REFERENCE,
        fn=_lazy_fn(module, symbol, is_available),
        vendor=None,
        priority=BackendPriority.REFERENCE,
    )


def register_builtins(registry) -> None:
    """Register PyTorch reference implementations."""
    from .reference import ReferenceBackend

    backend = ReferenceBackend()
    is_avail = backend.is_available
    # ReferenceBackend implements only part of the dispatch surface on some
    # vLLM 0.24 builds. One absent optional MoE helper must not prevent all
    # available PyTorch fallbacks from registering.
    op_names = (
        "bf16_indexer_cache_write",
        "bf16_indexer_decode",
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

    capability_ops = {
        "mhc_pre_with_norm": ("mhc", "mhc_pre"),
        "mhc_post": ("mhc", "mhc_post"),
        "mhc_fused_post_pre_with_norm": ("mhc", "mhc_fused_post_pre"),
        "silu_and_mul_with_clamp": ("activation", "silu_and_mul_with_clamp"),
        "causal_conv1d_fn": ("causal_conv1d", "causal_conv1d_fn"),
        "chunk_kda_with_safe_gate": ("kda", "chunk_kda_with_safe_gate"),
        "sparse_indexer_rotate_indexer_query": (
            "sparse_indexer",
            "rotate_indexer_query",
        ),
        "sparse_indexer_mqa_logits": ("sparse_indexer", "_torch_mqa_logits"),
        "top_k_per_row_prefill": ("top_k_per_row", "top_k_per_row_prefill"),
        "top_k_per_row_decode": ("top_k_per_row", "top_k_per_row_decode"),
        "sparse_indexer_pack_seq": ("sparse_indexer", "_torch_pack_seq"),
        "sparse_indexer_unpack_seq": ("sparse_indexer", "_torch_unpack_seq"),
        "sparse_indexer_expand_pools_to_tokens": (
            "sparse_indexer",
            "expand_pools_to_tokens",
        ),
        "sparse_indexer_append_tail_to_topk": (
            "sparse_indexer",
            "append_tail_to_topk",
        ),
    }
    for op_name, (module_name, symbol) in capability_ops.items():
        impls.append(
            _reference_impl(
                op_name,
                f"vllm_fl.dispatch.backends.reference.impl.{module_name}",
                symbol,
                is_avail,
            )
        )
    registry.register_many(impls)

# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend operator registrations.

This module registers all DEFAULT (FlagGems) implementations.
Only impls for which use_flaggems_op(op_name) is True are passed to the registry.
"""

from __future__ import annotations

import functools
import logging
from functools import lru_cache
from importlib import import_module

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from vllm_fl.utils import use_flaggems_op


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def _lazy_fn(module: str, name: str, is_available):
    resolved_fn = None

    def resolve():
        nonlocal resolved_fn
        if resolved_fn is None:
            resolved_fn = getattr(import_module(module), name)
        return resolved_fn

    def invoke(*args, **kwargs):
        # TorchDynamo intentionally traces through functools.lru_cache wrappers.
        # Keep the resolved callable in the closure so prepare_cached_ops() makes
        # graph capture take this import-free path.
        fn = resolved_fn
        if fn is None:
            fn = resolve()
        return fn(*args, **kwargs)

    invoke._is_available = is_available
    invoke._prepare = resolve
    return invoke


def _register_capability_ops(registry, is_available):
    """Register optional model capabilities implemented by FlagGems."""
    torch_module = import_module("torch")
    torch_module._dynamo.config.ignore_logger_methods.add(logging.Logger.debug)
    specs = {
        "mhc_pre_with_norm": (
            "mhc",
            "mhc_pre",
            ("flag_gems.fused.mhc", "mhc_pre"),
        ),
        "mhc_post": ("mhc", "mhc_post", ("flag_gems.fused.mhc", "mhc_post")),
        "mhc_fused_post_pre_with_norm": (
            "mhc",
            "mhc_fused_post_pre",
            ("flag_gems.fused.mhc", "mhc_pre"),
            ("flag_gems.fused.mhc", "mhc_post"),
        ),
        "silu_and_mul_with_clamp": (
            "activation",
            "silu_and_mul_with_clamp",
            (
                "flag_gems.fused.silu_and_mul_with_clamp",
                "silu_and_mul_with_clamp_kernel",
            ),
        ),
        "sparse_indexer_rotate_indexer_query": (
            "sparse_indexer",
            "rotate_indexer_query",
            ("flag_gems.ops.hadamard_transform", "hadamard_transform"),
        ),
        "sparse_indexer_mqa_logits": (
            "sparse_indexer",
            "mqa_logits",
            ("flaggems_vllm", "fp8_fp4_mqa_logits"),
        ),
        "top_k_per_row_prefill": (
            "top_k_per_row",
            "top_k_per_row_prefill",
            ("flag_gems.fused.top_k_per_row_prefill", "top_k_per_row_prefill"),
        ),
        "top_k_per_row_decode": (
            "top_k_per_row",
            "top_k_per_row_decode",
            ("flag_gems.fused.top_k_per_row_decode", "top_k_per_row_decode"),
        ),
        "sparse_indexer_gather_cache": (
            "sparse_indexer",
            "gather_cache",
            (
                "flaggems_vllm",
                "cp_gather_indexer_k_bf16_cache",
            ),
        ),
        "sparse_indexer_paged_mqa_logits": (
            "sparse_indexer",
            "paged_mqa_logits",
            (
                "flaggems_vllm",
                "bf16_paged_mqa_logits_graph_safe",
            ),
        ),
        "sparse_indexer_pack_seq": (
            "flag_gems.fused.pack_seq",
            "pack_seq_triton",
        ),
        "sparse_indexer_unpack_seq": (
            "flag_gems.fused.unpack_seq",
            "unpack_seq_triton",
        ),
        "sparse_indexer_expand_pools_to_tokens": (
            "flaggems_vllm",
            "expand_pools_to_tokens",
        ),
        "sparse_indexer_append_tail_to_topk": (
            "flaggems_vllm",
            "append_tail_to_topk",
        ),
        "sparse_indexer_persist_prefill_tail": (
            "sparse_indexer",
            "persist_prefill_tail",
            ("flaggems_vllm", "persist_prefill_tail"),
        ),
        "sparse_indexer_kpool_compress_and_write_cache": (
            "sparse_indexer",
            "kpool_compress_and_write_cache",
            (
                "flaggems_vllm",
                "kpool_compress_and_write_cache",
            ),
        ),
        "sparse_indexer_kpool_decode_update_and_maybe_write_cache_batched": (
            "sparse_indexer",
            "kpool_decode_update_and_maybe_write_cache_batched",
            (
                "flaggems_vllm",
                "kpool_decode_update_and_maybe_write_cache_batched",
            ),
        ),
        "causal_conv1d_update": (
            "flaggems_vllm",
            "causal_conv1d_update",
        ),
        "causal_conv1d_fn": (
            "flaggems_vllm",
            "causal_conv1d_fn",
        ),
        "fused_recurrent_kda": (
            "flaggems_vllm",
            "fused_recurrent_kda",
        ),
        "chunk_kda_with_safe_gate": (
            "flaggems_vllm",
            "chunk_kda_with_safe_gate",
        ),
        "fused_safe_kda_gate": (
            "flaggems_vllm",
            "fused_safe_kda_gate",
        ),
    }
    impls = []
    for op_name, (module_name, symbol, *dependencies) in specs.items():
        target_module = (
            module_name
            if module_name == "flaggems_vllm"
            or module_name.startswith("flag_gems.")
            else f"vllm_fl.dispatch.backends.flaggems.impl.{module_name}"
        )
        required_names = (op_name, symbol, *(name for _, name in dependencies))
        if not all(use_flaggems_op(name) for name in required_names):
            continue

        @lru_cache(None)
        def available(
            target=(target_module, symbol), dependencies=tuple(dependencies)
        ):
            if not is_available():
                return False
            try:
                for module, name in (target, *dependencies):
                    getattr(import_module(module), name)
            except (ImportError, AttributeError):
                return False
            return True

        impls.append(
            OpImpl(
                op_name=op_name,
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=_lazy_fn(
                    target_module,
                    symbol,
                    available,
                ),
                vendor=None,
                priority=BackendPriority.DEFAULT,
            )
        )
    registry.register_many(impls)


def register_builtins(registry) -> None:
    """
    Register all FlagGems (DEFAULT) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .flaggems import FlagGemsBackend

    backend = FlagGemsBackend()
    is_avail = backend.is_available

    @lru_cache(None)
    def flaggems_vllm_fused_moe_is_available():
        if not is_avail():
            return False
        try:
            module = import_module("flaggems_vllm")
            return callable(
                getattr(module, "invoke_fused_moe_triton_kernel")
            )
        except (AttributeError, ImportError, OSError):
            return False

    impls = [
        OpImpl(
            op_name="mla_prefill",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.mla_prefill, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # BF16 indexer graph operators
        OpImpl(
            op_name="bf16_indexer_cache_write",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.bf16_indexer_cache_write, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="bf16_indexer_decode",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.bf16_indexer_decode, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Quantization
        OpImpl(
            op_name="dynamic_per_token_quant_int8",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(
                backend.dynamic_per_token_quant_int8,
                is_avail,
            ),
            vendor=None,
            priority=BackendPriority.DEFAULT + 10,
        ),
        OpImpl(
            op_name="dynamic_per_token_quant_int8",
            impl_id="default.flagos_triton",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(
                backend.dynamic_per_token_quant_int8_triton,
                is_avail,
            ),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.gelu_and_mul, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # MoE align
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.moe_align_block_size, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # MoE sum
        OpImpl(
            op_name="moe_sum",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.moe_sum, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # topk softmax
        OpImpl(
            op_name="topk_softmax",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # invoke fused moe triton kernel
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(
                backend.invoke_fused_moe_triton_kernel,
                flaggems_vllm_fused_moe_is_available,
            ),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # grouped topk
        OpImpl(
            op_name="grouped_topk",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.grouped_topk, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
    ]

    filtered = [impl for impl in impls if use_flaggems_op(impl.op_name)]
    registry.register_many(filtered)
    _register_capability_ops(registry, is_avail)

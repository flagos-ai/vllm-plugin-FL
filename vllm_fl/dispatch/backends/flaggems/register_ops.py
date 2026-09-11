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
    """Register reusable row-wise top-k capabilities from FlagGems."""
    torch_module = import_module("torch")
    torch_module._dynamo.config.ignore_logger_methods.add(logging.Logger.debug)
    specs = {
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
    }
    impls = []
    for op_name, (module_name, symbol, *dependencies) in specs.items():
        target_module = f"vllm_fl.dispatch.backends.flaggems.impl.{module_name}"
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
                fn=_lazy_fn(target_module, symbol, available),
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
            fn=_bind_is_available(backend.invoke_fused_moe_triton_kernel, is_avail),
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

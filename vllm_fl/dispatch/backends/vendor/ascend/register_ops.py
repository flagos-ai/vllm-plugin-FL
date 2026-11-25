# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend operator registrations.

This module registers all VENDOR (Ascend) implementations.
"""

from __future__ import annotations

import functools
import importlib
import inspect

from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


# Check complete native call signatures without allocating or launching tensors.
_NATIVE_CALLS = {
    "npu_dynamic_quant": (1, ("dst_type",)),
    "npu_format_cast": (2, ()),
    "npu_quant_matmul": (3, ("pertoken_scale", "bias", "output_dtype")),
    "npu_moe_init_routing_v2": (
        2,
        (
            "active_num",
            "expert_num",
            "expert_tokens_num_type",
            "expert_tokens_num_flag",
            "active_expert_range",
            "quant_mode",
        ),
    ),
    "npu_grouped_matmul": (
        0,
        (
            "x",
            "weight",
            "scale",
            "per_token_scale",
            "split_item",
            "group_type",
            "group_list_type",
            "group_list",
            "output_dtype",
        ),
    ),
    "npu_moe_token_unpermute": (0, ("permuted_tokens", "sorted_indices", "probs")),
    "npu_swiglu": (1, ()),
}
_NATIVE_REQUIREMENTS = {
    "deepseek_v4_index_quant": ("npu_dynamic_quant",),
    "w8a8_prepare_weight": ("npu_format_cast",),
    "w8a8_prepare_experts": (),
    "w8a8_dynamic_linear": ("npu_dynamic_quant", "npu_quant_matmul"),
    "w8a8_experts": (
        "npu_moe_init_routing_v2",
        "npu_dynamic_quant",
        "npu_grouped_matmul",
        "npu_swiglu",
        "npu_moe_token_unpermute",
    ),
}


def _native_available(op_name, device_available):
    if not device_available():
        return False
    if not _NATIVE_REQUIREMENTS[op_name]:
        return True
    try:
        native = importlib.import_module("torch_npu")
        for name in _NATIVE_REQUIREMENTS[op_name]:
            fn = getattr(native, name, None)
            if not callable(fn):
                return False
            # torch_npu exports OpOverloadPackets: inspect.signature reports
            # (*args, **kwargs), while the default dispatcher schema is precise.
            schema = getattr(fn, "_schemas", {}).get("")
            if schema is not None:
                signature = inspect.Signature(
                    [
                        inspect.Parameter(
                            arg.name,
                            inspect.Parameter.KEYWORD_ONLY
                            if arg.kwarg_only
                            else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            default=None
                            if arg.has_default_value()
                            else inspect.Parameter.empty,
                        )
                        for arg in schema.arguments
                    ]
                )
            else:
                signature = inspect.signature(fn)
                if any(
                    p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                    for p in signature.parameters.values()
                ):
                    return False
            count, keywords = _NATIVE_CALLS[name]
            signature.bind(*([None] * count), **dict.fromkeys(keywords))
    except (ImportError, TypeError, ValueError):
        return False
    return True


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all Ascend (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .ascend import AscendBackend

    backend = AscendBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
    ]

    for op_name in (
        "deepseek_v4_index_quant",
        "w8a8_prepare_experts",
        "w8a8_experts",
        "gdn_decode",
        "gdn_prefill",
        "w8a8_prepare_weight",
        "w8a8_dynamic_linear",
    ):
        impls.append(
            OpImpl(
                op_name=op_name,
                impl_id="vendor.ascend",
                kind=BackendImplKind.VENDOR,
                fn=_bind_is_available(
                    getattr(backend, op_name),
                    functools.partial(_native_available, op_name, is_avail)
                    if op_name in _NATIVE_REQUIREMENTS
                    else is_avail,
                ),
                vendor="ascend",
                priority=BackendPriority.VENDOR,
                allow_execution_fallback=op_name not in _NATIVE_REQUIREMENTS,
            )
        )
    impls.append(
        OpImpl(
            op_name="fused_experts",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_experts, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        )
    )
    registry.register_many(impls)

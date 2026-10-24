# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend operator registrations.

This module registers all DEFAULT (FlagGems) implementations.
Only impls for which use_flaggems_op(op_name) is True are passed to the registry.
"""

from __future__ import annotations

import functools

from vllm.logger import init_logger

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from vllm_fl.utils import use_flaggems_op

logger = init_logger(__name__)


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all FlagGems (DEFAULT) operator implementations.

    Args:
        registry: Registry to register into
    """
    import flaggems_vllm

    from .flaggems import FlagGemsBackend

    logger.info_once(
        "FlagGems-vllm integration activated: package=%s",
        getattr(flaggems_vllm, "__file__", "<unknown>"),
    )

    backend = FlagGemsBackend()
    is_avail = backend.is_available
    from .impl.w8a8_moe import w8a8_moe_experts

    impls = [
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
        OpImpl(
            op_name="w8a8_moe_experts",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(w8a8_moe_experts, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
    ]

    filtered = [impl for impl in impls if use_flaggems_op(impl.op_name)]
    registry.register_many(filtered)

    from vllm_fl.patches.kimi_k3.profile import is_kimi_k3_profile

    if is_kimi_k3_profile():
        from .impl.kimi_k3 import grouped_topk, rms_norm_flaggems

        providers = {
            name: fn
            for name, fn in (
                ("rms_norm", rms_norm_flaggems),
                ("grouped_topk", grouped_topk),
            )
            if use_flaggems_op(name)
        }
        for name, fn in providers.items():
            identity = "flagos.kimi_k3." + name
            prior = registry.get_implementation(name, identity)
            if prior is not None:
                if prior.fn is not fn:
                    raise RuntimeError(f"Conflicting K3 provider: {name}")
                continue
            registry.register_impl(OpImpl(
                op_name=name, impl_id=identity,
                kind=BackendImplKind.DEFAULT, fn=fn,
                priority=BackendPriority.DEFAULT + 10,
            ))

# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend operator registrations.

This module registers all DEFAULT (FlagGems) implementations.
Only impls for which use_flaggems_op(op_name) is True are passed to the registry.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from vllm_fl.utils import use_flaggems_op


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
    from .flaggems import FlagGemsBackend

    # Import compiler-visible implementations while dispatch is still in its
    # initialization phase.  Registering a torch.library op during Dynamo
    # tracing would itself be Python control flow and is intentionally banned.
    from .impl.activation import (
        gelu_and_mul_flaggems,
        silu_and_mul_flaggems,
    )
    from .impl.fused_moe import (
        grouped_topk_flaggems,
        invoke_fused_moe_triton_kernel_flaggems,
        moe_align_block_size_flaggems,
        moe_sum_flaggems,
        topk_softmax_flaggems,
    )
    from .impl.normalization import rms_norm_flaggems
    from .impl.rotary import rotary_embedding_flaggems

    backend = FlagGemsBackend()
    is_avail = backend.is_available

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
            fn=_bind_is_available(silu_and_mul_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(gelu_and_mul_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(rms_norm_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(rotary_embedding_flaggems, is_avail),
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
            fn=_bind_is_available(moe_align_block_size_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # MoE sum
        OpImpl(
            op_name="moe_sum",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(moe_sum_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # topk softmax
        OpImpl(
            op_name="topk_softmax",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(topk_softmax_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # invoke fused moe triton kernel
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(invoke_fused_moe_triton_kernel_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        # grouped topk
        OpImpl(
            op_name="grouped_topk",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(grouped_topk_flaggems, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
    ]

    filtered = [impl for impl in impls if use_flaggems_op(impl.op_name)]
    registry.register_many(filtered)

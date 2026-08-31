# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
"""

from __future__ import annotations

import functools
import logging

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

logger = logging.getLogger(__name__)


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


# (op_name, backend method name) pairs. getattr is resolved lazily below so a
# single missing method logs a warning and skips that op instead of aborting
# the whole reference registration (which would leave every `reference` token
# in a platform yaml matching nothing).
_REFERENCE_OPS = [
    ("dynamic_per_token_quant_int8", "dynamic_per_token_quant_int8"),
    ("silu_and_mul", "silu_and_mul"),
    ("gelu_and_mul", "gelu_and_mul"),
    ("rms_norm", "rms_norm"),
    ("rotary_embedding", "rotary_embedding"),
    ("attention_backend", "attention_backend"),
    ("moe_align_block_size", "moe_align_block_size"),
    ("moe_sum", "moe_sum"),
    ("topk_softmax", "topk_softmax"),
    ("invoke_fused_moe_triton_kernel", "invoke_fused_moe_triton_kernel"),
    ("grouped_topk", "grouped_topk"),
]


def register_builtins(registry) -> None:
    """
    Register all PyTorch (REFERENCE) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .reference import ReferenceBackend

    backend = ReferenceBackend()
    is_avail = backend.is_available

    impls = []
    for op_name, method_name in _REFERENCE_OPS:
        try:
            method = getattr(backend, method_name)
        except AttributeError:
            logger.warning(
                "Reference backend missing %s; skipping op %s",
                method_name,
                op_name,
            )
            continue
        impls.append(
            OpImpl(
                op_name=op_name,
                impl_id="reference.torch",
                kind=BackendImplKind.REFERENCE,
                fn=_bind_is_available(method, is_avail),
                vendor=None,
                priority=BackendPriority.REFERENCE,
            )
        )

    registry.register_many(impls)

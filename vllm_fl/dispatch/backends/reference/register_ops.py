# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check.

    ``fn`` may be None when the reference backend does not implement that op;
    pass it through so the caller can filter the entry out.
    """
    if fn is None:
        return None

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

    def _method(name):
        """Return backend.<name>, or a placeholder that is filtered out below.

        The impls list below is built eagerly, so referencing a method the
        backend never implemented raised AttributeError and aborted
        registration of *every* reference op -- including the ones that do
        exist.  That silently emptied the `reference` fallback tier, so ops
        with a configured `[flagos, vendor, reference]` order had only one
        candidate and could not fall back at all.
        """
        return getattr(backend, name, None)

    impls = [
        # Quantization
        OpImpl(
            op_name="dynamic_per_token_quant_int8",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(
                backend.dynamic_per_token_quant_int8,
                is_avail,
            ),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("silu_and_mul"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("gelu_and_mul"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("rms_norm"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("rotary_embedding"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("attention_backend"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # MoE align
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("moe_align_block_size"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # MoE sum
        OpImpl(
            op_name="moe_sum",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("moe_sum"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # topk softmax
        OpImpl(
            op_name="topk_softmax",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("topk_softmax"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # invoke fused moe triton kernel
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("invoke_fused_moe_triton_kernel"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # grouped topk
        OpImpl(
            op_name="grouped_topk",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(_method("grouped_topk"), is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
    ]

    # Drop entries whose backend method does not exist rather than letting a
    # single gap take down the whole reference tier.
    missing = [i.op_name for i in impls if i.fn is None]
    impls = [i for i in impls if i.fn is not None]
    if missing:
        import logging

        logging.getLogger(__name__).debug(
            "Reference backend has no implementation for %s; skipping those ops.",
            ", ".join(sorted(missing)),
        )

    registry.register_many(impls)

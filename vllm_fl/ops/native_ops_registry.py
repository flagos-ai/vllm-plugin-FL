# Copyright (c) 2026 BAAI. All rights reserved.

"""Route missing vLLM native kernels through the FL YAML dispatcher.

Only native operator names in ``namespace::op`` form are supported.  The
operator name after ``::`` is also used as the FL/YAML dispatch name.

This registry only installs a bridge when an operator schema already exists
and the requested device dispatch key has no implementation.  Schema fallback
remains the responsibility of ``_C_ops_registry.py`` and
``_C_ops_schemas.py``.  Existing implementations are retained and cannot be
overridden through this mechanism.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch


_NATIVE_OPS = [
    "_moe_C::moe_sum",
    "_moe_C::moe_align_block_size",
    "_moe_C::topk_softplus_sqrt",
    "_C::silu_and_mul",
    "_C::dynamic_scaled_int8_quant",
]

NATIVE_OP_SCHEMAS = {
    "_moe_C::moe_sum": "moe_sum(Tensor input, Tensor! output) -> ()",
    "_moe_C::moe_align_block_size": (
        "moe_align_block_size(Tensor topk_ids, int num_experts, "
        "int block_size, Tensor! sorted_token_ids, Tensor! experts_ids, "
        "Tensor! num_tokens_post_pad, Tensor? maybe_expert_map) -> ()"
    ),
    "_moe_C::topk_softplus_sqrt": (
        "topk_softplus_sqrt(Tensor! topk_weights, Tensor! topk_indices, "
        "Tensor! token_expert_indices, Tensor gating_output, bool renormalize, "
        "float routed_scaling_factor, Tensor? bias, Tensor? input_ids, "
        "Tensor? tid2eid) -> ()"
    ),
    "_C::dynamic_scaled_int8_quant": (
        "dynamic_scaled_int8_quant(Tensor! result, Tensor input, "
        "Tensor! scale, Tensor!? azp) -> ()"
    ),
    "_C::silu_and_mul": "silu_and_mul(Tensor! result, Tensor input) -> ()",
}

# Native operators are mapped to the FL dispatch name after ``::`` by default.
# For example, ``_moe_C::moe_sum`` automatically dispatches to ``moe_sum`` and
# therefore does not need an entry below.  Add an override only when the native
# ABI differs from the existing logical FL operator, such as an out-parameter
# interface versus a tensor-returning interface.  The dedicated ``*_native``
# name keeps the two Python call signatures separate, while dispatch policy
# lets it inherit the backend order configured for the logical base operator.
_NATIVE_OP_DISPATCH_NAMES = {
    "_C::silu_and_mul": "silu_and_mul_native",
    "_moe_C::moe_align_block_size": "moe_align_block_size_native",
}


# Reuse the configured FL dispatch logger hierarchy.  Keeping this module under
# ``vllm_fl.ops`` would leave its early INFO messages to the root logger, which
# is not configured yet when native bridges are installed during startup.
logger = logging.getLogger("vllm_fl.dispatch.native_ops_registry")

# torch.library registrations disappear when their Library object is released.
_SCHEMA_LIBRARIES: dict[str, torch.library.Library] = {}
_LIBRARIES: dict[tuple[str, str], torch.library.Library] = {}
_REGISTERED: set[tuple[str, str]] = set()


def _split_name(name: str) -> tuple[str, str]:
    if name.count("::") != 1:
        raise ValueError(
            f"Native op name must have the form 'namespace::op', got {name!r}"
        )

    namespace, op = name.split("::")
    if not namespace or not op:
        raise ValueError(f"Invalid native op name: {name!r}")
    return namespace, op


def _schema_exists(name: str) -> bool:
    try:
        torch._C._dispatch_find_schema_or_throw(name, "")
    except RuntimeError:
        return False
    return True


def _has_kernel(name: str, dispatch_key: str) -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(name, dispatch_key)
    except RuntimeError:
        return False


def _ensure_schema(name: str, namespace: str) -> None:
    if _schema_exists(name):
        return

    schema = NATIVE_OP_SCHEMAS.get(name)
    if schema is None:
        raise RuntimeError(f"No bundled schema is available for native op {name}")

    if namespace not in _SCHEMA_LIBRARIES:
        _SCHEMA_LIBRARIES[namespace] = torch.library.Library(namespace, "FRAGMENT")
    _SCHEMA_LIBRARIES[namespace].define(schema)
    logger.info("Registered fallback schema for native op %s", name)


def _get_library(namespace: str, dispatch_key: str) -> torch.library.Library:
    key = (namespace, dispatch_key)
    if key not in _LIBRARIES:
        _LIBRARIES[key] = torch.library.Library(namespace, "IMPL", dispatch_key)
    return _LIBRARIES[key]


def _make_bridge(op: str):
    def bridge(*args, **kwargs):
        # Lazy import avoids a cycle during platform/schema initialization.
        from vllm_fl.dispatch import call_op

        return call_op(op, *args, **kwargs)

    return bridge


def register_native_op(name: str, dispatch_key: str = "CUDA") -> bool:
    """Install one missing-kernel bridge; return whether it was installed."""
    namespace, op = _split_name(name)
    identity = (name, dispatch_key)
    if identity in _REGISTERED:
        return False

    _ensure_schema(name, namespace)

    if _has_kernel(name, dispatch_key):
        logger.info(
            "Keeping existing %s implementation for %s; native bridges only "
            "handle missing kernels",
            dispatch_key,
            name,
        )
        return False

    dispatch_name = _NATIVE_OP_DISPATCH_NAMES.get(name, op)
    _get_library(namespace, dispatch_key).impl(op, _make_bridge(dispatch_name))
    _REGISTERED.add(identity)
    logger.info(
        "Native op %s is now managed by FL dispatch as '%s'",
        name,
        dispatch_name,
    )
    return True


def register_native_ops(
    names: Iterable[str] | None = None,
    dispatch_key: str = "CUDA",
) -> tuple[str, ...]:
    """Register the supplied names, or every entry in ``_NATIVE_OPS``."""
    selected = tuple(names) if names is not None else tuple(_NATIVE_OPS)
    return tuple(name for name in selected if register_native_op(name, dispatch_key))


__all__ = [
    "register_native_op",
    "register_native_ops",
]

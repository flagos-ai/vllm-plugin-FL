# Copyright (c) 2026 BAAI. All rights reserved.

"""
Template backend operator registrations.

This module registers all VENDOR (Template) implementations.
Replace 'template' with your vendor name and register your operator implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all Template (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .template import TemplateBackend

    backend = TemplateBackend()
    is_avail = backend.is_available

    # TODO: Add your operator implementations here
    # Example:
    impls = [
        # OpImpl(
        #     op_name="silu_and_mul",
        #     impl_id="vendor.template",
        #     kind=BackendImplKind.VENDOR,
        #     fn=_bind_is_available(backend.silu_and_mul, is_avail),
        #     vendor="template",
        #     priority=BackendPriority.VENDOR,
        # ),
    ]

    registry.register_many(impls)

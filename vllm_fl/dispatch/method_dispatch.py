# Copyright (c) 2026 BAAI. All rights reserved.

"""
Descriptor-based method dispatch for operator implementations.

Allows operator classes to declare `forward_oot` as a descriptor that
automatically dispatches to the resolved backend implementation, with
the backend function bound as a method so `self` is naturally available.
"""

from __future__ import annotations


class dispatch_method:
    """
    Descriptor that dispatches to the resolved backend implementation.

    The backend function is bound as a method to the operator instance
    via ``types.MethodType``, so ``self`` is naturally available — just
    like vLLM's ``forward_cuda`` / ``forward_xpu`` pattern.

    Usage::

        class RMSNormFL(RMSNorm):
            forward_oot = dispatch_method("rms_norm")
    """

    def __init__(self, op_name: str) -> None:
        self.op_name = op_name

    def __set_name__(self, owner, name):
        self.attr_name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        def dispatched(*args, **kwargs):
            from vllm_fl.dispatch import get_default_manager
            return get_default_manager().call_as_method(
                self.op_name, obj, *args, **kwargs
            )

        return dispatched

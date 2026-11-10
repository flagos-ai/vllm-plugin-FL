# Copyright (c) 2026 BAAI. All rights reserved.

"""Patches for vLLM CustomOp dispatch on the Hygon OOT platform."""

import functools
import importlib
import logging


logger = logging.getLogger(__name__)


def patch_custom_op_dispatch() -> None:
    """Prefer explicit FL OOT implementations over ROCm on Hygon.

    Hygon reports both OOT and ROCm because it reuses vLLM's HIP control flow.
    Upstream checks ROCm first, so select ``forward_oot`` only for explicitly
    registered OOT classes that actually override it. Unregistered or disabled
    CustomOps keep their upstream HIP/native fallback.
    """
    custom_op_mod = importlib.import_module("vllm.model_executor.custom_op")
    custom_op_cls = custom_op_mod.CustomOp
    original = custom_op_cls.dispatch_forward
    if getattr(original, "_vllm_fl_hygon_oot_first", False):
        return

    @functools.wraps(original)
    def _dispatch_forward_hygon_oot_first(self, compile_native: bool):
        selected = original(self, compile_native)
        platform = custom_op_mod.current_platform
        if (
            str(getattr(platform, "vendor_name", "")).lower() != "hygon"
            or not platform.is_out_of_tree()
        ):
            return selected

        registration_name = getattr(type(self), "name", None)
        oot_cls = custom_op_mod.op_registry_oot.get(registration_name)
        if oot_cls is None or not isinstance(self, oot_cls):
            return selected
        if type(self).forward_oot is custom_op_cls.forward_oot:
            return selected

        selected_func = getattr(selected, "__func__", None)
        hip_func = getattr(self.forward_hip, "__func__", None)
        if selected_func is not hip_func:
            return selected
        return self.forward_oot

    _dispatch_forward_hygon_oot_first._vllm_fl_hygon_oot_first = True
    _dispatch_forward_hygon_oot_first._vllm_fl_original = original
    custom_op_cls.dispatch_forward = _dispatch_forward_hygon_oot_first
    logger.info("Patched Hygon CustomOp dispatch to prefer explicit FL OOT ops")


# Backward-compatible private name used by existing tests and callers.
_patch_custom_op_dispatch = patch_custom_op_dispatch

__all__ = ["patch_custom_op_dispatch"]

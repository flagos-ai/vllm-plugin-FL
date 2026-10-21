# Copyright (c) 2026 BAAI. All rights reserved.
"""Route vLLM's monolithic Marlin MoE kernel through OpManager."""

import logging
from functools import wraps
from types import ModuleType

from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import use_flaggems_op

logger = logging.getLogger(__name__)


_dispatch_fused_marlin_moe = CachedOp("fused_marlin_moe")


def patch_vllm_fused_marlin_moe(
    marlin_moe_module: ModuleType | None = None,
) -> bool:
    """Bridge vLLM's Marlin experts into the plugin dispatch manager.

    ``MarlinExperts.apply`` resolves ``fused_marlin_moe`` from its module on
    every invocation, so replacing that module global covers the modular MoE
    path without changing the vLLM package or its model implementation.
    """
    if not use_flaggems_op("fused_marlin_moe"):
        return False

    if marlin_moe_module is None:
        from vllm.model_executor.layers.fused_moe.experts import (
            marlin_moe as marlin_moe_module,
        )

    original = marlin_moe_module.fused_marlin_moe
    if getattr(original, "_vllm_fl_fused_marlin_moe_patch", False):
        return False

    @wraps(original)
    def fused_marlin_moe_flagos(*args, **kwargs):
        return _dispatch_fused_marlin_moe(*args, **kwargs)

    fused_marlin_moe_flagos._vllm_fl_fused_marlin_moe_patch = True
    fused_marlin_moe_flagos._vllm_fl_original = original
    marlin_moe_module.fused_marlin_moe = fused_marlin_moe_flagos
    logger.info("Bridged vLLM fused_marlin_moe -> vllm-plugin-FL dispatch manager")
    return True


__all__ = ["patch_vllm_fused_marlin_moe"]

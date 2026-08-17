# Copyright (c) 2026 BAAI. All rights reserved.
"""Route vLLM MoE reduction through the plugin dispatch manager."""

import logging
from functools import wraps
from types import ModuleType

from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import use_flaggems_op

logger = logging.getLogger(__name__)


_dispatch_moe_sum = CachedOp("moe_sum")


def _torch_moe_sum(input, output):
    """Stride-safe fallback for layouts unsupported by the FlagGems kernel."""
    reduced = input.float().sum(dim=1)
    output.copy_(reduced.to(dtype=output.dtype))


def patch_vllm_moe_sum(ops_module: ModuleType | None = None) -> bool:
    """Bridge ``vllm._custom_ops.moe_sum`` into the plugin OpManager.

    vLLM's MoE implementations resolve this function through the module on
    every call. The adapter therefore covers Triton/FP8 experts while leaving
    the vLLM wheel and its compiled extensions untouched. Backend selection,
    fallback, diagnostics, and per-op policy remain owned by OpManager.
    """
    if not use_flaggems_op("moe_sum"):
        return False

    if ops_module is None:
        import vllm._custom_ops as ops_module

    original = ops_module.moe_sum
    if getattr(original, "_vllm_fl_moe_sum_patch", False):
        return False

    @wraps(original)
    def moe_sum_flagos(input, output):
        # Triton does not launch a zero-sized grid.  This also mirrors the
        # zero-token guard required by the vLLM MoE call path.
        if input.numel() == 0 or output.numel() == 0:
            return None
        # FlagGems accepts arbitrary token/top-k strides but its current Triton
        # kernel assumes a contiguous hidden dimension for input and output.
        # Preserve non-contiguous correctness via a rare fallback.
        if input.stride(-1) != 1 or output.stride(-1) != 1:
            return _torch_moe_sum(input, output)
        return _dispatch_moe_sum(input, output)

    moe_sum_flagos._vllm_fl_moe_sum_patch = True
    moe_sum_flagos._vllm_fl_original = original
    ops_module.moe_sum = moe_sum_flagos
    logger.info("Bridged vLLM moe_sum -> vllm-plugin-FL dispatch manager")
    return True


__all__ = ["patch_vllm_moe_sum"]

# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for ``TritonExperts.moe_sum``.

The upstream ``moe_sum`` calls ``ops.moe_sum`` which is a CUDA-only custom op.
On GCU this op is not available and causes a RuntimeError.

This patch replaces the ``moe_sum`` method on ``TritonExperts`` (and
inheriting subclasses such as ``TritonWNA16Experts``) with a version that
uses ``CachedOp("moe_sum")`` from the FlagOS dispatch system, which selects
a GCU-compatible implementation.
"""

from __future__ import annotations

import logging

import torch

from vllm_fl.dispatch import CachedOp

logger = logging.getLogger(__name__)

_patched = False

_gcu_moe_sum = CachedOp("moe_sum")


def _gcu_moe_sum_impl(self, input: torch.Tensor, output: torch.Tensor) -> None:
    """GCU-compatible ``moe_sum`` via the FlagOS dispatch system."""
    _gcu_moe_sum(input, output)

def apply_moe_sum_gcu_patch() -> None:
    """Patch ``TritonExperts.moe_sum`` for GCU devices."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

        # Replace the method on the class so all instances (including
        # TritonWNA16Experts which inherits from TritonExperts) use the
        # GCU-compatible implementation.
        TritonExperts.moe_sum = _gcu_moe_sum_impl
        _patched = True
        logger.info(
            "Patched TritonExperts.moe_sum for GCU (using FlagOS dispatch)"
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch TritonExperts.moe_sum for GCU: %s",
            exc,
        )

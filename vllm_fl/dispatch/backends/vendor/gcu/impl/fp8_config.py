# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for Fp8Config.get_min_capability.

vLLM upstream requires CUDA compute capability >= 75 (Turing) for FP8 support.
GCU hardware supports FP8 with a lower capability (40), so we patch
``Fp8Config.get_min_capability`` to return 40 when running on GCU.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False

GCU_MIN_CAPABILITY = 40


def _gcu_get_min_capability(cls) -> int:
    return GCU_MIN_CAPABILITY


def apply_fp8_config_gcu_patch() -> None:
    """Patch Fp8Config.get_min_capability to accept GCU hardware (capability 40)."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        Fp8Config.get_min_capability = classmethod(_gcu_get_min_capability)
        _patched = True
        logger.info(
            "Patched Fp8Config.get_min_capability for GCU (min capability: %d)",
            GCU_MIN_CAPABILITY,
        )
    except Exception as exc:
        logger.warning("Failed to patch Fp8Config for GCU: %s", exc)

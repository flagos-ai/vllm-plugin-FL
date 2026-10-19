# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for ``apply_moe_activation``.

The upstream ``apply_moe_activation`` directly calls ``torch.ops._C.*`` custom
ops (e.g. ``silu_and_mul``, ``gelu_and_mul``) that are CUDA-only.  On GCU
these ops are not available.

The vllm_fl variant (:func:`vllm_fl.ops.fused_moe.activation.apply_moe_activation`)
uses ``CachedOp("silu_and_mul")`` / ``CachedOp("gelu_and_mul")``, which
dispatches through the FlagOS system and selects a GCU-compatible
implementation.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def apply_moe_activation_gcu(
    activation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """GCU-compatible ``apply_moe_activation`` via the FlagOS dispatch system."""
    from vllm_fl.ops.fused_moe.activation import apply_moe_activation as _fl_impl

    return _fl_impl(activation, output, input)


def apply_moe_activation_gcu_patch() -> None:
    """Patch ``apply_moe_activation`` for GCU devices."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    # Modules that hold a reference to apply_moe_activation via
    # ``from ...activation import apply_moe_activation``.
    _IMPORTERS: list[str] = [
        "vllm.model_executor.layers.fused_moe.activation",
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.experts.cutlass_moe",
        "vllm.model_executor.layers.fused_moe.fused_marlin_moe",
        "vllm.model_executor.layers.quantization.gguf",
    ]

    try:
        for module_name in _IMPORTERS:
            try:
                mod = __import__(module_name, fromlist=["apply_moe_activation"])
            except ImportError:
                continue
            if hasattr(mod, "apply_moe_activation"):
                mod.apply_moe_activation = apply_moe_activation_gcu

        _patched = True
        logger.info(
            "Patched apply_moe_activation for GCU (using FlagOS dispatch)"
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch apply_moe_activation for GCU: %s",
            exc,
        )

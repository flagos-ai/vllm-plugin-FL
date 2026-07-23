# Copyright (c) 2026 BAAI. All rights reserved.
"""Keep vLLM's native Marlin WNA16 MoE path NVIDIA-only."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MARLIN_GUARD_MARKER = "_vllm_fl_marlin_platform_guard"


def is_marlin_moe_platform(platform: Any) -> bool:
    """Return whether vLLM's PTX Marlin MoE backend is valid."""
    return bool(platform.is_cuda())


def _install_platform_guard() -> None:
    """Prevent vLLM 0.20.2 from choosing Marlin on non-NVIDIA OOT devices."""
    from vllm.platforms import current_platform

    moe_module = import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe"
    )
    current_check = moe_module.check_moe_marlin_supports_layer
    if getattr(current_check, _MARLIN_GUARD_MARKER, False):
        return

    def guarded_check(layer, group_size):
        from vllm_fl.quantization.wna16.kernels import (
            is_wna16_moe_available,
        )

        return (
            not is_wna16_moe_available()
            and is_marlin_moe_platform(current_platform)
            and current_check(layer, group_size)
        )

    setattr(guarded_check, _MARLIN_GUARD_MARKER, True)
    moe_module.check_moe_marlin_supports_layer = guarded_check


def configure_wna16_moe_backend() -> str:
    """Guard upstream selection and otherwise leave vLLM's backend untouched."""
    from vllm.platforms import current_platform

    from vllm_fl.quantization.wna16.kernels import (
        is_wna16_moe_available,
    )

    _install_platform_guard()
    if is_wna16_moe_available():
        return "plugin-local"
    if not is_marlin_moe_platform(current_platform):
        return "generic"
    return "marlin"


__all__ = [
    "configure_wna16_moe_backend",
    "is_marlin_moe_platform",
]

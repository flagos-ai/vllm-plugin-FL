# Copyright (c) 2026 BAAI. All rights reserved.

"""
SUPA backend implementation.

This backend provides operator implementations for Biren SUPA GPUs.
"""

from __future__ import annotations

import torch

from vllm_fl.dispatch.backends.base import Backend


class SupaBackend(Backend):
    """Expose SUPA-specific dispatch choices without using NVIDIA kernels."""

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "supa"

    @property
    def vendor(self) -> str:
        return "supa"

    def is_available(self) -> bool:
        if SupaBackend._available is None:
            try:
                import torch_supa  # noqa: F401

                from vllm.platforms import current_platform

                SupaBackend._available = (
                    getattr(current_platform, "vendor_name", None) == "biren"
                    and hasattr(torch, "supa")
                    and torch.supa.is_available()
                    and torch.supa.device_count() > 0
                )
            except (ImportError, AttributeError, OSError, RuntimeError):
                SupaBackend._available = False
        return SupaBackend._available

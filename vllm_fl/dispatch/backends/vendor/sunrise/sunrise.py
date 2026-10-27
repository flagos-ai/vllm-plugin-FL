# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise backend implementation.

This backend provides operator implementations for Sunrise GPUs.

"""

from __future__ import annotations

import torch

from vllm_fl.dispatch.backends.base import Backend


class SunriseBackend(Backend):
    """
    Sunrise backend for operator implementations.

    """

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "sunrise"

    @property
    def vendor(self) -> str | None:
        return "sunrise"

    def is_available(self) -> bool:
        """
        Check if the sunrise hardware and libraries are available.

        Implement this method to detect if your vendor's hardware/software
        is present and functional.
        """
        if SunriseBackend._available is None:
            ptpu = getattr(torch, "ptpu", None)
            try:
                SunriseBackend._available = bool(
                    ptpu is not None and ptpu.is_available() and ptpu.device_count() > 0
                )
            except (AttributeError, RuntimeError):
                SunriseBackend._available = False
        return SunriseBackend._available

    # ==================== Operator Implementations ====================
    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Sunrise.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            raise NotImplementedError("NOT support mla now!")

        if use_sparse:
            raise ValueError("use_sparse=True requires use_mla=True.")

        return (
            "vllm_fl.dispatch.backends.vendor.sunrise.impl.attention.AttentionFLBackend"
        )

# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend implementation.

This backend provides operator implementations using the FlagGems library.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class FlagGemsBackend(Backend):
    """
    FlagGems backend for operator implementations.

    This backend uses the flag_gems library to provide high-performance
    operator implementations.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "flagos"

    def is_available(self) -> bool:
        """Check if FlagGems is available."""
        if FlagGemsBackend._available is None:
            try:
                import flag_gems  # noqa F401

                FlagGemsBackend._available = True
            except ImportError:
                FlagGemsBackend._available = False
        return FlagGemsBackend._available

    # ==================== Operator Implementations ====================

    def attention_backend(self, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for FlagGems.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.attention.backends.registry import AttentionBackendEnum

        # TritonAttentionBackend requires CUDA, check if available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TritonAttentionBackend requires CUDA but CUDA is not available. "
                "Falling back to vendor implementation."
            )

        if use_mla:
            raise NotImplementedError("NOT support mla now!")

        return AttentionBackendEnum.TRITON_ATTN.get_path()

# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda backend implementation.

This backend provides operator implementations for TXDA devices.
"""

from __future__ import annotations

from typing import Optional

import torch
from vllm_fl.dispatch.backends.base import Backend

class TxdaBackend(Backend):
    """
    Txda backend for operator implementations.

    This backend routes dispatch ops to the TXDA device implementations:
    attention and MoE go through the FlagGems TX8 stack, normalization
    through the reference torch fallback in ``impl.normalization``.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "txda"

    @property
    def vendor(self) -> Optional[str]:
        return "txda"

    def is_available(self) -> bool:
        """Check if TXDA hardware and torch_txda are available."""
        if TxdaBackend._available is None:
            try:
                # Check for torch_txda (TXDA PyTorch extension)
                import torch_txda

                # Check if TXDA device is available
                if torch.txda.is_available() and torch.txda.device_count() > 0:
                    TxdaBackend._available = True
                else:
                    TxdaBackend._available = False
            except (ImportError, AttributeError):
                TxdaBackend._available = False
        return TxdaBackend._available

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ):
        from .impl.normalization import rms_norm_txda

        return rms_norm_txda(obj, x, residual)

    def attention_backend(
        self, use_mla: bool = False, use_sparse: bool = False
    ) -> str:
        """
        Get the attention backend class path for TXDA.

        Returns the FlagGems attention backend, which calls
        ``flag_gems.flash_attn_varlen_func`` on the TX8 stack. Validated
        end-to-end on Qwen3.6-27B and Qwen3.6-35B-A3B with flaggems_vllm 0.3.0.dev1 + triton
        3.7.1 (the 0.28-era silently-wrong flash_attn results do not
        reproduce on this stack).

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use sparse attention (unsupported)

        Returns:
            Fully qualified class path string
        """
        if use_sparse:
            raise ValueError("use_sparse=True is not supported on Txda.")
        if use_mla:
            return "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"
        return "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"

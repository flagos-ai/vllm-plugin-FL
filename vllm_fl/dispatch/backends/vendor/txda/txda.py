# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (tsingmicro) backend implementation.

This backend provides operator implementations for Tsingmicro TX devices.
For attention it uses the FlagGems attention backend.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class TxdaBackend(Backend):
    """
    Txda (tsingmicro) backend for operator implementations.

    Tsingmicro TX devices use torch_txda (a PrivateUse1-based runtime), so the
    CUDA-style fused ops are not applicable; dispatch falls back to the
    flag_gems implementations. Only the attention backend is registered here.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "txda"

    @property
    def vendor(self) -> Optional[str]:
        return "txda"

    def is_available(self) -> bool:
        """
        Check if tsingmicro TX hardware is available.

        Detection is based on the torch_txda runtime.
        """
        if TxdaBackend._available is None:
            try:
                import torch_txda  # noqa: F401
                TxdaBackend._available = (
                    torch.txda.is_available() and torch.txda.device_count() > 0
                )
            except Exception:
                TxdaBackend._available = False
        return TxdaBackend._available

    # ==================== Operator Implementations ====================

    def attention_backend(
        self, use_mla: bool = False, use_sparse: bool = False
    ) -> str:
        """
        Get the attention backend class path for tsingmicro TX.

        Returns the txda SDPA backend (reuses the flag_gems metadata machinery
        but computes attention with torch SDPA, which is numerically correct on
        TX8110 where flag_gems flash_attn_varlen_func is not). The MLA branch
        still points at the flag_gems MLA backend; MLA is unverified on TX8110.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            return "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"
        return (
            "vllm_fl.dispatch.backends.vendor.txda.impl.attention."
            "TxdaSDPAAttentionBackend"
        )

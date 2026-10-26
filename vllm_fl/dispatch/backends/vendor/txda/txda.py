# Copyright (c) 2026 BAAI. All rights reserved.
# Ported from the vLLM 0.24 TXDA enablement (537284eb, #460) for vLLM 0.28.

"""
Txda (tsingmicro) backend implementation.

This backend provides operator implementations for Tsingmicro TX devices.
Attention uses the TXDA SDPA backend with FlagGems metadata.
"""

from __future__ import annotations

import torch

from vllm_fl.dispatch.backends.base import Backend


class TxdaBackend(Backend):
    """
    Txda (tsingmicro) backend for operator implementations.

    Tsingmicro TX devices use torch_txda (a PrivateUse1-based runtime), so the
    CUDA-style fused ops are not applicable. Only the SDPA attention backend
    is registered here; normalization and rotary use the reference backend.
    """

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "txda"

    @property
    def vendor(self) -> str | None:
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

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for tsingmicro TX.

        Returns the txda SDPA backend (reuses the flag_gems metadata machinery
        but computes attention with torch SDPA, which is numerically correct on
        TX8110 where flag_gems flash_attn_varlen_func is not). MLA and sparse
        attention are unsupported and must not silently select a CUDA kernel.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla or use_sparse:
            raise NotImplementedError(
                "TXDA SDPA does not support MLA or sparse attention"
            )
        return (
            "vllm_fl.dispatch.backends.vendor.txda.impl.attention."
            "TxdaSDPAAttentionBackend"
        )

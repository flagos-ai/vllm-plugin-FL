# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda backend implementation.

This backend provides operator implementations for Tsingmicro TX devices.
Attention is served by the torch-SDPA backend in impl/attention.py; the
flag_gems attention kernels compute wrong values on TX8110.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class TxdaBackend(Backend):
    """
    Txda backend for operator implementations.

    This backend uses Tsingmicro TX libraries to provide operator
    implementations for Tsingmicro TX devices.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "txda"

    @property
    def vendor(self) -> Optional[str]:
        return "txda"

    def is_available(self) -> bool:
        """Check if Txda hardware and libraries are available."""
        if TxdaBackend._available is None:
            try:
                import torch_txda  # noqa: F401  (registers the torch.txda namespace)

                TxdaBackend._available = (
                    torch.txda.is_available() and torch.txda.device_count() > 0
                )
            except Exception:
                TxdaBackend._available = False
        return TxdaBackend._available

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Txda devices.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use sparse attention (unsupported here)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            return "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"
        return "vllm_fl.dispatch.backends.vendor.txda.impl.attention.TxdaSDPAAttentionBackend"

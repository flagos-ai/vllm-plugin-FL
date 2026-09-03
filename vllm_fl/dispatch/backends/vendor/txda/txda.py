# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda backend implementation.

This backend provides operator implementations for Tsingmiocro Txda NPUs.
"""

from __future__ import annotations

from typing import Optional

import torch

from torch_txda import transfer_to_txda

# from vllm_fl.dispatch.backends.flaggems import FlagGemsBackend
from vllm_fl.dispatch.backends.base import Backend

class TxdaBackend(Backend):
    """
    Txda backend for operator implementations.

    This backend uses Txda CANN libraries to provide high-performance
    operator implementations for Tsingmiocro Txda NPUs.
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
                # Check for torch_npu (Txda PyTorch extension)
                import torch_txda

                # Check if NPU device is available
                if torch.txda.is_available() and torch.txda.device_count() > 0:
                    TxdaBackend._available = True
                else:
                    TxdaBackend._available = False
            except (ImportError, AttributeError):
                TxdaBackend._available = False
        return TxdaBackend._available

    def attention_backend(self, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for Txda NPU.

        This method returns the native Txda attention backend that uses
        torch_npu operators (npu_fused_infer_attention_score, etc.)
        instead of flag_gems operators.

        Uses vllm_fl's native Txda implementation which directly calls
        torch_npu operators without depending on vllm-Txda package.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            return "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"
        # return "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
        return "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
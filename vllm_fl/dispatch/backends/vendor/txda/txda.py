# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda backend implementation.

This backend provides operator implementations for Tsingmiocro Txda NPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

# from vllm_fl.dispatch.backends.flaggems import FlagGemsBackend
from vllm_fl.dispatch.backends.base import Backend
from torch_txda import transfer_to_txda

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

    # ==================== Operator Implementations ====================

    # def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     SiLU activation followed by element-wise multiplication.

    #     Args:
    #         obj: The calling obj (for interface consistency)
    #         x: Input tensor of shape [..., 2*d]

    #     Returns:
    #         Output tensor of shape [..., d]
    #     """
    #     from .impl.activation import silu_and_mul_Txda

    #     return silu_and_mul_Txda(obj, x)

    # def rms_norm(
    #     self,
    #     obj,
    #     x: torch.Tensor,
    #     residual: Optional[torch.Tensor] = None,
    # ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    #     """
    #     RMS normalization.

    #     Args:
    #         obj: The calling obj (e.g., RMSNorm layer)
    #         x: Input tensor
    #         residual: Optional residual tensor

    #     Returns:
    #         Normalized tensor, or tuple of (normalized, residual) if residual is provided
    #     """
    #     from .impl.normalization import rms_norm_Txda

    #     return rms_norm_Txda(obj, x, residual)

    # def rotary_embedding(
    #     self,
    #     obj,
    #     query: torch.Tensor,
    #     key: torch.Tensor,
    #     cos: torch.Tensor,
    #     sin: torch.Tensor,
    #     position_ids: torch.Tensor,
    #     rotary_interleaved: bool = False,
    #     inplace: bool = True,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Apply rotary position embedding.

    #     Args:
    #         obj: The calling obj (for interface consistency)
    #         query: Query tensor
    #         key: Key tensor
    #         cos: Cosine cache
    #         sin: Sine cache
    #         position_ids: Position indices
    #         rotary_interleaved: Whether to use interleaved rotary
    #         inplace: Whether to modify tensors in-place

    #     Returns:
    #         Tuple of (embedded_query, embedded_key)
    #     """
    #     from .impl.rotary import rotary_embedding_Txda

    #     return rotary_embedding_Txda(
    #         obj,
    #         query,
    #         key,
    #         cos,
    #         sin,
    #         position_ids,
    #         rotary_interleaved=rotary_interleaved,
    #         inplace=inplace,
    #     )

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
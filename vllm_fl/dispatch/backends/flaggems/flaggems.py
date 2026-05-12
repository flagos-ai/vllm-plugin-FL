# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend implementation.

This backend provides operator implementations using the FlagGems library.
"""

from __future__ import annotations

from typing import Optional, Union

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

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_flaggems

        return silu_and_mul_flaggems(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_flaggems

        return gelu_and_mul_flaggems(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_flaggems

        return rms_norm_flaggems(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_flaggems

        return rotary_embedding_flaggems(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for FlagGems.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

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

        if use_sparse:
            raise ValueError("use_sparse=True requires use_mla=True.")

        return AttentionBackendEnum.TRITON_ATTN.get_path()

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: Optional[torch.Tensor] = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ):
        from .impl.fused_moe import moe_align_block_size_flaggems

        return moe_align_block_size_flaggems(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    def moe_sum(self, inp, out):
        from .impl.fused_moe import moe_sum_flaggems

        moe_sum_flaggems(inp, out)

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from .impl.fused_moe import topk_softmax_flaggems

        return topk_softmax_flaggems(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )

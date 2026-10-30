# Copyright (c) 2026 BAAI. All rights reserved.

"""
Backend base interface definitions for vllm-plugin-FL dispatch.

This module defines the abstract base class that all backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


class VLLMFLBackendBase(ABC):
    """
    Abstract base class for vllm-plugin-FL operator backends.

    Each backend provides implementations for a set of operators.
    Backends should implement is_available() to indicate whether
    the backend can be used in the current environment.

    All operator methods should be implemented by concrete backend classes.
    Methods that are not supported should raise NotImplementedError.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available in the current environment.

        Returns:
            True if the backend can be used, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            Backend name string.
        """
        pass

    @property
    def vendor(self) -> Optional[str]:
        """
        Get the vendor name for this backend (if applicable).

        Returns:
            Vendor name string, or None for non-vendor backends.
        """
        return None

    # ==================== Quantization Operators ====================

    # Optional logical operators: unsupported backends need not override these.
    def modelslim_w8a8_prepare_weight(
        self, weight: torch.Tensor, weight_scale: torch.Tensor, nz_mode: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare logical INT8 [K,N] weights and per-output-channel scales."""
        raise NotImplementedError

    def modelslim_w8a8_dynamic_linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dynamic INT8 Linear: floating [T,K] input and prepared [K,N] weight."""
        raise NotImplementedError

    def modelslim_w8a8_prepare_experts(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare INT8 [E,N,K] experts and channel scales [E,N,1]."""
        raise NotImplementedError

    def modelslim_w8a8_experts(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        first_expert: int = 0,
        global_num_experts: int = -1,
        activation: str = "silu",
        limit: float | None = None,
    ) -> torch.Tensor:
        """Return this local expert range's weighted [T,K] contribution."""
        raise NotImplementedError

    def topk_softplus_sqrt(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        e_score_correction_bias: torch.Tensor | None = None,
        indices_type: torch.dtype | None = None,
        input_tokens: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts with optional hash/bias; return unscaled FP32 weights."""
        raise NotImplementedError

    def deepseek_v4_partial_rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        """Apply adjacent-pair RoPE to trailing head dimensions."""
        raise NotImplementedError

    def deepseek_v4_hadamard(
        self,
        x: torch.Tensor,
        matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply padded Hadamard projection then normalized dtype-round scaling."""
        raise NotImplementedError

    def deepseek_v4_index_quant(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return same-shape INT8 rows and FP16 scales [*leading,1]."""
        raise NotImplementedError

    def deepseek_v4_grouped_project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        groups: int,
    ) -> torch.Tensor:
        """Project [T,H,D] using grouped logical weights [groups*out,in]."""
        raise NotImplementedError

    def deepseek_v4_project(
        self,
        x: torch.Tensor,
        wkv: torch.Tensor,
        wgate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute compressor KV/gate projections with FP32 accumulation/output."""
        raise NotImplementedError

    def deepseek_v4_hc_pre(
        self,
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """BF16 HC input [T,C,D] -> layer [T,D], FP32 post [T,C]/mix [T,C,C]."""
        raise NotImplementedError

    def deepseek_v4_hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Combine layer output with HC residual streams, preserving mix order."""
        raise NotImplementedError

    def deepseek_v4_hc_head(
        self,
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        """Collapse HC streams [T,C,D] to [T,D] with FP32 mixing parameters."""
        raise NotImplementedError

    def deepseek_v4_cache_write(
        self,
        values: torch.Tensor,
        slots: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        """Mutate paged [P,B,D] cache at unique valid slots; negative slots skip."""
        raise NotImplementedError

    def deepseek_v4_save_state(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        ape: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        state_cache: torch.Tensor,
        compress_ratio: int,
    ) -> None:
        """Store FP32 projected KV/gate+APE in continuous paged state."""
        raise NotImplementedError

    def deepseek_v4_compress(
        self,
        state_cache: torch.Tensor,
        block_table: torch.Tensor,
        request_ids: torch.Tensor,
        positions: torch.Tensor,
        rms_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        compress_ratio: int = 4,
        rms_eps: float = 1e-6,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Compress completed original-token boundaries; padded rows return zero."""
        raise NotImplementedError

    def deepseek_v4_indexer(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        request_ids: torch.Tensor,
        positions: torch.Tensor,
        key_lengths: torch.Tensor,
        topk: int,
        compress_ratio: int = 4,
        query_scale: torch.Tensor | None = None,
        key_scale_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return causal C4 logical compact IDs [T,topk] INT32, padding with -1."""
        raise NotImplementedError

    def deepseek_v4_sparse_attention(
        self,
        query: torch.Tensor,
        swa_cache: torch.Tensor,
        compact_cache: torch.Tensor | None,
        swa_block_table: torch.Tensor,
        compact_block_table: torch.Tensor | None,
        request_ids: torch.Tensor,
        positions: torch.Tensor,
        swa_lengths: torch.Tensor,
        compact_lengths: torch.Tensor | None,
        compact_indices: torch.Tensor | None,
        window_size: int,
        compress_ratio: int = 4,
        scale: float | None = None,
        attn_sink: torch.Tensor | None = None,
        value_dim: int | None = None,
    ) -> torch.Tensor:
        """Joint SWA/compact normalization with optional zero-value sink."""
        raise NotImplementedError

    @abstractmethod
    def dynamic_per_token_quant_int8(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamically quantize each token to symmetric INT8.

        Args:
            x: Floating-point input shaped ``[tokens, hidden_size]``.

        Returns:
            Tuple of the INT8 output and FP32 per-token scales.
        """
        pass

    # ==================== Activation Operators ====================

    @abstractmethod
    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        pass

    @abstractmethod
    def gelu_and_mul(self, x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            x: Input tensor of shape [..., 2*d]
            approximate: GELU approximation mode ("none" or "tanh")

        Returns:
            Output tensor of shape [..., d]
        """
        pass

    # ==================== Normalization Operators ====================

    @abstractmethod
    def rms_norm(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
        epsilon: float,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            x: Input tensor
            residual: Optional residual tensor
            weight: Normalization weight
            epsilon: Small constant for numerical stability

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        pass

    # ==================== Position Embedding Operators ====================

    @abstractmethod
    def rotary_embedding(
        self,
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
        pass

    # ==================== Attention Backend ====================

    @abstractmethod
    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for this platform.

        This method returns the fully qualified class path of the attention
        backend implementation suitable for the current hardware platform.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use sparse attention

        Returns:
            Fully qualified class path string, e.g.:
            - "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend"
            - "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
        """
        pass

# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin backend implementation.

This backend provides operator implementations for Kunlunxin hardware.
Kunlunxin operators are exposed via xtorch_ops (primary). The Kunlunxin hardware masquerades as a CUDA device
via torch_xmlir.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class KunlunxinBackend(Backend):
    """
    Kunlunxin backend for operator implementations.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "kunlunxin"

    @property
    def vendor(self) -> Optional[str]:
        return "kunlunxin"

    def is_available(self) -> bool:
        """Check if Kunlunxin hardware and libraries are available."""
        if KunlunxinBackend._available is None:
            try:
                import torch_xmlir  # noqa: F401
                # Kunlunxin hardware masquerades as CUDA device
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    KunlunxinBackend._available = True
                else:
                    KunlunxinBackend._available = False
            except (ImportError, AttributeError):
                KunlunxinBackend._available = False
        return KunlunxinBackend._available

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
        from .impl.activation import silu_and_mul_kunlunxin

        return silu_and_mul_kunlunxin(obj, x)

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
        from .impl.normalization import rms_norm_kunlunxin

        return rms_norm_kunlunxin(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_kunlunxin

        return rotary_embedding_kunlunxin(
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
        Get the attention backend class path for Kunlunxin hardware.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            if use_sparse:
                raise NotImplementedError("MLA with sparse attention is not implemented for Kunlunxin yet.")
            raise NotImplementedError("MLA attention is not implemented for Kunlunxin yet.")
        return "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention.KunlunxinAttentionBackend"

    # ==================== FLA Operator Implementations ====================

    def chunk_gated_delta_rule_fwd(self, *args, **kwargs):
        """Chunk gated delta rule forward (prefill path)."""
        from .impl.fla.chunk import chunk_gated_delta_rule_fwd

        return chunk_gated_delta_rule_fwd(*args, **kwargs)

    def fused_recurrent_gated_delta_rule_fwd(self, *args, **kwargs):
        """Fused recurrent gated delta rule forward (decode path)."""
        from .impl.fla.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

        return fused_recurrent_gated_delta_rule_fwd(*args, **kwargs)

    # ==================== Causal Conv1d Implementations ====================

    def causal_conv1d_fn(self, *args, **kwargs):
        """Causal conv1d forward (prefill path)."""
        from .impl.causal_conv1d import causal_conv1d_fn_kunlunxin

        return causal_conv1d_fn_kunlunxin(*args, **kwargs)

    def causal_conv1d_update(self, *args, **kwargs):
        """Causal conv1d update (decode path)."""
        from .impl.causal_conv1d import causal_conv1d_update_kunlunxin

        return causal_conv1d_update_kunlunxin(*args, **kwargs)

    # ==================== Fused GDN Gating ====================

    def fused_gdn_gating(self, *args, **kwargs):
        """Fused GDN gating."""
        from .impl.fused_gdn_gating import fused_gdn_gating_kunlunxin

        return fused_gdn_gating_kunlunxin(*args, **kwargs)

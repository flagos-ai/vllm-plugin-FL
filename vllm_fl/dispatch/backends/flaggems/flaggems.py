# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend implementation.

This backend provides operator implementations using the FlagGems library.
"""

from __future__ import annotations

import os
from typing import ClassVar, Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class FlagGemsBackend(Backend):
    """
    FlagGems backend for operator implementations.

    This backend uses the flag_gems library to provide high-performance
    operator implementations.
    """

    _available: Optional[bool] = None
    _fused_op_availability: ClassVar[dict[str, bool]] = {}

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

    def fused_op_is_available(self, op_name: str) -> bool:
        """Check one optional FlagGems fused operator without coupling ABIs."""
        if op_name not in FlagGemsBackend._fused_op_availability:
            try:
                from flag_gems import fused

                available = callable(getattr(fused, op_name, None))
            except ImportError:
                available = False
            FlagGemsBackend._fused_op_availability[op_name] = available
        return FlagGemsBackend._fused_op_availability[op_name]

    # ==================== Operator Implementations ====================

    def mla_prefill(self, **kwargs):
        from .impl.mla_prefill import mla_prefill_flaggems

        return mla_prefill_flaggems(**kwargs)

    def dynamic_per_token_quant_int8(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.quantization import dynamic_per_token_quant_int8_flaggems_vllm

        return dynamic_per_token_quant_int8_flaggems_vllm(x)

    def dynamic_per_token_quant_int8_triton(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.quantization import dynamic_per_token_quant_int8_flaggems_triton

        return dynamic_per_token_quant_int8_flaggems_triton(x)

    def bf16_indexer_cache_write(
        self,
        keys: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        from .impl.bf16_indexer import bf16_indexer_cache_write_flaggems

        bf16_indexer_cache_write_flaggems(keys, cache, slot_mapping)

    def bf16_indexer_decode(self, *args, **kwargs) -> None:
        from .impl.bf16_indexer import bf16_indexer_decode_flaggems

        bf16_indexer_decode_flaggems(*args, **kwargs)

    def qsa_mqa_paged(self, *args, **kwargs):
        from flag_gems.fused import qsa_mqa_paged

        return qsa_mqa_paged(*args, **kwargs)

    def expand_qsa_block_indices(self, *args, **kwargs):
        from flag_gems.fused import expand_qsa_block_indices

        return expand_qsa_block_indices(*args, **kwargs)

    def qsa_select_paged_tokens(self, *args, **kwargs):
        from flag_gems.fused import qsa_select_paged_tokens

        return qsa_select_paged_tokens(*args, **kwargs)

    def qsa_sparse_paged_attention(self, *args, **kwargs):
        from flag_gems.fused import qsa_sparse_paged_attention

        return qsa_sparse_paged_attention(*args, **kwargs)

    def qsa_store_cache_rows(self, *args, **kwargs):
        from flag_gems.fused import qsa_store_cache_rows

        return qsa_store_cache_rows(*args, **kwargs)

    def qsa_compress_groups_with_ratio(self, *args, **kwargs):
        from flag_gems.fused import qsa_compress_groups_with_ratio

        return qsa_compress_groups_with_ratio(*args, **kwargs)

    def ple_state_gather(self, *args, **kwargs):
        from flag_gems.fused import ple_state_gather

        return ple_state_gather(*args, **kwargs)

    def ple_state_scatter_(self, *args, **kwargs):
        from flag_gems.fused import ple_state_scatter_

        return ple_state_scatter_(*args, **kwargs)

    def gdn_packed_decode(self, *args, **kwargs):
        from flag_gems.fused import gdn_packed_decode

        return gdn_packed_decode(*args, **kwargs)

    def compute_common_slot_mapping(self, *args, **kwargs):
        from .impl.common_slot_mapping import compute_common_slot_mapping_flaggems

        return compute_common_slot_mapping_flaggems(*args, **kwargs)

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
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            if use_sparse:
                return "vllm_fl.dispatch.backends.flaggems.impl.mla_sparse.MLASparseFLBackend"
            return "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"

        if use_sparse:
            raise ValueError("use_sparse=True requires use_mla=True.")

        # The generic Triton attention backend below is CUDA-specific. MLA
        # paths above are implemented through FlagGems and remain available on
        # every accelerator supported by that library.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TritonAttentionBackend requires CUDA but CUDA is not available. "
                "Falling back to vendor implementation."
            )

        use_flaggems_attn = os.environ.get(
            "VLLM_FL_USE_FLAGGEMS_ATTN", "0"
        ).lower() in ("1", "true", "yes")

        if use_flaggems_attn:
            print("Using FlagGems attention backend.")
            return "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"

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

    def invoke_fused_moe_triton_kernel(
        self,
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=None,
        B_bias=None,
    ):
        from .impl.fused_moe import invoke_fused_moe_triton_kernel_flaggems

        invoke_fused_moe_triton_kernel_flaggems(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape=block_shape,
            B_bias=B_bias,
        )

    def grouped_topk(
        self,
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func=0,
    ):
        from .impl.fused_moe import grouped_topk_flaggems

        return grouped_topk_flaggems(
            scores, n_group, topk_group, topk,
            renormalize, routed_scaling_factor, bias, scoring_func,
        )

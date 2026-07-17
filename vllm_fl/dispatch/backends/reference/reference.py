# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend implementation using PyTorch.

This backend provides reference operator implementations using native PyTorch
operations. These implementations are always available when PyTorch is installed
and serve as fallback implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class ReferenceBackend(Backend):
    """
    Reference backend for operator implementations.

    This backend uses native PyTorch operations to provide reference
    implementations that are always available as fallbacks.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "reference"

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        if ReferenceBackend._available is None:
            try:
                import torch

                ReferenceBackend._available = True
            except ImportError:
                ReferenceBackend._available = False
        return ReferenceBackend._available

    # ==================== Operator Implementations ====================

    def fused_marlin_moe(self, *args, **kwargs) -> torch.Tensor:
        """Call vLLM's native Marlin MoE implementation as a fallback."""
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            fused_marlin_moe,
        )

        return fused_marlin_moe(*args, **kwargs)

    def router_gemm_bf16_fp32(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Run the MoE router GEMM with an FP32 output."""
        from .impl.router_gemm import router_gemm_bf16_fp32_torch

        return router_gemm_bf16_fp32_torch(x, weight)

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_torch

        return silu_and_mul_torch(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_torch

        return gelu_and_mul_torch(obj, x)

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
        from .impl.normalization import rms_norm_torch

        return rms_norm_torch(obj, x, residual)

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
            inplace: Whether to modify tensors in-place (ignored in reference impl)

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_torch

        return rotary_embedding_torch(
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
        Get the attention backend class path for reference (vLLM native).

        This method returns the vLLM native flash attention backend path,
        which serves as a fallback implementation.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string (vLLM native backend)
        """
        # Return vLLM's native flash attention backend as reference
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            # vLLM native MLA backend
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: Optional[torch.Tensor] = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ):
        from .impl.fused_moe import moe_align_block_size_torch

        return moe_align_block_size_torch(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    def moe_sum(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        from .impl.fused_moe import moe_sum_torch

        moe_sum_torch(inp, out)

    def topk_softmax(
        self,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
    ):
        from .impl.fused_moe import topk_softmax_torch

        return topk_softmax_torch(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )

    def grouped_topk(
        self,
        scores: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
        bias: torch.Tensor,
        scoring_func: int = 0,
    ):
        from .impl.fused_moe import grouped_topk_torch

        return grouped_topk_torch(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
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
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            invoke_fused_moe_triton_kernel,
        )

        invoke_fused_moe_triton_kernel(
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

    def deepseek_v4_fp8_einsum(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        out: torch.Tensor,
        equation: str,
        recipe: list[int],
    ):
        from .impl.deepseek_v4_attn import deepseek_v4_fp8_einsum_torch

        deepseek_v4_fp8_einsum_torch(a, a_scale, b, b_scale, out, equation, recipe)

    def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache_2d: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
        block_size: int,
    ):
        from .impl.deepseek_v4_attn import (
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_torch,
        )

        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_torch(
            q, kv, swa_kv_cache_2d, slot_mapping, positions, cos_sin_cache,
            eps, block_size,
        )

    def combine_topk_swa_indices(
        self,
        topk_indices: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        topk_tokens: int,
        window_size: int,
        compress_ratio: int,
        block_size: int,
    ):
        from .impl.deepseek_v4_attn import combine_topk_swa_indices_torch

        combine_topk_swa_indices_torch(
            topk_indices, combined_indices, combined_lens,
            query_start_loc, seq_lens, block_table,
            topk_tokens, window_size, compress_ratio, block_size,
        )

    def compute_global_topk_indices_and_lens(
        self,
        topk_indices: torch.Tensor,
        global_indices: torch.Tensor,
        global_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        topk_tokens: int,
        compress_ratio: int,
        block_size: int,
    ):
        from .impl.deepseek_v4_attn import compute_global_topk_indices_and_lens_torch

        compute_global_topk_indices_and_lens_torch(
            topk_indices, global_indices, global_lens,
            query_start_loc, seq_lens, block_table,
            topk_tokens, compress_ratio, block_size,
        )

    def dequantize_and_gather_k_cache(
        self,
        k_cache: torch.Tensor,
        dst: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        block_size: int,
    ):
        from .impl.deepseek_v4_attn import dequantize_and_gather_k_cache_torch

        dequantize_and_gather_k_cache_torch(
            k_cache, dst, block_table, cu_seq_lens, block_size,
        )

    def fused_indexer_q_rope_quant(
        self,
        positions: torch.Tensor,
        index_q: torch.Tensor,
        index_q_cos_sin_cache: torch.Tensor,
        index_weights: torch.Tensor,
        index_weights_softmax_scale: float,
        index_weights_head_scale: float,
        use_fp4: bool = False,
    ):
        from .impl.deepseek_v4_attn import fused_indexer_q_rope_quant_torch

        return fused_indexer_q_rope_quant_torch(
            positions, index_q, index_q_cos_sin_cache,
            index_weights, index_weights_softmax_scale,
            index_weights_head_scale, use_fp4,
        )

    def fused_inv_rope_fp8_quant(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        heads_per_group: int,
        quant_group_size: int,
        chunks_per_head: int,
        rope_start: int,
        half_rope: int,
        tma_aligned_scales: bool,
        fp8_max: float,
        tma_aligned_T: int,
        num_tokens: int,
        n_groups: int,
        d: int,
        scale_inner: int,
    ):
        from .impl.deepseek_v4_attn import fused_inv_rope_fp8_quant_torch

        return fused_inv_rope_fp8_quant_torch(
            o, positions, cos_sin_cache,
            heads_per_group, quant_group_size, chunks_per_head,
            rope_start, half_rope, tma_aligned_scales, fp8_max,
            tma_aligned_T, num_tokens, n_groups, d, scale_inner,
        )

    def fused_q_kv_rmsnorm(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
    ):
        from .impl.deepseek_v4_attn import fused_q_kv_rmsnorm_torch

        return fused_q_kv_rmsnorm_torch(qr, kv, q_weight, kv_weight, eps)

    def indexer_k_quant_and_cache(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str,
    ):
        from .impl.deepseek_v4_attn import indexer_k_quant_and_cache_torch

        indexer_k_quant_and_cache_torch(
            k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
        )

    def cp_gather_indexer_k_quant_cache(
        self,
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        dst_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ):
        from .impl.deepseek_v4_attn import cp_gather_indexer_k_quant_cache_torch

        cp_gather_indexer_k_quant_cache_torch(
            kv_cache, dst_k, dst_scale, block_table, cu_seq_lens,
        )

    def top_k_per_row_prefill(
        self,
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ):
        from .impl.deepseek_v4_attn import top_k_per_row_prefill_torch

        top_k_per_row_prefill_torch(
            logits, cu_seqlen_ks, cu_seqlen_ke, raw_topk_indices,
            num_rows, stride0, stride1, topk_tokens,
        )

    def pack_seq_triton(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        pad_value: float | int = -float("inf"),
    ):
        from .impl.deepseek_v4_attn import pack_seq_triton_torch

        return pack_seq_triton_torch(x, lengths, pad_value)

    def top_k_per_row_decode(
        self,
        logits: torch.Tensor,
        next_n: int,
        seq_lens: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ):
        from .impl.deepseek_v4_attn import top_k_per_row_decode_torch

        top_k_per_row_decode_torch(
            logits, next_n, seq_lens, raw_topk_indices,
            num_rows, stride0, stride1, topk_tokens,
        )

    def unpack_seq_triton(
        self,
        packed_tensor: torch.Tensor,
        lengths: torch.Tensor,
    ):
        from .impl.deepseek_v4_attn import unpack_seq_triton_torch

        return unpack_seq_triton_torch(packed_tensor, lengths)

# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend implementation.

This backend provides operator implementations for NVIDIA CUDA GPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class CudaBackend(Backend):
    """
    CUDA backend for operator implementations.

    This backend uses CUDA libraries to provide high-performance
    operator implementations for NVIDIA GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def vendor(self) -> Optional[str]:
        return "nvidia"

    def is_available(self) -> bool:
        """
        Check if CUDA hardware and libraries are available.

        This method uses the platform's vendor information from FlagGems
        to determine if the device is a real NVIDIA GPU, decoupling from
        CUDA-alike devices (MACA, MUSA, etc.) which have their own vendor names.
        """
        if CudaBackend._available is None:
            try:
                # Check if CUDA device is available
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    CudaBackend._available = False
                    return False

                from vllm.platforms import current_platform

                if (
                    hasattr(current_platform, "device_name")
                    and current_platform.device_name == "nvidia"
                ):
                    CudaBackend._available = True
                else:
                    CudaBackend._available = False
            except Exception:
                CudaBackend._available = False
        return CudaBackend._available

    # ==================== Operator Implementations ====================

    def fused_marlin_moe(self, *args, **kwargs) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
        return fused_marlin_moe(*args, **kwargs)

    def router_gemm_bf16_fp32(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Run vLLM's CUDA BF16 router GEMM with an FP32 output."""
        from .impl.router_gemm import router_gemm_bf16_fp32_cuda

        return router_gemm_bf16_fp32_cuda(x, weight)

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Uses vLLM's native CUDA implementation.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_cuda

        return silu_and_mul_cuda(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Uses vLLM's native CUDA implementation.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_cuda

        return gelu_and_mul_cuda(obj, x)

    def silu_and_mul_with_clamp(self, x: torch.Tensor, swiglu_limit: float, swiglu_limit_tensor: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_with_clamp_cuda

        return silu_and_mul_with_clamp_cuda(x, swiglu_limit)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization using vLLM's CUDA implementation.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_cuda

        return rms_norm_cuda(obj, x, residual)

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
        Apply rotary position embedding using vLLM's CUDA implementation.

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
        from .impl.rotary import rotary_embedding_cuda

        return rotary_embedding_cuda(
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
        Get the attention backend class path for CUDA.

        Supports:
        - FLASH_ATTN (default)
        - TRITON_ATTN (when use_flaggems_op("triton_attn") is True)
        - FLASHMLA_SPARSE (when use_mla and use_sparse are both True)

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def fused_topk_bias(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        scoring_func: str,
        e_score_correction_bias,
        topk: int,
        renormalize: bool,
        indices_type=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor: float = 1.0,
    ):
        from .impl.fused_moe import fused_topk_bias_cuda

        return fused_topk_bias_cuda(
            hidden_states,
            gating_output,
            scoring_func,
            e_score_correction_bias,
            topk,
            renormalize,
            indices_type,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: Optional[torch.Tensor] = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ):
        from .impl.fused_moe import moe_align_block_size_cuda

        return moe_align_block_size_cuda(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    def moe_sum(self, inp, out):
        from .impl.fused_moe import moe_sum_cuda

        moe_sum_cuda(inp, out)

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from .impl.fused_moe import topk_softmax_cuda

        return topk_softmax_cuda(
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
        from .impl.fused_moe import invoke_fused_moe_triton_kernel_cuda

        invoke_fused_moe_triton_kernel_cuda(
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
        from .impl.fused_moe import grouped_topk_cuda

        return grouped_topk_cuda(
            scores, n_group, topk_group, topk,
            renormalize, routed_scaling_factor, bias, scoring_func,
        )

    def mhc_pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
    ):
        from .impl.mhc import mhc_pre_cuda

        return mhc_pre_cuda(
            residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, n_splits
        )

    def mhc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        from .impl.mhc import mhc_post_cuda

        return mhc_post_cuda(x, residual, post, comb)

    def hc_head_fused_kernel(
        self,
        hs_flat: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        out: torch.Tensor,
        hidden_size: int,
        rms_eps: float,
        hc_eps: float,
        hc_mult: int,
    ):
        from .impl.mhc import hc_head_fused_kernel_cuda

        hc_head_fused_kernel_cuda(
            hs_flat, fn, hc_scale, hc_base, out,
            hidden_size, rms_eps, hc_eps, hc_mult,
        )

    def deepseek_v4_mega_moe_experts(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor,
        layer_name: str,
        activation_clamp: float | None,
        fast_math: bool,
    ):
        torch.ops.vllm.deepseek_v4_mega_moe_experts(
            hidden_states, topk_weights, topk_ids, out,
            layer_name, activation_clamp, fast_math,
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
        from .impl.deepseek_v4_attn import deepseek_v4_fp8_einsum_cuda

        deepseek_v4_fp8_einsum_cuda(a, a_scale, b, b_scale, out, equation, recipe)

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
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_cuda,
        )

        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_cuda(
            q, kv, swa_kv_cache_2d, slot_mapping, positions, cos_sin_cache,
            eps, block_size,
        )

    def combine_topk_swa_indices(
        self,
        topk_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor,
        window_size: int,
        compress_ratio: int,
        topk: int,
        M: int,
        N: int,
    ):
        from .impl.deepseek_v4_attn import combine_topk_swa_indices_cuda

        return combine_topk_swa_indices_cuda(
            topk_indices, query_start_loc, seq_lens,
            gather_lens, window_size, compress_ratio,
            topk, M, N
        )

    def compute_global_topk_indices_and_lens(
        self,
        topk_indices: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        is_valid_token: torch.Tensor,
    ):
        from .impl.deepseek_v4_attn import compute_global_topk_indices_and_lens_cuda

        return compute_global_topk_indices_and_lens_cuda(
            topk_indices, token_to_req_indices, block_table, block_size, is_valid_token
        )

    def dequantize_and_gather_k_cache(
        self,
        out: torch.Tensor,
        # [num_blocks, block_size, head_bytes]
        k_cache: torch.Tensor,
        # [num_reqs]
        seq_lens: torch.Tensor,
        # [num_reqs]
        gather_lens: torch.Tensor | None,
        # [num_reqs, max_blocks_per_seq]
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
    ):
        from .impl.deepseek_v4_attn import dequantize_and_gather_k_cache_cuda

        dequantize_and_gather_k_cache_cuda(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
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
        from .impl.deepseek_v4_attn import fused_indexer_q_rope_quant_cuda

        return fused_indexer_q_rope_quant_cuda(
            positions, index_q, index_q_cos_sin_cache,
            index_weights, index_weights_softmax_scale,
            index_weights_head_scale, use_fp4,
        )

    def fused_inv_rope_fp8_quant(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int = 448,
        rope_dim: int = 64,
        quant_group_size: int = 128,
        tma_aligned_scales: bool = False
    ):
        from .impl.deepseek_v4_attn import fused_inv_rope_fp8_quant_cuda

        return fused_inv_rope_fp8_quant_cuda(
            o, positions, cos_sin_cache,
            n_groups, heads_per_group, nope_dim, rope_dim,
            quant_group_size, tma_aligned_scales,
        )

    def fused_q_kv_rmsnorm(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
    ):
        from .impl.deepseek_v4_attn import fused_q_kv_rmsnorm_cuda

        return fused_q_kv_rmsnorm_cuda(qr, kv, q_weight, kv_weight, eps)

    def indexer_k_quant_and_cache(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str,
    ):
        from .impl.deepseek_v4_attn import indexer_k_quant_and_cache_cuda

        indexer_k_quant_and_cache_cuda(
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
        from .impl.deepseek_v4_attn import cp_gather_indexer_k_quant_cache_cuda

        cp_gather_indexer_k_quant_cache_cuda(
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
        from .impl.deepseek_v4_attn import top_k_per_row_prefill_cuda

        top_k_per_row_prefill_cuda(
            logits, cu_seqlen_ks, cu_seqlen_ke, raw_topk_indices,
            num_rows, stride0, stride1, topk_tokens,
        )

    def pack_seq_triton(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        pad_value: float | int = -float("inf"),
    ):
        from .impl.deepseek_v4_attn import pack_seq_triton_cuda

        return pack_seq_triton_cuda(x, lengths, pad_value)

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
        from .impl.deepseek_v4_attn import top_k_per_row_decode_cuda

        top_k_per_row_decode_cuda(
            logits, next_n, seq_lens, raw_topk_indices,
            num_rows, stride0, stride1, topk_tokens,
        )

    def unpack_seq_triton(
        self,
        packed_tensor: torch.Tensor,
        lengths: torch.Tensor,
    ):
        from .impl.deepseek_v4_attn import unpack_seq_triton_cuda

        return unpack_seq_triton_cuda(packed_tensor, lengths)

    def flash_mla_with_kvcache(
        self,
        q,
        k_cache,
        block_table,
        head_dim_v,
        tile_scheduler_metadata,
        cache_seqlens,
        is_fp8_kvcache,
        indices,
        topk_length,
        softmax_scale,
        attn_sink,
        extra_k_cache,
        extra_indices_in_kvcache,
        extra_topk_length,
        out,
    ):
        from .impl.deepseek_v4_attn import flash_mla_with_kvcache_cuda

        return flash_mla_with_kvcache_cuda(
            q, k_cache, block_table, head_dim_v, tile_scheduler_metadata,
            cache_seqlens, is_fp8_kvcache, indices, topk_length, softmax_scale,
            attn_sink, extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            out,
        )
    
    def flash_mla_sparse_fwd(
        self,
        q,
        kv,
        indices,
        sm_scale,
        attn_sink,
        topk_length,
        out,
):
        from .impl.deepseek_v4_attn import flash_mla_sparse_fwd_cuda
        return flash_mla_sparse_fwd_cuda(
            q, kv, indices, sm_scale, attn_sink, topk_length, out,
        )

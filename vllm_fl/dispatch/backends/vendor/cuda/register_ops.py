# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend operator registrations.

This module registers all VENDOR (CUDA) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all CUDA (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .cuda import CudaBackend

    backend = CudaBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(op_name="fused_marlin_moe", impl_id="vendor.cuda",
               kind=BackendImplKind.VENDOR,
               fn=_bind_is_available(backend.fused_marlin_moe, is_avail),
               vendor="cuda", priority=BackendPriority.VENDOR),
        # MoE router GEMM (BF16 inputs, FP32 output)
        OpImpl(
            op_name="router_gemm_bf16_fp32",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.router_gemm_bf16_fp32, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gelu_and_mul, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="silu_and_mul_with_clamp",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul_with_clamp, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # MoE align
        OpImpl(
            op_name="fused_topk_bias",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_topk_bias, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # MoE align
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_align_block_size, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # MoE sum
        OpImpl(
            op_name="moe_sum",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.moe_sum, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # topk softmax
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # invoke fused moe triton kernel
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.invoke_fused_moe_triton_kernel, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # grouped topk
        OpImpl(
            op_name="grouped_topk",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.grouped_topk, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # mhc_pre
        OpImpl(
            op_name="mhc_pre",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.mhc_pre, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # mhc_post
        OpImpl(
            op_name="mhc_post",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.mhc_post, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # hc_head_fused_kernel
        OpImpl(
            op_name="hc_head_fused_kernel",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.hc_head_fused_kernel, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # deepseek_v4_mega_moe_experts
        OpImpl(
            op_name="deepseek_v4_mega_moe_experts",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.deepseek_v4_mega_moe_experts, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # deepseek_v4_fp8_einsum
        OpImpl(
            op_name="deepseek_v4_fp8_einsum",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.deepseek_v4_fp8_einsum, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
        OpImpl(
            op_name="fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # combine_topk_swa_indices
        OpImpl(
            op_name="combine_topk_swa_indices",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.combine_topk_swa_indices, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # compute_global_topk_indices_and_lens
        OpImpl(
            op_name="compute_global_topk_indices_and_lens",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.compute_global_topk_indices_and_lens, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # dequantize_and_gather_k_cache
        OpImpl(
            op_name="dequantize_and_gather_k_cache",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.dequantize_and_gather_k_cache, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # fused_indexer_q_rope_quant
        OpImpl(
            op_name="fused_indexer_q_rope_quant",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_indexer_q_rope_quant, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # fused_inv_rope_fp8_quant
        OpImpl(
            op_name="fused_inv_rope_fp8_quant",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_inv_rope_fp8_quant, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # fused_q_kv_rmsnorm
        OpImpl(
            op_name="fused_q_kv_rmsnorm",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_q_kv_rmsnorm, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # indexer_k_quant_and_cache
        OpImpl(
            op_name="indexer_k_quant_and_cache",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.indexer_k_quant_and_cache, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # cp_gather_indexer_k_quant_cache
        OpImpl(
            op_name="cp_gather_indexer_k_quant_cache",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.cp_gather_indexer_k_quant_cache, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # top_k_per_row_prefill
        OpImpl(
            op_name="top_k_per_row_prefill",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.top_k_per_row_prefill, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # pack_seq_triton
        OpImpl(
            op_name="pack_seq_triton",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.pack_seq_triton, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # top_k_per_row_decode
        OpImpl(
            op_name="top_k_per_row_decode",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.top_k_per_row_decode, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # unpack_seq_triton
        OpImpl(
            op_name="unpack_seq_triton",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.unpack_seq_triton, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # flash_mla_with_kvcache
        OpImpl(
            op_name="flash_mla_with_kvcache",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.flash_mla_with_kvcache, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
        # flash_mla_sparse_fwd
        OpImpl(
            op_name="flash_mla_sparse_fwd",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.flash_mla_sparse_fwd, is_avail),
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

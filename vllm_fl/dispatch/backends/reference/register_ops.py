# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
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
    Register all PyTorch (REFERENCE) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .reference import ReferenceBackend

    backend = ReferenceBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="fused_marlin_moe",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.fused_marlin_moe, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # MoE router GEMM (BF16 inputs, FP32 output)
        OpImpl(
            op_name="router_gemm_bf16_fp32",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.router_gemm_bf16_fp32, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.gelu_and_mul, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # MoE align
        OpImpl(
            op_name="moe_align_block_size",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.moe_align_block_size, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # MoE sum
        OpImpl(
            op_name="moe_sum",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.moe_sum, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # topk softmax
        OpImpl(
            op_name="topk_softmax",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # invoke fused moe triton kernel
        OpImpl(
            op_name="invoke_fused_moe_triton_kernel",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.invoke_fused_moe_triton_kernel, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # grouped topk
        OpImpl(
            op_name="grouped_topk",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.grouped_topk, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # deepseek_v4_fp8_einsum
        OpImpl(
            op_name="deepseek_v4_fp8_einsum",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.deepseek_v4_fp8_einsum, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
        OpImpl(
            op_name="fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # combine_topk_swa_indices
        OpImpl(
            op_name="combine_topk_swa_indices",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.combine_topk_swa_indices, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # compute_global_topk_indices_and_lens
        OpImpl(
            op_name="compute_global_topk_indices_and_lens",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.compute_global_topk_indices_and_lens, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # dequantize_and_gather_k_cache
        OpImpl(
            op_name="dequantize_and_gather_k_cache",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.dequantize_and_gather_k_cache, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # fused_indexer_q_rope_quant
        OpImpl(
            op_name="fused_indexer_q_rope_quant",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.fused_indexer_q_rope_quant, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # fused_inv_rope_fp8_quant
        OpImpl(
            op_name="fused_inv_rope_fp8_quant",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.fused_inv_rope_fp8_quant, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # fused_q_kv_rmsnorm
        OpImpl(
            op_name="fused_q_kv_rmsnorm",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.fused_q_kv_rmsnorm, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # indexer_k_quant_and_cache
        OpImpl(
            op_name="indexer_k_quant_and_cache",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.indexer_k_quant_and_cache, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # cp_gather_indexer_k_quant_cache
        OpImpl(
            op_name="cp_gather_indexer_k_quant_cache",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.cp_gather_indexer_k_quant_cache, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # top_k_per_row_prefill
        OpImpl(
            op_name="top_k_per_row_prefill",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.top_k_per_row_prefill, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # pack_seq_triton
        OpImpl(
            op_name="pack_seq_triton",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.pack_seq_triton, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # top_k_per_row_decode
        OpImpl(
            op_name="top_k_per_row_decode",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.top_k_per_row_decode, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
        # unpack_seq_triton
        OpImpl(
            op_name="unpack_seq_triton",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=_bind_is_available(backend.unpack_seq_triton, is_avail),
            vendor=None,
            priority=BackendPriority.REFERENCE,
        ),
    ]

    registry.register_many(impls)

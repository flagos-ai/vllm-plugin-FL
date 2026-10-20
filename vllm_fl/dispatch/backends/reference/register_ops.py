# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


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

    # ReferenceBackend implements only part of the dispatch surface on some
    # vLLM 0.24 builds. One absent optional MoE helper must not prevent all
    # available PyTorch fallbacks from registering.
    op_names = (
        "deepseek_v4_inv_rope_quant_int8",
        "deepseek_v4_inv_rope_quant_fp8",
        "deepseek_v4_int8_scaled_mm",
        "deepseek_v4_mhc_pre",
        "deepseek_v4_mhc_fused_post_pre",
        "deepseek_v4_mhc_post",
        "deepseek_v4_hc_head",
        "deepseek_v4_fused_q_kv_rmsnorm",
        "deepseek_v4_fused_q_kv_rmsnorm_int8",
        "deepseek_v4_qnorm_rope_kv_quant_insert",
        "deepseek_v4_qnorm_rope_kv_bf16_insert",
        "deepseek_v4_qnorm_rope_kv_fp8_insert",
        "deepseek_v4_compute_global_topk_indices_and_lens",
        "deepseek_v4_flash_mla_with_kvcache",
        "deepseek_v4_dequantize_and_gather_k_cache",
        "deepseek_v4_combine_topk_swa_indices",
        "deepseek_v4_flash_mla_sparse_fwd",
        "deepseek_v4_fused_indexer_q_rope_quant",
        "deepseek_v4_fused_indexer_q_rope_quant_int8",
        "deepseek_v4_compress_int8_indexer_k_cache",
        "deepseek_v4_int8_mqa_logits",
        "deepseek_v4_int8_paged_mqa_logits",
        "indexer_k_quant_and_cache",
        "cp_gather_indexer_k_quant_cache",
        "top_k_per_row_prefill",
        "top_k_per_row_decode",
        "pack_seq_triton",
        "unpack_seq_triton",
        "dynamic_per_token_quant_int8",
        "silu_and_mul",
        "silu_and_mul_with_clamp",
        "gelu_and_mul",
        "rms_norm",
        "rotary_embedding",
        "apply_rotary_emb",
        "attention_backend",
        "moe_align_block_size",
        "moe_sum",
        "topk_softmax",
        "invoke_fused_moe_triton_kernel",
        "grouped_topk",
        "fused_marlin_moe",
    )
    impls = []
    for op_name in op_names:
        fn = getattr(backend, op_name, None)
        if fn is None:
            continue
        impls.append(
            OpImpl(
                op_name=op_name,
                impl_id="reference.torch",
                kind=BackendImplKind.REFERENCE,
                fn=_bind_is_available(fn, is_avail),
                vendor=None,
                priority=BackendPriority.REFERENCE,
            )
        )

    registry.register_many(impls)

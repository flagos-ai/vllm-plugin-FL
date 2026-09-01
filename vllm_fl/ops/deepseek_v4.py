# SPDX-License-Identifier: Apache-2.0
"""OpManager frontends for DeepSeek-V4 model-specific compute."""

from __future__ import annotations

from vllm_fl.dispatch import resolve_op

_OPS = {
    # Resolve once while the model module is imported.  These frontends are
    # called from vLLM's full-graph compiled model; a lazy CachedOp lookup would
    # make Dynamo trace OpManager's RLock on the first profile run.  Import-time
    # resolution keeps backend selection in OpManager while exposing only the
    # selected callable to torch.compile and CUDA graph capture.
    name: resolve_op(f"deepseek_v4_{name}")
    for name in (
        "inv_rope_quant_fp8",
        "int8_scaled_mm",
        "mhc_pre",
        "mhc_fused_post_pre",
        "mhc_post",
        "hc_head",
        "fused_q_kv_rmsnorm",
        "qnorm_rope_kv_quant_insert",
        "qnorm_rope_kv_bf16_insert",
        "qnorm_rope_kv_fp8_insert",
        "compute_global_topk_indices_and_lens",
        "flash_mla_with_kvcache",
        "dequantize_and_gather_k_cache",
        "combine_topk_swa_indices",
        "flash_mla_sparse_fwd",
        "fused_indexer_q_rope_quant",
        "fused_indexer_q_rope_quant_int8",
        "compress_int8_indexer_k_cache",
        "int8_mqa_logits",
        "int8_paged_mqa_logits",
    )
}


def inv_rope_quant_fp8(*args, **kwargs):
    return _OPS["inv_rope_quant_fp8"](*args, **kwargs)


def int8_scaled_mm(*args, **kwargs):
    return _OPS["int8_scaled_mm"](*args, **kwargs)


def mhc_pre(*args, **kwargs):
    return _OPS["mhc_pre"](*args, **kwargs)


def mhc_fused_post_pre(*args, **kwargs):
    return _OPS["mhc_fused_post_pre"](*args, **kwargs)


def mhc_post(*args, **kwargs):
    return _OPS["mhc_post"](*args, **kwargs)


def hc_head(*args, **kwargs):
    return _OPS["hc_head"](*args, **kwargs)


def fused_q_kv_rmsnorm(*args, **kwargs):
    return _OPS["fused_q_kv_rmsnorm"](*args, **kwargs)


def qnorm_rope_kv_quant_insert(*args, **kwargs):
    return _OPS["qnorm_rope_kv_quant_insert"](*args, **kwargs)


def qnorm_rope_kv_bf16_insert(*args, **kwargs):
    return _OPS["qnorm_rope_kv_bf16_insert"](*args, **kwargs)


def qnorm_rope_kv_fp8_insert(*args, **kwargs):
    return _OPS["qnorm_rope_kv_fp8_insert"](*args, **kwargs)


def compute_global_topk_indices_and_lens(*args, **kwargs):
    return _OPS["compute_global_topk_indices_and_lens"](*args, **kwargs)


def flash_mla_with_kvcache(*args, **kwargs):
    return _OPS["flash_mla_with_kvcache"](*args, **kwargs)


def dequantize_and_gather_k_cache(*args, **kwargs):
    return _OPS["dequantize_and_gather_k_cache"](*args, **kwargs)


def combine_topk_swa_indices(*args, **kwargs):
    return _OPS["combine_topk_swa_indices"](*args, **kwargs)


def flash_mla_sparse_fwd(*args, **kwargs):
    return _OPS["flash_mla_sparse_fwd"](*args, **kwargs)


def fused_indexer_q_rope_quant(*args, **kwargs):
    return _OPS["fused_indexer_q_rope_quant"](*args, **kwargs)


def fused_indexer_q_rope_quant_int8(*args, **kwargs):
    return _OPS["fused_indexer_q_rope_quant_int8"](*args, **kwargs)


def compress_int8_indexer_k_cache(*args, **kwargs):
    return _OPS["compress_int8_indexer_k_cache"](*args, **kwargs)


def int8_mqa_logits(*args, **kwargs):
    return _OPS["int8_mqa_logits"](*args, **kwargs)


def int8_paged_mqa_logits(*args, **kwargs):
    return _OPS["int8_paged_mqa_logits"](*args, **kwargs)

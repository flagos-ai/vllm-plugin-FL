# Copyright (c) 2026 BAAI. All rights reserved.

"""Executable sparse-indexer cache/top-k bridges for NVIDIA EMPTY builds."""

import torch

from vllm_fl.nvidia_compat import has_native_vllm_kernels
from vllm_fl.ops.nvidia_dsv41_indexer import (
    _same_cuda,
    _validate_fp8_indexer_cache,
)

_libs: list[torch.library.Library] = []


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    from flag_gems import top_k_per_row_prefill as gems_topk

    _same_cuda(logits, row_starts, row_ends, indices)
    if logits.dtype != torch.float32 or logits.ndim != 2:
        raise ValueError("Prefill top-k requires FP32 logits [rows, columns]")
    if num_rows != logits.shape[0] or (stride0, stride1) != logits.stride():
        raise ValueError("Prefill top-k dimensions/strides do not match logits")
    if top_k not in (512, 1024, 2048):
        raise NotImplementedError("Prefill top-k supports k=512, 1024, or 2048")
    if any(
        t.dtype != torch.int32 or t.shape != (num_rows,) or not t.is_contiguous()
        for t in (row_starts, row_ends)
    ):
        raise ValueError("Prefill ranges must be contiguous int32 [rows]")
    if (
        indices.dtype != torch.int32
        or indices.shape != (num_rows, top_k)
        or not indices.is_contiguous()
    ):
        raise ValueError("Prefill output must be contiguous int32 [rows, k]")
    if num_rows:
        gems_topk(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )


def cp_gather_indexer_k_quant_cache(
    kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
):
    from flag_gems import cp_gather_indexer_k_quant_cache as gems_gather

    _same_cuda(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)
    _validate_fp8_indexer_cache(kv_cache)
    if (
        dst_k.dtype != torch.float8_e4m3fn
        or dst_k.ndim != 2
        or dst_k.shape[1] != 128
        or dst_k.stride(1) != 1
    ):
        raise NotImplementedError(
            "Indexer gather supports E4M3 FP8 keys of dimension 128"
        )
    if (
        dst_scale.dtype != torch.uint8
        or dst_scale.shape != (dst_k.shape[0], 4)
        or not dst_scale.is_contiguous()
    ):
        raise ValueError("Indexer scale destination must be uint8 [tokens, 4]")
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.stride(1) != 1
    ):
        raise ValueError("Indexer block table must be int32 with contiguous rows")
    if (
        cu_seq_lens.dtype != torch.int32
        or cu_seq_lens.shape != (block_table.shape[0] + 1,)
        or not cu_seq_lens.is_contiguous()
    ):
        raise ValueError(
            "Indexer cumulative lengths must be contiguous int32 [batch+1]"
        )
    if dst_k.shape[0]:
        gems_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)


def register_nvidia_dsv41_ops() -> bool:
    global _libs
    if has_native_vllm_kernels():
        return False
    if _libs:
        return True
    from vllm_fl.ops.nvidia_dsv41_cache import (
        kv_rope_insert,
        qnorm_rope_kv_rope_quant_insert,
    )

    definitions = (
        (
            "_C",
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            "(Tensor q_in, Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
            "Tensor position_ids, Tensor cos_sin_cache, int q_head_padded, "
            "float eps, int cache_block_size, bool apply_q_norm=True, "
            "bool kv_mxfp8=False, bool apply_q_rope=True, "
            "bool is_q_interleaved=False) -> Tensor",
            qnorm_rope_kv_rope_quant_insert,
        ),
        (
            "_C",
            "fused_deepseek_v4_kv_rope_insert",
            "(Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
            "Tensor position_ids, Tensor cos_sin_cache, int cache_block_size, "
            "Tensor? fp8_scale=None, bool kv_mxfp8=False) -> ()",
            kv_rope_insert,
        ),
        (
            "_C",
            "top_k_per_row_prefill",
            "(Tensor logits, Tensor rowStarts, Tensor rowEnds, Tensor! indices, int numRows, int stride0, int stride1, int topK) -> ()",
            top_k_per_row_prefill,
        ),
        (
            "_C_cache_ops",
            "cp_gather_indexer_k_quant_cache",
            "(Tensor kv_cache, Tensor! dst_k, Tensor! dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()",
            cp_gather_indexer_k_quant_cache,
        ),
    )
    for namespace, name, schema, implementation in definitions:
        lib = torch.library.Library(namespace, "FRAGMENT")
        if not hasattr(getattr(torch.ops, namespace), name):
            lib.define(name + schema)
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"{namespace}::{name}", "CUDA"
        ):
            lib.impl(name, implementation, "CUDA")
        _libs.append(lib)
    return True

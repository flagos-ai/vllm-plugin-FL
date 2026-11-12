# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch

from . import operators
from .operators import native_decode as original_decode, native_prefill as original
from vllm_fl.minimax_m3_config import load, thaw


def sparse_attn(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
):
    ok = (
        q.ndim == 3
        and q.shape[1:] == (4, 128)
        and (num_kv_heads == 1)
        and (q.dtype == torch.bfloat16)
        and (kv_cache.dtype == torch.bfloat16)
        and (kv_cache.ndim == 5)
        and (kv_cache.shape[1:] == (2, 128, 1, 128))
        and (topk_idx.shape == (1, q.shape[0], 16))
        and (topk_idx.dtype == torch.int32)
        and (output.shape == q.shape)
        and (output.dtype == q.dtype)
        and (block_table.dtype == torch.int32)
        and (cu_seqlens_q.dtype == torch.int32)
        and (seq_lens.dtype == torch.int32)
        and (prefix_lens.dtype == torch.int32)
        and (max_query_len > 0)
        and (q.shape[0] > 0)
    )
    if ok:
        return operators.prefill(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            num_kv_heads,
            sm_scale,
            output,
            dict(thaw(load().attention["base_prefill"]), selected_union=False),
        )
    return original(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
    )


def sparse_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
):
    ok = (
        q.ndim == 3
        and q.shape[1:] == (4, 128)
        and (num_kv_heads == 1)
        and (q.dtype == torch.bfloat16)
        and (kv_cache.dtype == torch.bfloat16)
        and (kv_cache.ndim == 5)
        and (kv_cache.shape[1:] == (2, 128, 1, 128))
        and (topk_idx.shape == (1, q.shape[0], 16))
        and (topk_idx.dtype == torch.int32)
        and (output.shape == q.shape)
        and (output.dtype == q.dtype)
        and (block_table.dtype == torch.int32)
        and (seq_lens.dtype == torch.int32)
        and (decode_query_len == 1)
        and (q.shape[0] == seq_lens.numel())
        and (0 < q.shape[0] <= 64)
    )
    if ok:
        cfg = thaw(load().attention["base_decode"])
        target = max(
            1, min(topk_idx.shape[-1], 256 // max(1, q.shape[0] * num_kv_heads))
        )
        chunks = cfg["split"] or (1 << (target.bit_length() - 1))
        launch = dict(
            num_topk_chunks=chunks,
            page_parts=1,
            ktile=cfg["ktile"],
            warps=cfg["warps"],
            stages=cfg["stages"],
        )
        return operators.decode(
            q,
            kv_cache,
            topk_idx,
            block_table,
            seq_lens,
            num_kv_heads,
            sm_scale,
            output,
            decode_query_len,
            launch,
        )
    return original_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
    )

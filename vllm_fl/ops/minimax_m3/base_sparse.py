# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch

from . import packed_prefill as kernel, subtile_decode as decode_kernel
from .native_sparse import (
    minimax_m3_sparse_attn as original,
    minimax_m3_sparse_attn_decode as original_decode,
)

COUNTS = {"prefill_packed": 0, "decode_subtile": 0, "fallback": 0}
PREFILL_SHAPES = {}


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
        key = f"{q.shape[0]},{max_query_len}"
        if key not in PREFILL_SHAPES:
            print(
                "M3_SPARSE_TUNING_HIT " + key + " QPACK8 K32 all_query_lengths",
                flush=True,
            )
        PREFILL_SHAPES[key] = PREFILL_SHAPES.get(key, 0) + 1
        COUNTS["prefill_packed"] += 1
        return kernel.minimax_m3_sparse_attn(
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
    COUNTS["fallback"] += 1
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
        COUNTS["decode_subtile"] += 1
        return decode_kernel.minimax_m3_sparse_attn_decode(
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
    COUNTS["fallback"] += 1
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

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch
import triton

from . import index_score as idx, long_decode as sd, long_prefill as sp


def index_score(q, kv, bt, cu, lens, prefix, maxq, maxseq, nkv, config):
    return idx.score(q, kv, bt, cu, lens, prefix, maxq, maxseq, nkv, **config)


def prefill(q, kv, ids, bt, cu, lens, prefix, maxq, nkv, scale, out, config):
    (qp, kt, w, s, u) = (
        config[k] for k in ["qpack", "ktile", "warps", "stages", "selected_union"]
    )
    return sp.packed_prefill_kernel[triton.cdiv(maxq, qp), cu.numel() - 1](
        q,
        kv,
        ids,
        bt,
        cu,
        lens,
        prefix,
        out,
        *q.stride(),
        *kv.stride(),
        *ids.stride(),
        bt.stride(0),
        *out.stride(),
        scale,
        qp,
        kt,
        u,
        num_warps=w,
        num_stages=s,
    )


def workspace_bytes(batch, parts):
    chunks = 16 * parts
    return dict(
        chunks=chunks,
        partial=chunks * batch * 4 * 128 * 2,
        lse=chunks * batch * 4 * 4,
        total=chunks * batch * 4 * (128 * 2 + 4),
    )


def decode(q, kv, ids, bt, lens, nkv, scale, out, decode_query_len, config):
    (total, heads, dim) = q.shape
    parts = config["page_parts"]
    chunks = 16 * parts
    op = torch.empty((chunks, total, heads, dim), dtype=q.dtype, device=q.device)
    lp = torch.empty((chunks, total, heads), dtype=torch.float32, device=q.device)
    a = sd._gqa_sparse_decode_kernel[total * chunks, nkv](
        q,
        kv,
        ids,
        op,
        lp,
        bt,
        lens,
        total,
        heads // nkv,
        dim,
        ids.shape[-1],
        scale,
        decode_query_len,
        *q.stride(),
        *kv.stride(),
        *ids.stride(),
        *op.stride(),
        *lp.stride(),
        bt.stride(0),
        BLOCK_SIZE_K=128,
        NUM_TOPK_CHUNKS=chunks,
        INNER_K=config["ktile"],
        PAGE_PARTS=parts,
        USE_FP8=False,
        USE_PDL=False,
        num_warps=config["warps"],
        num_stages=config["stages"],
    )
    b = sd._merge_topk_attn_out_kernel[total, heads](
        op,
        lp,
        out,
        dim,
        *op.stride(),
        *lp.stride(),
        *out.stride(),
        NUM_TOPK_CHUNKS=chunks,
        USE_PDL=False,
    )
    return (a, b)

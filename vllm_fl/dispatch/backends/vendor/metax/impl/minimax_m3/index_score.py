# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Experimental M3 H1/D128 block scores. TopK selection is deliberately separate.

Vector scoring removes the Q1->Q64 matrix padding. Dot variants split independent
KV pages without splitting the head-dimension reduction. Nothing installs itself.
"""

import torch
import triton
import triton.language as tl

LAST_KERNELS = []


@triton.jit
def index_vector_kernel(
    Q,
    IK,
    BT,
    CU,
    LENS,
    PREFIX,
    S,
    QN: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KN: tl.constexpr,
    KD: tl.constexpr,
    BS: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    CHUNKS: tl.constexpr,
    KTILE: tl.constexpr,
    D: tl.constexpr = 128,
):
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    start = tl.load(CU + req)
    qlen = tl.load(CU + req + 1) - start
    if qlen == 0:
        return
    seq = tl.load(LENS + req)
    prefix = tl.load(PREFIX + req)
    visible = tl.minimum(seq, prefix + 1)
    pages = tl.cdiv(tl.maximum(visible, 0), 128)
    per = tl.cdiv(pages, CHUNKS)
    lo = chunk * per
    hi = tl.minimum(lo + per, pages)
    if lo >= hi:
        return
    dd = tl.arange(0, D)
    kk = tl.arange(0, KTILE)
    q = tl.load(Q + start * QN + dd * QD).to(tl.float32)
    for blk in tl.range(lo, hi):
        physical = tl.load(BT + req * BS + blk).to(tl.int64)
        best = tl.full((), float("-inf"), tl.float32)
        for sub in range(0, 128, KTILE):
            k = tl.load(
                IK + physical * KB + (sub + kk[:, None]) * KN + dd[None, :] * KD
            ).to(tl.float32)
            dots = tl.sum(k * q[None, :], 1)
            dots = tl.where(blk * 128 + sub + kk < visible, dots, float("-inf"))
            best = tl.maximum(best, tl.max(dots, 0))
        tl.store(S + start * SN + blk * SK, best)


@triton.jit
def index_dot_kernel(
    Q,
    IK,
    BT,
    CU,
    LENS,
    PREFIX,
    S,
    QN: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KN: tl.constexpr,
    KD: tl.constexpr,
    BS: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    BQ: tl.constexpr,
    KTILE: tl.constexpr,
    CHUNKS: tl.constexpr,
    D: tl.constexpr = 128,
):
    qt = tl.program_id(0)
    req = tl.program_id(1)
    chunk = tl.program_id(2)
    start = tl.load(CU + req)
    qlen = tl.load(CU + req + 1) - start
    if qt * BQ >= qlen:
        return
    seq = tl.load(LENS + req)
    prefix = tl.load(PREFIX + req)
    qr = qt * BQ + tl.arange(0, BQ)
    dd = tl.arange(0, D)
    kk = tl.arange(0, KTILE)
    visible = tl.minimum(seq, prefix + tl.minimum((qt + 1) * BQ, qlen))
    pages = tl.cdiv(tl.maximum(visible, 0), 128)
    per = tl.cdiv(pages, CHUNKS)
    lo = chunk * per
    hi = tl.minimum(lo + per, pages)
    if lo >= hi:
        return
    q = tl.load(
        Q + (start + qr[:, None]) * QN + dd[None, :] * QD,
        mask=qr[:, None] < qlen,
        other=0,
    )
    for blk in tl.range(lo, hi):
        physical = tl.load(BT + req * BS + blk).to(tl.int64)
        best = tl.full((BQ,), float("-inf"), tl.float32)
        for sub in range(0, 128, KTILE):
            k = tl.load(
                IK + physical * KB + (sub + kk[None, :]) * KN + dd[:, None] * KD
            )
            dots = tl.dot(q, k)
            pos = blk * 128 + sub + kk
            # Like the reference, historical pages need no per-element causal mask.
            if prefix + qt * BQ < blk * 128 + sub + KTILE:
                dots = tl.where(
                    pos[None, :] <= prefix + qr[:, None], dots, float("-inf")
                )
            best = tl.maximum(best, tl.max(dots, 1))
        tl.store(S + (start + qr) * SN + blk * SK, best, mask=qr < qlen)


def chunk_count(batch, qtiles, target=512, cap=64):
    value = max(1, min(cap, target // max(1, batch * qtiles)))
    return 1 << (value.bit_length() - 1)


@torch.no_grad()
def score(
    idx_q,
    index_kv_cache,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    num_kv_heads,
    *,
    mode="vector",
    bq=16,
    ktile=128,
    chunks=None,
    warps=4,
    stages=1,
    collect=False,
):
    assert idx_q.shape[1:] == (1, 128) and num_kv_heads == 1
    assert idx_q.dtype == index_kv_cache.dtype == torch.bfloat16
    assert index_kv_cache.shape[1:] == (128, 128)
    assert ktile in (32, 64, 128)
    batch = cu_seqlens_q.numel() - 1
    stride = triton.cdiv(triton.cdiv(max_seq_len, 128), 16) * 16
    out = torch.empty(
        (1, idx_q.shape[0], stride), dtype=torch.float32, device=idx_q.device
    )
    args = (
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        out,
        idx_q.stride(0),
        idx_q.stride(2),
        *index_kv_cache.stride(),
        block_table.stride(0),
        out.stride(1),
        out.stride(2),
    )
    if mode == "vector":
        assert max_query_len == 1
        splits = chunks or chunk_count(batch, 1)
        k = index_vector_kernel[(batch, splits)](
            *args, CHUNKS=splits, KTILE=ktile, num_warps=warps, num_stages=stages
        )
    elif mode == "dot":
        assert bq in (16, 32, 64)
        splits = chunks or chunk_count(batch, triton.cdiv(max_query_len, bq), cap=16)
        k = index_dot_kernel[(triton.cdiv(max_query_len, bq), batch, splits)](
            *args, BQ=bq, KTILE=ktile, CHUNKS=splits, num_warps=warps, num_stages=stages
        )
    else:
        raise ValueError(mode)
    if collect:
        LAST_KERNELS.append(k)
    return out

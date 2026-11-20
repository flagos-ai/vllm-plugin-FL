# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Length-independent local-query packed BF16 GQA; no cross-rank exchange.
Kernel handles query-specific top-k membership and arbitrary page tables. Caller
guard restricts H4/HKV1/D128, block128, K16, BF16. GPU lengths and membership remain dynamic; wholly unselected pages skip GEMMs. Top-k order is
replayed ascending, retaining exactly the selected set and causal mask; therefore
numerical rather than bitwise equivalence is required."""

import torch
import triton
import triton.language as tl

LAST_KERNELS = []
COLLECT_METADATA = True
WARPS = 2
STAGES = 1
QPACK = 8
KTILE = 32
TRAVERSE_SELECTED = False


@triton.jit
def packed_prefill_kernel(
    Q,
    KV,
    IDX,
    BT,
    CU,
    SEQ,
    PRE,
    O,
    SQ: tl.constexpr,
    SH: tl.constexpr,
    SD: tl.constexpr,
    KB: tl.constexpr,
    KK: tl.constexpr,
    KP: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    IT: tl.constexpr,
    IN: tl.constexpr,
    IK: tl.constexpr,
    BS: tl.constexpr,
    ON: tl.constexpr,
    OH: tl.constexpr,
    OD: tl.constexpr,
    SCALE: tl.constexpr,
    QPACK: tl.constexpr,
    KTILE: tl.constexpr,
    TRAVERSE_SELECTED: tl.constexpr,
    H: tl.constexpr = 4,
    D: tl.constexpr = 128,
):
    qt = tl.program_id(0)
    req = tl.program_id(1)
    start = tl.load(CU + req)
    qlen = tl.load(CU + req + 1) - start
    seq = tl.load(SEQ + req)
    prefix = tl.load(PRE + req)
    if qt * QPACK >= qlen:
        return
    rr = tl.arange(0, QPACK * H)
    dd = tl.arange(0, D)
    nn = tl.arange(0, KTILE)
    qr = qt * QPACK + rr // H
    hr = rr % H
    q = tl.load(
        Q + (start + qr[:, None]) * SQ + hr[:, None] * SH + dd[None, :] * SD,
        mask=(qr[:, None] < qlen),
        other=0,
    )
    top = tl.load(
        IDX + (start + qr[:, None]) * IN + tl.arange(0, 16)[None, :] * IK,
        mask=qr[:, None] < qlen,
        other=-1,
    )
    mi = tl.full((QPACK * H,), float("-inf"), tl.float32)
    lsei = tl.full((QPACK * H,), float("-inf"), tl.float32)
    acc = tl.zeros((QPACK * H, D), tl.float32)
    hi = tl.minimum(seq, prefix + tl.minimum((qt + 1) * QPACK, qlen))
    if TRAVERSE_SELECTED:
        sentinel = 2147483647
        blk = tl.min(tl.min(tl.where(top >= 0, top, sentinel), 1), 0)
        while blk < tl.cdiv(hi, 128):
            chosen = tl.sum((top == blk).to(tl.int32), 1) > 0
            if tl.sum(chosen.to(tl.int32), 0) > 0:
                page = tl.load(BT + req * BS + blk).to(tl.int64)
                for sub in range(0, 128, KTILE):
                    pos = blk * 128 + sub + nn
                    k = tl.load(
                        KV + page * KB + (sub + nn[None, :]) * KP + dd[:, None] * KD,
                        mask=pos[None, :] < seq,
                        other=0,
                    )
                    valid = (
                        (pos[None, :] < seq)
                        & (pos[None, :] <= prefix + qr[:, None])
                        & chosen[:, None]
                        & (qr[:, None] < qlen)
                    )
                    score = tl.dot(q, k) * (SCALE * 1.4426950409)
                    score = tl.where(valid, score, float("-inf"))
                    mj = tl.maximum(mi, tl.max(score, 1))
                    safe = tl.where(mj > float("-inf"), mj, 0.0)
                    p = tl.exp2(score - safe[:, None])
                    alpha = tl.exp2(mi - safe)
                    v = tl.load(
                        KV
                        + page * KB
                        + KK
                        + (sub + nn[:, None]) * KP
                        + dd[None, :] * KD,
                        mask=pos[:, None] < seq,
                        other=0,
                    )
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    l = tl.exp2(lsei - safe) + tl.sum(p, 1)
                    lsei = tl.where(l > 0, safe + tl.log2(l), float("-inf"))
                    mi = mj
            blk = tl.min(tl.min(tl.where(top > blk, top, sentinel), 1), 0)
    else:
        for blk in range(tl.cdiv(hi, 128)):
            chosen = tl.sum((top == blk).to(tl.int32), 1) > 0
            if tl.sum(chosen.to(tl.int32), 0) > 0:
                page = tl.load(BT + req * BS + blk).to(tl.int64)
                for sub in range(0, 128, KTILE):
                    pos = blk * 128 + sub + nn
                    k = tl.load(
                        KV + page * KB + (sub + nn[None, :]) * KP + dd[:, None] * KD,
                        mask=pos[None, :] < seq,
                        other=0,
                    )
                    valid = (
                        (pos[None, :] < seq)
                        & (pos[None, :] <= prefix + qr[:, None])
                        & chosen[:, None]
                        & (qr[:, None] < qlen)
                    )
                    score = tl.dot(q, k) * (SCALE * 1.4426950409)
                    score = tl.where(valid, score, float("-inf"))
                    mj = tl.maximum(mi, tl.max(score, 1))
                    safe = tl.where(mj > float("-inf"), mj, 0.0)
                    p = tl.exp2(score - safe[:, None])
                    alpha = tl.exp2(mi - safe)
                    v = tl.load(
                        KV
                        + page * KB
                        + KK
                        + (sub + nn[:, None]) * KP
                        + dd[None, :] * KD,
                        mask=pos[:, None] < seq,
                        other=0,
                    )
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                    l = tl.exp2(lsei - safe) + tl.sum(p, 1)
                    lsei = tl.where(l > 0, safe + tl.log2(l), float("-inf"))
                    mi = mj
    factor = tl.where(lsei > float("-inf"), tl.exp2(mi - lsei), 0.0)
    acc *= factor[:, None]
    tl.store(
        O + (start + qr[:, None]) * ON + hr[:, None] * OH + dd[None, :] * OD,
        acc,
        mask=qr[:, None] < qlen,
    )


@torch.no_grad()
def minimax_m3_sparse_attn(
    q, kv, idx, bt, cu, lens, prefix, max_query_len, num_kv_heads, scale, out
):
    assert q.shape[1:] == (4, 128) and num_kv_heads == 1 and idx.shape[-1] == 16
    assert kv.dtype == q.dtype == torch.bfloat16
    k = packed_prefill_kernel[(triton.cdiv(max_query_len, QPACK), cu.numel() - 1)](
        q,
        kv,
        idx,
        bt,
        cu,
        lens,
        prefix,
        out,
        *q.stride(),
        *kv.stride(),
        *idx.stride(),
        bt.stride(0),
        *out.stride(),
        scale,
        QPACK,
        KTILE,
        TRAVERSE_SELECTED,
        num_warps=WARPS,
        num_stages=STAGES,
    )
    if COLLECT_METADATA:
        LAST_KERNELS.append(k)

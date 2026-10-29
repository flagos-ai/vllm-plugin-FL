# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Portable FGV/Marlin-style MXFP4 layout and K3 expert chain.

Derived from FlagGems-vllm snapshot3e9e47c grouped scheduling/interleave.
PTX is replaced by validated exact integer BF16 decode. No BF16 weight expansion.
G1 uses a fixed128-K storage tile; large-M computation reads64-K halves.
"""

import torch
import triton
import triton.language as tl
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.decode_bits import decode_bits


@triton.jit
def _pack_words(
    W,
    O,
    WORDS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < WORDS
    n = idx % N
    q = idx // N
    e = q // (K // 8)
    pk = q % (K // 8)
    kp = TK // 8
    block = pk // kp
    lane = pk % kp
    word = tl.full((BLOCK,), 0, tl.uint32)
    for j in tl.static_range(8):
        k = block * TK + j * kp + lane
        byte = tl.load(W + (e * N + n) * (K // 2) + k // 2, valid, 0).to(tl.uint32)
        word = word | (byte >> k % 2 * 4 & 15) << (j // 2 + j % 2 * 4) * 4
    tl.store(O + idx, word, valid)


def pack_matrix(w, s, tile_k):
    (e, n, kh) = w.shape
    k = kh * 2
    assert k % tile_k == 0 and tile_k in (32, 64, 128)
    out = torch.empty((e, k // 8, n), device=w.device, dtype=torch.int32)
    _pack_words[triton.cdiv(out.numel(), 256),](
        w, out, out.numel(), n, k, tile_k, 256, num_warps=4
    )
    return (out, s.transpose(1, 2).contiguous())


@triton.jit
def _decode_words(b, s0, s1, s2, s3, BK: tl.constexpr):
    if BK == 32:
        a0 = s0
        a1 = s0
        a2 = s0
        a3 = s0
        a4 = s0
        a5 = s0
        a6 = s0
        a7 = s0
    elif BK == 64:
        a0 = s0
        a1 = s0
        a2 = s0
        a3 = s0
        a4 = s1
        a5 = s1
        a6 = s1
        a7 = s1
    else:
        a0 = s0
        a1 = s0
        a2 = s1
        a3 = s1
        a4 = s2
        a5 = s2
        a6 = s3
        a7 = s3
    return (
        decode_bits(b >> 0 & 15, a0),
        decode_bits(b >> 16 & 15, a1),
        decode_bits(b >> 4 & 15, a2),
        decode_bits(b >> 20 & 15, a3),
        decode_bits(b >> 8 & 15, a4),
        decode_bits(b >> 24 & 15, a5),
        decode_bits(b >> 12 & 15, a6),
        decode_bits(b >> 28 & 15, a7),
    )


@triton.jit
def _concat_k(bs, BN: tl.constexpr, BK: tl.constexpr):
    j0 = tl.join(bs[0], bs[1])
    j1 = tl.join(bs[2], bs[3])
    j2 = tl.join(bs[4], bs[5])
    j3 = tl.join(bs[6], bs[7])
    p0 = tl.join(j0, j1)
    p1 = tl.join(j2, j3)
    q = tl.join(p0, p1)
    return tl.reshape(tl.trans(q, 4, 3, 2, 0, 1), (BK, BN))


@triton.jit
def _marlin_tail_gemv(
    X,
    W,
    S,
    Y,
    IDS,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    route = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    expert = tl.load(IDS + route).to(tl.int64)
    kp = TK // 8
    acc = tl.zeros((BN,), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        valid = k < K
        pk = k // TK * kp + k % kp
        j = k % TK // kp
        shift = (j // 2 + j % 2 * 4) * 4
        x = tl.load(X + route // 16 * K + k, valid, 0).to(tl.float32)
        word = tl.load(
            W + (expert * (K // 8) + pk[None, :]) * N + cols[:, None],
            (cols[:, None] < N) & valid[None, :],
            0,
        ).to(tl.uint32)
        scale = tl.load(
            S + (expert * (K // 32) + k[None, :] // 32) * N + cols[:, None],
            (cols[:, None] < N) & valid[None, :],
            127,
        ).to(tl.int32)
        weight = decode_bits(word >> shift[None, :] & 15, scale).to(tl.float32)
        acc += tl.sum(weight * x[None, :], 1)
    tl.store(Y + route * N + cols, acc.to(tl.bfloat16), cols < N)


@triton.jit
def _decode_half_words(b, s0, s1, half):
    b = b >> half * 8
    return (
        decode_bits(b >> 0 & 15, s0),
        decode_bits(b >> 16 & 15, s0),
        decode_bits(b >> 4 & 15, s1),
        decode_bits(b >> 20 & 15, s1),
    )


@triton.jit
def _concat_half(bs, BN: tl.constexpr):
    a = tl.join(bs[0], bs[1])
    b = tl.join(bs[2], bs[3])
    q = tl.join(a, b)
    return tl.reshape(tl.trans(q, 3, 2, 0, 1), (64, BN))

"""Shared packed-expert decode scheduling kernels used by TP8.
Uses exact arithmetic FP4 decoding with a startup-checked normal-scale specialization,
and DeepGEMM-inspired GPU-count-driven SM-strided G2 work scheduling.
Baseline MXFP4/E8M0 storage, BF16 boundaries and FP32 SiTU/route remain unchanged.
"""

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.expert_gemm import _decode_words_math

# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.marlin_mxfp4 import _decode_words, _concat_k


@triton.jit
def _paired_scheduled_mm(
    X,
    W,
    S,
    Y,
    Sorted,
    Experts,
    Total,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    IT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    PROGRAMS: tl.constexpr,
    EXPANDED: tl.constexpr = False,
    MAX_BLOCKS: tl.constexpr = 0,
    DECODE_MODE: tl.constexpr = 0,
):
    # Actual task count remains a GPU scalar. Captured launch shape is static.
    nm = tl.cdiv(tl.load(Total), BM)
    if MAX_BLOCKS > 0:
        if nm > MAX_BLOCKS:
            return
    nn = tl.cdiv(N, BN)
    for task in range(tl.program_id(0), nm * nn, PROGRAMS):
        group = task // (GM * nn)
        first = group * GM
        size = tl.minimum(nm - first, GM)
        rm = first + (task % (GM * nn)) % size
        cn = (task % (GM * nn)) // size
        rr = rm * BM + tl.arange(0, BM)
        route = tl.load(Sorted + rr).to(tl.int64)
        valid = route < R
        expert = tl.load(Experts + rm).to(tl.int64)
        if (expert >= 0) & (expert < 896):
            col = (cn * BN + tl.arange(0, BN).to(tl.int64)) % N
            kp = tl.arange(0, BK // 8)
            kk = tl.arange(0, BK)
            bp = W + (expert * (K // 8) + kp[:, None]) * N + col[None, :]
            sp = S + expert * (K // 32) * N + col
            acc = tl.zeros((BM, BN), tl.float32)
            for block in range(K // BK):
                packed = tl.load(bp).to(tl.uint32)
                sg = block * (BK // 32)
                s0 = tl.load(sp + sg * N).to(tl.int32)
                s1 = tl.load(sp + (sg + 1) * N, BK >= 64, 127).to(tl.int32)
                s2 = tl.load(sp + (sg + 2) * N, BK >= 128, 127).to(tl.int32)
                s3 = tl.load(sp + (sg + 3) * N, BK >= 128, 127).to(tl.int32)
                if DECODE_MODE > 0:
                    parts = _decode_words_math(
                        packed,
                        s0[None, :],
                        s1[None, :],
                        s2[None, :],
                        s3[None, :],
                        BK,
                        DECODE_MODE == 2,
                    )
                else:
                    parts = _decode_words(
                        packed, s0[None, :], s1[None, :], s2[None, :], s3[None, :], BK
                    )
                b = _concat_k(parts, BN, BK)
                rows = rr if EXPANDED else route // IT
                a = tl.load(
                    X + rows[:, None] * K + block * BK + kk[None, :], valid[:, None], 0
                )
                acc = tl.dot(a, b, acc)
                bp += BK // 8 * N
            cols = cn * BN + tl.arange(0, BN)
            tl.store(
                Y + route[:, None] * N + cols[None, :],
                acc.to(tl.bfloat16),
                valid[:, None] & (cols[None, :] < N),
            )


@triton.jit
def _route_gemv_g2(
    X,
    W,
    S,
    Y,
    Ids,
    R: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    PROGRAMS: tl.constexpr,
    NORMAL: tl.constexpr,
    K: tl.constexpr = 96,
):
    nn = tl.cdiv(N, BN)
    kp = tl.arange(0, 4)
    kk = tl.arange(0, 32)
    for task in range(tl.program_id(0), R * nn, PROGRAMS):
        route = task // nn
        cols = task % nn * BN + tl.arange(0, BN)
        expert = tl.load(Ids + route).to(tl.int64)
        acc = tl.zeros((BN,), tl.float32)
        for block in range(K // 32):
            word = tl.load(
                W + (expert * (K // 8) + block * 4 + kp[:, None]) * N + cols[None, :],
                cols[None, :] < N,
                0,
            ).to(tl.uint32)
            scale = tl.load(
                S + (expert * (K // 32) + block) * N + cols, cols < N, 127
            ).to(tl.int32)
            parts = _decode_words_math(
                word,
                scale[None, :],
                scale[None, :],
                scale[None, :],
                scale[None, :],
                32,
                NORMAL,
            )
            b = _concat_k(parts, BN, 32).to(tl.float32)
            x = tl.load(X + route * K + block * 32 + kk).to(tl.float32)
            acc += tl.sum(x[:, None] * b, 0)
        tl.store(Y + route * N + cols, acc.to(tl.bfloat16), cols < N)

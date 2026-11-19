from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.marlin_mxfp4 import _decode_half_words

"""Shared canonical expert GEMM and MXFP4 decode primitives."""

import torch
import triton
import triton.language as tl
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.decode_bits import decode_bits
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.marlin_mxfp4 import _concat_k, _concat_half


@triton.jit
def _decode_math(code, scale, NORMAL: tl.constexpr = False):
    code = code.to(tl.int32)
    scale = scale.to(tl.int32)
    magnitude = code & 7
    biased = scale + (magnitude >> 1) - 1
    mantissa = tl.where(magnitude > 1, (magnitude & 1) << 6, 0)
    value = (biased << 7) | mantissa
    if not NORMAL:
        subnormal = (128 | mantissa) >> tl.maximum(1 - biased, 0)
        value = tl.where(biased <= 0, subnormal, value)
        value = tl.where(biased >= 255, 0x7F80, value)
    value = tl.where(magnitude == 0, 0, value) | ((code & 8) << 12)
    return value.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _decode_normal_pair(word, scale, SHIFT: tl.constexpr):
    pair = (word.to(tl.uint32) >> SHIFT) & 0x000F000F
    mag = pair & 0x00070007
    biased = (scale.to(tl.uint32) - 1) * 0x00010001 + ((mag >> 1) & 0x00030003)
    exponent = biased << 7
    mantissa = ((mag + 0x00060006) & (mag << 3) & 0x00080008) << 3
    nonzero = ((mag + 0x00070007) >> 3) & 0x00010001
    bits = ((exponent | mantissa) & (nonzero * 0x7FFF)) | ((pair & 0x00080008) << 12)
    return (
        (bits & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        (bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True),
    )


@triton.jit
def _decode_words_math(
    b, s0, s1, s2, s3, BK: tl.constexpr, NORMAL: tl.constexpr = False
):
    if NORMAL:
        if BK == 32:
            a0 = s0
            a1 = s0
            a2 = s0
            a3 = s0
        elif BK == 64:
            a0 = s0
            a1 = s0
            a2 = s1
            a3 = s1
        else:
            a0 = s0
            a1 = s1
            a2 = s2
            a3 = s3
        (p0, p1) = _decode_normal_pair(b, a0, 0)
        (p2, p3) = _decode_normal_pair(b, a1, 4)
        (p4, p5) = _decode_normal_pair(b, a2, 8)
        (p6, p7) = _decode_normal_pair(b, a3, 12)
        return (p0, p1, p2, p3, p4, p5, p6, p7)
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
        _decode_math(b >> 0 & 15, a0, NORMAL),
        _decode_math(b >> 16 & 15, a1, NORMAL),
        _decode_math(b >> 4 & 15, a2, NORMAL),
        _decode_math(b >> 20 & 15, a3, NORMAL),
        _decode_math(b >> 8 & 15, a4, NORMAL),
        _decode_math(b >> 24 & 15, a5, NORMAL),
        _decode_math(b >> 12 & 15, a6, NORMAL),
        _decode_math(b >> 28 & 15, a7, NORMAL),
    )


@triton.jit
def _decode_half_words_math(word, s0, s1, half, NORMAL: tl.constexpr):
    if NORMAL:
        shifted = word.to(tl.uint32) >> (half * 8)
        a0, a1 = _decode_normal_pair(shifted, s0, 0)
        a2, a3 = _decode_normal_pair(shifted, s1, 4)
        return a0, a1, a2, a3
    return _decode_half_words(word, s0, s1, half)


@triton.jit
def _paired_arithmetic_mm(
    X,
    W,
    S,
    Y,
    Sorted,
    Experts,
    Total,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    R: tl.constexpr,
    IT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    PACK_K: tl.constexpr = 0,
    MODE: tl.constexpr = 1,
    EXPERT_LIMIT: tl.constexpr = 896,
    N_START: tl.constexpr = 0,
    OUTPUT_N: tl.constexpr = 0,
):
    ON: tl.constexpr = N if OUTPUT_N == 0 else OUTPUT_N
    tl.static_assert(0 <= N_START and N_START + ON <= N)
    PK: tl.constexpr = BK if PACK_K == 0 else PACK_K
    tl.static_assert(PK == BK or (PK == 128 and BK == 64))
    pid = tl.program_id(0)
    nm = tl.cdiv(EM, BM)
    nn = tl.cdiv(ON, BN)
    ng = GM * nn
    group = pid // ng
    first = group * GM
    size = tl.minimum(nm - first, GM)
    rm = first + pid % ng % size
    cn = pid % ng // size
    if rm * BM >= tl.load(Total):
        return
    routes = tl.load(Sorted + rm * BM + tl.arange(0, BM)).to(tl.int64)
    valid = routes < R
    expert = tl.load(Experts + rm).to(tl.int64)
    if (expert < 0) | (expert >= EXPERT_LIMIT):
        return
    cols = (N_START + cn * BN + tl.arange(0, BN).to(tl.int64)) % N
    kp = tl.arange(0, PK // 8)
    kk = tl.arange(0, BK)
    bp = W + (expert * (K // 8) + kp[:, None]) * N + cols[None, :]
    sp = S + expert * (K // 32) * N + cols
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(K // BK):
        packed = tl.load(bp).to(tl.uint32)
        sg = block * (BK // 32)
        s0 = tl.load(sp + sg * N).to(tl.int32)
        s1 = tl.load(sp + (sg + 1) * N, BK >= 64, 127).to(tl.int32)
        s2 = tl.load(sp + (sg + 2) * N, BK >= 128, 127).to(tl.int32)
        s3 = tl.load(sp + (sg + 3) * N, BK >= 128, 127).to(tl.int32)
        if PK == BK:
            parts = _decode_words_math(
                packed,
                s0[None, :],
                s1[None, :],
                s2[None, :],
                s3[None, :],
                BK,
                MODE == 2,
            )
            b = _concat_k(parts, BN, BK)
        else:
            parts = _decode_half_words_math(
                packed, s0[None, :], s1[None, :], block % 2, MODE == 2
            )
            b = _concat_half(parts, BN)
        a = tl.load(
            X + routes[:, None] // IT * K + block * BK + kk[None, :], valid[:, None], 0
        )
        acc = tl.dot(a, b, acc)
        if PK == BK:
            bp += BK // 8 * N
        else:
            bp += tl.where(block % 2 == 1, PK // 8 * N, 0)
    outcols = cn * BN + tl.arange(0, BN)
    tl.store(
        Y + routes[:, None] * ON + outcols[None, :],
        acc.to(tl.bfloat16),
        valid[:, None] & (outcols[None, :] < ON),
    )


def meta(obj):
    m = obj.metadata
    return {
        "n_regs": getattr(obj, "n_regs", None),
        "n_spills": getattr(obj, "n_spills", None),
        "shared": getattr(m, "shared", None),
        "num_warps": getattr(m, "num_warps", None),
        "num_stages": getattr(m, "num_stages", None),
    }

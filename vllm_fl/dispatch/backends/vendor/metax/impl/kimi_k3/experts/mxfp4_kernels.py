"""BF16 reference/adaptation kernels that read original packed MXFP4 weights."""

import triton
import triton.language as tl


@triton.jit
def _e2m1(code):
    magnitude = code & 7
    exponent = magnitude >> 1
    mantissa = magnitude & 1
    value = tl.where(
        exponent == 0,
        mantissa.to(tl.float32) * 0.5,
        tl.exp2((exponent - 1).to(tl.float32)) * (1.0 + mantissa.to(tl.float32) * 0.5),
    )
    return value


@triton.jit
def _signed_bf16(code, exponent):
    value = _e2m1(code) * tl.exp2((exponent - 127).to(tl.float32))
    # Bitwise sign insertion after scaling preserves the encoded negative zero.
    bits = value.to(tl.int32, bitcast=True) | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def _decode(W, S, O, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // K
    col = offset % K
    packed = tl.load(W + row * (K // 2) + col // 2, row < N, other=0).to(tl.int32)
    code = (packed >> ((col & 1) * 4)) & 15
    exponent = tl.load(S + row * (K // 32) + col // 32, row < N, other=127).to(tl.int32)
    value = _signed_bf16(code, exponent)
    tl.store(O + offset, value.to(O.dtype.element_ty), row < N)


@triton.jit
def _fused_mm(
    X,
    W,
    S,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        x = tl.load(
            X + m[:, None] * K + k[None, :],
            (m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        packed = tl.load(
            W + n[None, :] * (K // 2) + k[:, None] // 2,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        ).to(tl.int32)
        code = (packed >> ((k[:, None] & 1) * 4)) & 15
        exponent = tl.load(
            S + n[None, :] * (K // 32) + k[:, None] // 32,
            (n[None, :] < N) & (k[:, None] < K),
            other=127,
        ).to(tl.int32)
        weight = _signed_bf16(code, exponent)
        acc = tl.dot(x, weight, acc)
    tl.store(O + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def decode_into(packed, scales, out):
    n, k = out.shape
    _decode[(triton.cdiv(n * k, 2048),)](packed, scales, out, n, k, 2048, num_warps=4)
    return out


def fused_mm_into(x, packed, scales, out):
    m, k = x.shape
    n = packed.shape[0]
    _fused_mm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
        x, packed, scales, out, m, n, k, 16, 32, 64, num_warps=4, num_stages=1
    )
    return out

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

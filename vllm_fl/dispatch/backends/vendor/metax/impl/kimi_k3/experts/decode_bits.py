"""Independent exact-bit E2M1/E8M0 decode candidate; 4080 finite code/scale GPU pairs verified byte exact."""

import triton
import triton.language as tl


@triton.jit
def decode_bits(code, scale):
    magnitude = code & 7
    adjust = tl.where(
        magnitude <= 1, -1, tl.where(magnitude <= 3, 0, tl.where(magnitude <= 5, 1, 2))
    )
    biased = scale + adjust
    mantissa = tl.where((magnitude > 1) & ((magnitude & 1) != 0), 64, 0)
    normal = (biased << 7) | mantissa
    subnormal = (128 | mantissa) >> tl.maximum(1 - biased, 0)
    value = tl.where(biased <= 0, subnormal, normal)
    value = tl.where(biased >= 255, 0x7F80, value)
    value = tl.where(magnitude == 0, 0, value)
    value = value | ((code & 8) << 12)
    return value.to(tl.uint16).to(tl.bfloat16, bitcast=True)

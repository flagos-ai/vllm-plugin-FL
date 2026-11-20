# Copyright (c) 2026 BAAI. All rights reserved.

"""Contiguous DV4 top-6 BF16 reduction for large token batches."""

import triton
import triton.language as tl


@triton.jit
def _prefill_moe_sum_kernel(x, out, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    hidden = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = x + token * (6 * 4096) + hidden
    a = tl.load(base).to(tl.float32)
    b = tl.load(base + 4096).to(tl.float32)
    c = tl.load(base + 8192).to(tl.float32)
    d = tl.load(base + 12288).to(tl.float32)
    e = tl.load(base + 16384).to(tl.float32)
    f = tl.load(base + 20480).to(tl.float32)
    # Match ATen Reduce.cuh's four accumulators and final combine order.
    # A sequential top-k loop can change BF16 rounding at halfway values.
    result = (((a + e) + (b + f)) + c) + d
    tl.store(out + token * 4096 + hidden, result)


def prefill_moe_sum(input, output):
    # Explicit configuration avoids sharing FlagGems' autotune cache with
    # decode: its key does not distinguish small and large token counts.
    block, warps = (1024, 4) if input.shape[0] <= 1024 else (2048, 8)
    _prefill_moe_sum_kernel[(input.shape[0], 4096 // block)](
        input, output, BLOCK=block, num_warps=warps, enable_fp_fusion=False,
    )

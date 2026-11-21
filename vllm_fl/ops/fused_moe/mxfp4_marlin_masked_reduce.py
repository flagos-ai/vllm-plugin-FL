# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Mask route outputs during FP32 reduction; preserve pinned accumulation order.

Arithmetic follows flaggems_vllm/ops/moe_sum.py at b5c2c224. The mask is a
load predicate, so invalid dummy-expert NaN/Inf values cannot reach arithmetic.
A flat launch avoids CUDA grid-Y limits for large token counts.
"""

import triton
import triton.language as tl


@triton.jit
def _masked_sum(
    X,
    R,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    PAIR: tl.constexpr,
):
    pid = tl.program_id(0)
    nh = tl.cdiv(H, B)
    row = (pid // nh) * T + tl.arange(0, T)
    h = (pid % nh) * B + tl.arange(0, B)
    inside = (row[:, None] < M) & (h[None, :] < H)
    acc = tl.full((T, B), 0, tl.float32)
    if PAIR:
        for k in tl.static_range(0, K, 2):
            valid = tl.load(R + row * K + k, row < M, other=0)
            a = tl.load(
                X + row[:, None] * K * H + k * H + h[None, :],
                inside & valid[:, None],
                other=0,
            ).to(tl.float32)
            if k + 1 < K:
                valid1 = tl.load(R + row * K + k + 1, row < M, other=0)
                b = tl.load(
                    X + row[:, None] * K * H + (k + 1) * H + h[None, :],
                    inside & valid1[:, None],
                    other=0,
                ).to(tl.float32)
                acc += a + b
            else:
                acc += a
    else:
        for k in tl.static_range(K):
            valid = tl.load(R + row * K + k, row < M, other=0)
            a = tl.load(
                X + row[:, None] * K * H + k * H + h[None, :],
                inside & valid[:, None],
                other=0,
            )
            acc += a.to(tl.float32)
    tl.store(Y + row[:, None] * H + h[None, :], acc, inside)


def masked_sum(input, route_mask, output):
    m, k, h = input.shape
    if m == 0:
        return
    t, b = (1, 512) if m < 128 else (4, 256)
    _masked_sum[(triton.cdiv(m, t) * triton.cdiv(h, b),)](
        input, route_mask, output, m, k, h, t, b, m < 128, num_warps=4
    )

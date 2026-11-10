# Copyright (c) 2026 BAAI. All rights reserved.

"""BF16 compressor projections with deterministic FP32 split-K reduction."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _projection_split_k(
    x_ptr, w_ptr, partial_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    inner = split * (K // SPLITS) + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // SPLITS // BK):
        x = tl.load(
            x_ptr + rows[:, None] * K + inner[None, :],
            rows[:, None] < M, 0,
        )
        w = tl.load(
            w_ptr + cols[None, :] * K + inner[:, None],
            cols[None, :] < N, 0,
        )
        acc = tl.dot(x, w, acc)
        inner += BK
    tl.store(
        partial_ptr + split * M * N + rows[:, None] * N + cols[None, :],
        acc, (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _projection_reduce(
    partial_ptr, output_ptr,
    ELEMENTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        partial_ptr + splits[:, None] * ELEMENTS + offsets[None, :],
        offsets[None, :] < ELEMENTS, 0,
    )
    tl.store(output_ptr + offsets, tl.sum(values, axis=0), offsets < ELEMENTS)


def decode_projection_bf16_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """DV4 M1..256, K4096, N512/1024/2048 contiguous projections.

    Separate partials avoid atomic accumulation and workspace reset kernels.
    Masked row tiles also cover non-aligned and partially filled decode batches.
    The caller guards shape, layout, device and dtype before selecting this path.
    """
    m, k = x.shape
    n = w.shape[0]
    if n == 512:
        bm, bk, splits = 32, 64, 8
    else:
        bm, bk, splits = 64, 128, 8 if n == 1024 else 4
    bn, waves = 64, 4
    if os.getenv("VLLM_FL_HYGON_PROJECTION_TILE_TUNING", "1").lower() not in {
        "0", "off", "false", "native",
    }:
        # Larger decode batches amortize longer K loops. Fewer splits reduce
        # partial traffic; larger M/N tiles reuse weights for the wide C4 GEMM.
        # Keep small-M and N512 configurations: their measured gains were small.
        if n == 2048 and 65 <= m <= 256:
            splits = 2
            if m > 128:
                bm, bn, waves = 128, 128, 8
        elif n == 1024 and 193 <= m <= 256:
            splits = 2
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _projection_split_k[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        x, w, partial, m, n, k, bm, bn, bk, splits,
        num_warps=waves, num_stages=2,
    )
    _projection_reduce[(triton.cdiv(m * n, 256),)](
        partial, output, m * n, splits, 256, num_warps=4,
    )
    return output

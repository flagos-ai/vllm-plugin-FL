# SPDX-License-Identifier: Apache-2.0
"""Copy-only layout fusion for PR-style CP; no floating-point arithmetic."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kv(K, V, OUT, ROWS: tl.constexpr, PADDED: tl.constexpr,
             HEADS: tl.constexpr, DIM: tl.constexpr,
             KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
             VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // (HEADS * DIM)
    head = i // DIM % HEADS
    col = i % DIM
    k = tl.load(K + row * KS0 + head * KS1 + col * KS2,
                row < ROWS, other=0)
    v = tl.load(V + row * VS0 + head * VS1 + col * VS2,
                row < ROWS, other=0)
    dst = row * (2 * HEADS * DIM) + head * DIM + col
    tl.store(OUT + dst, k, row < PADDED)
    tl.store(OUT + dst + HEADS * DIM, v, row < PADDED)


@triton.jit
def _restore_native_kv(IN, RESTORE, OUT, ROWS: tl.constexpr,
                       HEADS: tl.constexpr, DIM: tl.constexpr,
                       FIRST: tl.constexpr, LOCAL_HEADS: tl.constexpr,
                       BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // (LOCAL_HEADS * DIM)
    head = i // DIM % LOCAL_HEADS + FIRST
    col = i % DIM
    source = tl.load(RESTORE + row, row < ROWS, other=0)
    offset = source * (2 * HEADS * DIM) + head * DIM + col
    k = tl.load(IN + offset, row < ROWS, other=0)
    v = tl.load(IN + offset + HEADS * DIM, row < ROWS, other=0)
    tl.store(OUT + i, k, row < ROWS)
    tl.store(OUT + ROWS * LOCAL_HEADS * DIM + i, v, row < ROWS)


def pack_kv(k: torch.Tensor, v: torch.Tensor, padded_rows: int):
    rows, heads, dim = k.shape
    packed = k.new_empty((padded_rows, 2, heads, dim))
    _pack_kv[(triton.cdiv(padded_rows * heads * dim, 1024),)](
        k, v, packed, rows, padded_rows, heads, dim,
        *k.stride(), *v.stride(), BLOCK=1024,
    )
    return packed


def restore_native_kv(gathered, restore, head_start, num_heads):
    rows = restore.numel()
    _, _, heads, dim = gathered.shape
    out = gathered.new_empty((2, rows, num_heads, dim))
    if rows:
        _restore_native_kv[(triton.cdiv(rows * num_heads * dim, 1024),)](
            gathered, restore, out, rows, heads, dim, head_start, num_heads,
            BLOCK=1024,
        )
    return out[0], out[1]


@triton.jit
def _pad_prefill_kv(K, V, SOURCES, OUT, ROWS: tl.constexpr,
                    WIDTH: tl.constexpr, KS0: tl.constexpr, VS0: tl.constexpr,
                    BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    source = tl.load(SOURCES + row, row < ROWS, other=-1)
    valid = (row < ROWS) & (source >= 0)
    k = tl.load(K + source * KS0 + col, valid, other=0)
    v = tl.load(V + source * VS0 + col, valid, other=0)
    tl.store(OUT + i, k, row < ROWS)
    tl.store(OUT + ROWS * WIDTH + i, v, row < ROWS)


def pad_prefill_kv(k, v, sources):
    rows, heads, dim = sources.numel(), k.shape[1], k.shape[2]
    assert k.stride(1) == v.stride(1) == dim
    assert k.stride(2) == v.stride(2) == 1
    out = k.new_empty((2, rows, heads, dim))
    _pad_prefill_kv[(triton.cdiv(rows * heads * dim, 1024),)](
        k, v, sources, out, rows, heads * dim, k.stride(0), v.stride(0),
        BLOCK=1024,
    )
    return out[0], out[1]

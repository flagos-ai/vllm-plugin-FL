# Copyright (c) 2026 BAAI. All rights reserved.
"""Build small decode ragged metadata directly in persistent graph storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ragged_decode_metadata_kernel(
    dense, lengths, flat, indptr,
    stride_row: tl.constexpr, stride_col: tl.constexpr,
    rows, width: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    r = tl.arange(0, BLOCK_ROWS)
    lens = tl.load(lengths + r, r < rows, other=0)
    lens = tl.minimum(tl.maximum(lens, 0), width)
    start = tl.sum(tl.where(r < row, lens, 0), 0)
    count = tl.sum(tl.where(r == row, lens, 0), 0)
    if block == 0:
        tl.store(indptr + row, start)
        if row == rows - 1:
            tl.store(indptr + rows, start + count)
    col = block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    values = tl.load(dense + row * stride_row + col * stride_col,
                     col < count, other=0).to(tl.int32)
    tl.store(flat + start + col, values, col < count)


def build_decode_ragged_into(dense, lengths, flat_buffer, indptr_buffer):
    """Return stable views, or None when the generic builder should be used.

    Only indptr-bounded entries are initialized, matching the upstream packer.
    Redundant prefix sums are cheap for small decode batches and avoid a
    separate scan launch, temporary tensors and two copies to graph buffers.
    """
    rows = dense.shape[0]
    if (not dense.is_cuda or lengths.device != dense.device
            or lengths.dtype != torch.int32 or not lengths.is_contiguous()
            or dense.dtype not in (torch.int32, torch.int64)
            or not 0 < rows <= 256 or lengths.numel() != rows):
        return None
    dense = dense.reshape(rows, -1)
    width = dense.shape[1]
    if width == 0 or flat_buffer.numel() < rows * width or indptr_buffer.numel() < rows + 1:
        return None
    flat = flat_buffer[:rows * width]
    indptr = indptr_buffer[:rows + 1]
    _ragged_decode_metadata_kernel[(rows, triton.cdiv(width, 128))](
        dense, lengths, flat, indptr, dense.stride(0), dense.stride(1),
        rows, width, triton.next_power_of_2(rows), 128,
    )
    return flat, indptr

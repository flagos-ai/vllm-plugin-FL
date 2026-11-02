# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework-owned sparse token/page mapping and stable slot compaction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _convert_sparse_indices_kernel(
    Requests,
    Blocks,
    Indices,
    Physical,
    Lengths,
    Discard,
    request_stride: tl.constexpr,
    block_row_stride: tl.constexpr,
    block_col_stride: tl.constexpr,
    index_row_stride: tl.constexpr,
    index_col_stride: tl.constexpr,
    N_REQUESTS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(Requests + row * request_stride).to(tl.int64)
    # Match PyTorch's negative row indexing for otherwise legal request IDs.
    req = tl.where(req < 0, req + N_REQUESTS, req)
    row_valid = (req >= 0) & (req < N_REQUESTS)
    safe_req = tl.minimum(tl.maximum(req, 0), N_REQUESTS - 1)
    lane = tl.arange(0, TILE)
    count = tl.full((), 0, tl.int32)
    for start in range(tl.cdiv(WIDTH, TILE)):
        col = start * TILE + lane
        # Form only in-bounds load addresses, including inactive tail lanes.
        safe_col = tl.minimum(col, WIDTH - 1).to(tl.int64)
        token = tl.load(
            Indices + row * index_row_stride + safe_col * index_col_stride,
        ).to(tl.int64)
        token = tl.where(col < WIDTH, token, -1)
        logical_block = tl.maximum(token, 0) // BLOCK_SIZE
        valid = (col < WIDTH) & (token >= 0) & (logical_block < N_BLOCKS)
        valid = valid & row_valid
        if N_BLOCKS > 0:
            safe_block = tl.minimum(logical_block, N_BLOCKS - 1)
            block = tl.load(
                Blocks + safe_req * block_row_stride + safe_block * block_col_stride,
            ).to(tl.int64)
        else:
            block = tl.full((TILE,), -1, tl.int64)
        valid = valid & (block >= 0)
        dst = count + tl.cumsum(valid.to(tl.int32), 0) - 1
        physical = block * BLOCK_SIZE + token % BLOCK_SIZE
        # Inactive scatter lanes get distinct, in-bounds discard slots. This
        # avoids masked scatter lowering and any racing writes to valid data.
        destination = tl.where(
            valid,
            Physical + row * WIDTH + tl.maximum(dst, 0).to(tl.int64),
            Discard + row * TILE + lane,
        )
        tl.store(destination, physical)
        count += tl.sum(valid.to(tl.int32), 0)
    tl.store(Lengths + row, count)


def convert_sparse_indices(request_ids, block_table, token_indices, block_size):
    """Map and stably compact sparse token indices to int32 physical slots.

    Inputs are integer request IDs [M], page table [R, B], and logical tokens
    [M, K]. Negative logical tokens, absent pages and out-of-range logical pages
    are removed, duplicates remain, and each output row has a -1 tail. Request
    IDs must be legal page-table rows (negative row indexing is supported).
    This inference-only API does not modify inputs or synchronize the host.
    """
    if request_ids.ndim != 1 or block_table.ndim != 2 or token_indices.ndim != 2:
        raise ValueError("expected request_ids[M], block_table[R,B], indices[M,K]")
    if request_ids.shape[0] != token_indices.shape[0]:
        raise ValueError("request_ids and indices must have the same row count")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    tensors = (request_ids, block_table, token_indices)
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors):
        raise NotImplementedError("sparse indices require int32 or int64 inputs")
    if any(t.device != token_indices.device for t in tensors):
        raise ValueError("all sparse index inputs must be on the same device")
    if token_indices.device.type == "cpu":
        raise NotImplementedError("sparse index conversion requires a Triton device")
    rows, width = token_indices.shape
    if rows and not block_table.shape[0]:
        raise ValueError("nonempty indices require at least one page-table row")
    # Initialize before compaction; a separate launch cannot overwrite its
    # compact prefix even on backends with restricted masked-store lowering.
    output = torch.full(
        (rows, width), -1, dtype=torch.int32, device=token_indices.device
    )
    lengths = torch.zeros((rows,), dtype=torch.int32, device=token_indices.device)
    if rows and width:
        # A fixed-size scan bounds metadata workspace independently of width.
        tile = min(1024, triton.next_power_of_2(width))
        discard = torch.empty(
            (rows, tile), dtype=torch.int32, device=token_indices.device
        )
        with torch.get_device_module(token_indices.device).device(token_indices.device):
            _convert_sparse_indices_kernel[(rows,)](
                request_ids,
                block_table,
                token_indices,
                output,
                lengths,
                discard,
                request_ids.stride(0),
                *block_table.stride(),
                *token_indices.stride(),
                block_table.shape[0],
                block_table.shape[1],
                width,
                block_size,
                tile,
                num_warps=4,
            )
    return output, lengths

# Copyright (c) 2026 BAAI. All rights reserved.
"""One-token decode SWA indices and ragged layout in one GPU launch."""

import numpy as np
import triton
import triton.language as tl


@triton.jit
def _swa_decode_metadata_kernel(
    seq_lens, slots, blocks, token_to_req, valid_tokens,
    dense, lens_out, flat, indptr,
    rows, window: tl.constexpr, block_size: tl.constexpr,
    block_stride: tl.constexpr, dense_stride: tl.constexpr,
    capacity: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_WINDOW: tl.constexpr,
):
    row = tl.program_id(0)
    tail = row * 128 + tl.arange(0, 128)
    tl.store(lens_out + tail, 0, (tail >= rows) & (tail < capacity))
    if row < rows:
        r = tl.arange(0, BLOCK_ROWS)
        seq = tl.load(seq_lens + r, r < rows, other=0)
        valid = tl.load(slots + r, r < rows, other=-1) >= 0
        lengths = tl.where(valid, tl.minimum(seq, window), 0)
        start = tl.sum(tl.where(r < row, lengths, 0), 0)
        count = tl.sum(tl.where(r == row, lengths, 0), 0)
        seq_len = tl.load(seq_lens + row)
        is_valid = tl.load(slots + row) >= 0
        tl.store(token_to_req + row, row)
        tl.store(valid_tokens + row, is_valid)
        tl.store(lens_out + row, count)
        tl.store(indptr + row, start)
        if row == rows - 1:
            tl.store(indptr + rows, start + count)
        col = tl.arange(0, BLOCK_WINDOW)
        pos = tl.maximum(seq_len - window, 0) + col
        block = tl.load(blocks + row * block_stride + pos // block_size,
                        col < count, other=0)
        index = block * block_size + pos % block_size
        tl.store(dense + row * dense_stride + col,
                 tl.where(col < count, index, -1), col < window)
        tl.store(flat + start + col, index, col < count)


def try_build_swa_decode(builder, common, metadata_cls):
    """Use the specialization only for exactly one query token per request."""
    rows = common.num_reqs
    if (common.max_query_len != 1 or common.num_actual_tokens != rows
            or not 0 < rows <= 256 or builder.num_speculative_tokens != 0
            or not 0 < builder.window_size <= 1024):
        return None
    # Check CPU scheduler metadata, never read GPU lengths back to the host.
    query_loc = common.query_start_loc_cpu.numpy()
    if not np.array_equal(query_loc, np.arange(rows + 1)):
        return None
    capacity = builder.decode_swa_lens.numel()
    _swa_decode_metadata_kernel[(max(rows, triton.cdiv(capacity, 128)),)](
        common.seq_lens, common.slot_mapping, common.block_table_tensor,
        builder.token_to_req_indices, builder.is_valid_token,
        builder.decode_swa_indices, builder.decode_swa_lens,
        builder.decode_swa_ragged_indices_buffer,
        builder.decode_swa_ragged_indptr_buffer,
        rows, builder.window_size, builder.block_size,
        common.block_table_tensor.stride(0), builder.decode_swa_indices.stride(0),
        capacity, triton.next_power_of_2(rows), triton.next_power_of_2(builder.window_size),
    )
    return metadata_cls(
        block_table=common.block_table_tensor, slot_mapping=common.slot_mapping,
        block_size=builder.block_size, seq_lens=common.seq_lens,
        query_start_loc=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        is_valid_token=builder.is_valid_token[:rows],
        token_to_req_indices=builder.token_to_req_indices[:rows],
        decode_swa_indices=builder.decode_swa_indices[:rows],
        decode_swa_lens=builder.decode_swa_lens[:rows],
        num_decodes=rows, num_decode_tokens=rows,
        decode_swa_ragged_indices=builder.decode_swa_ragged_indices_buffer[:rows * builder.window_size],
        decode_swa_ragged_indptr=builder.decode_swa_ragged_indptr_buffer[:rows + 1],
    )

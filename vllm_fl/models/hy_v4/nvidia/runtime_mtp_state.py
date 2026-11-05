# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 tail state update: native Eagle integer rules plus inactive-row mask."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ACTIVE_B"])
def _mtp_tail_step_kernel(
    P,
    S,
    T,
    Q,
    OP,
    OS,
    SLOT,
    COUNT,
    PSTRIDE: tl.constexpr,
    OPSTRIDE: tl.constexpr,
    TSTRIDE: tl.constexpr,
    NCOLS: tl.constexpr,
    AXES: tl.constexpr,
    ACTIVE_B,
    BLOCK_SIZE: tl.constexpr,
    MAX_LEN: tl.constexpr,
):
    r = tl.program_id(0)
    active = r < ACTIVE_B
    begin = tl.load(Q + r)
    end = tl.load(Q + r + 1)
    active = active & (end - begin == 1)
    # Match eagle_step_slot_mapping_metadata_kernel, including its max-length
    # wrap-to-zero/seq-to-one rule. Padding/capture dummy rows never write KV.
    position = tl.load(P + r)
    new_position = position + 1
    exceeds = new_position >= MAX_LEN
    clamped = tl.where(exceeds, 0, new_position)
    block = tl.minimum(clamped // BLOCK_SIZE, NCOLS - 1)
    block_id = tl.load(T + r * TSTRIDE + block, mask=active, other=0)
    slot = block_id * BLOCK_SIZE + clamped % BLOCK_SIZE
    slot = tl.where(active & ~exceeds, slot, -1)
    seq = tl.load(S + r)
    seq = tl.minimum(tl.where(exceeds, 1, seq + 1), MAX_LEN)
    seq = tl.where(active, seq, 0)
    clamped = tl.where(active, clamped, 0)
    for axis in tl.static_range(AXES):
        # Native M-RoPE tail update copies temporal plane0 into all planes.
        # It does not independently add one to each spatial plane.
        tl.store(OP + axis * OPSTRIDE + r, clamped)
    tl.store(OS + r, seq)
    tl.store(SLOT + r, slot)
    tl.store(COUNT + r, active.to(tl.int32))


def mtp_tail_step(
    positions,
    seq_lens,
    block_table,
    query_start_loc,
    out_positions,
    out_seq_lens,
    out_slots,
    out_counts,
    *,
    active_batch: int,
    block_size: int,
    max_model_len: int,
):
    """All outputs are caller-owned; no allocation, CPU read or fallback."""
    tensors = (
        positions,
        seq_lens,
        block_table,
        query_start_loc,
        out_positions,
        out_seq_lens,
        out_slots,
        out_counts,
    )
    if not all(x.is_cuda and x.device == positions.device for x in tensors):
        raise ValueError("mtp_tail_step requires one CUDA device")
    rows = seq_lens.numel()
    if (
        positions.ndim not in (1, 2)
        or positions.shape[-1] != rows
        or out_positions.shape != positions.shape
        or positions.dtype != torch.int64
        or out_positions.dtype != torch.int64
        or seq_lens.ndim != 1
        or seq_lens.dtype != torch.int32
        or out_seq_lens.dtype != torch.int32
        or out_seq_lens.shape != (rows,)
        or block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.shape[0] != rows
        or block_table.shape[1] < 1
        or query_start_loc.dtype != torch.int32
        or query_start_loc.shape != (rows + 1,)
        or out_slots.dtype != torch.int64
        or out_slots.shape != (rows,)
        or out_counts.dtype != torch.int32
        or out_counts.shape != (rows,)
        or not 0 < active_batch <= rows
        or block_size <= 0
        or max_model_len <= 0
    ):
        raise ValueError("unsupported HY4 tail state layout")
    if (
        positions.stride(-1) != 1
        or out_positions.stride(-1) != 1
        or block_table.stride(1) != 1
        or any(
            not x.is_contiguous()
            for x in (seq_lens, query_start_loc, out_seq_lens, out_slots, out_counts)
        )
    ):
        raise ValueError("unsupported HY4 tail state stride")
    axes = positions.shape[0] if positions.ndim == 2 else 1
    _mtp_tail_step_kernel[(rows,)](
        positions,
        seq_lens,
        block_table,
        query_start_loc,
        out_positions,
        out_seq_lens,
        out_slots,
        out_counts,
        positions.stride(0) if positions.ndim == 2 else rows,
        out_positions.stride(0) if out_positions.ndim == 2 else rows,
        block_table.stride(0),
        block_table.shape[1],
        axes,
        active_batch,
        block_size,
        max_model_len,
        num_warps=1,
    )


__all__ = ["mtp_tail_step"]

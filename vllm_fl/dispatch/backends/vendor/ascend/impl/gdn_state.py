# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Copy active GDN rows without materializing a strided shared cache."""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _copy_rows(
    State,
    Indices,
    Rows,
    NUM_SLOTS: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Indices + row).to(tl.int64)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    state_offset = (
        slot * S0
        + (offset // (D2 * D3)) * S1
        + ((offset // D3) % D2) * S2
        + (offset % D3) * S3
    )
    active = (slot > 0) & (slot < NUM_SLOTS) & (offset < ROW_SIZE)
    if SCATTER:
        value = tl.load(Rows + row * ROW_SIZE + offset, mask=offset < ROW_SIZE, other=0)
        tl.store(State + state_offset, value, mask=active)
    else:
        value = tl.load(State + state_offset, mask=active, other=0)
        tl.store(Rows + row * ROW_SIZE + offset, value, mask=offset < ROW_SIZE)


def _copy(state, indices, rows, *, scatter):
    if state.ndim not in (3, 4):
        raise ValueError("GDN state rows must have two or three dimensions")
    shape = (*state.shape[1:], *((1,) if state.ndim == 3 else ()))
    strides = (*state.stride(), *((0,) if state.ndim == 3 else ()))
    row_size = math.prod(shape)
    if indices.numel() == 0 or row_size == 0:
        return
    _copy_rows[(indices.numel(), triton.cdiv(row_size, 1024))](
        state,
        indices,
        rows,
        NUM_SLOTS=state.shape[0],
        ROW_SIZE=row_size,
        D2=shape[1],
        D3=shape[2],
        S0=strides[0],
        S1=strides[1],
        S2=strides[2],
        S3=strides[3],
        SCATTER=scatter,
        BLOCK=1024,
    )


def gather_state_rows(state, indices):
    rows = torch.empty(
        (indices.numel(), *state.shape[1:]), device=state.device, dtype=state.dtype
    )
    _copy(state, indices, rows, scatter=False)
    return rows


def scatter_state_rows(state, indices, rows):
    # Only the active working set is compacted. The shared backing stays put.
    _copy(state, indices, rows.contiguous(), scatter=True)

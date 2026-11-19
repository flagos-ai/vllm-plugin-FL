# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-external PLE history preparation for MRV1 without speculation."""

import triton
import triton.language as tl


@triton.jit
def _update_ple_history(
    ids,
    starts,
    metadata,
    previous,
    following,
    context,
    num_reqs,
    context_len: tl.constexpr,
    eos: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < context_len
    if row < num_reqs:
        prior_row = tl.load(metadata + row * (context_len + 1))
        seed = tl.load(
            metadata + row * (context_len + 1) + 1 + col,
            mask=valid,
            other=eos,
        )
        old = tl.load(
            previous + tl.maximum(prior_row, 0) * context_len + col,
            mask=valid & (prior_row >= 0),
            other=eos,
        )
        history = tl.where(prior_row >= 0, old, seed)
        tl.store(context + row * context_len + col, history, mask=valid)
        begin = tl.load(starts + row)
        end = tl.load(starts + row + 1)
        length = end - begin
        source = length - context_len + col
        # A short query retains the unconsumed part of the old suffix.
        shifted_col = length + col
        shifted_seed = tl.load(
            metadata + row * (context_len + 1) + 1 + shifted_col,
            mask=valid & (source < 0),
            other=eos,
        )
        shifted_old = tl.load(
            previous + tl.maximum(prior_row, 0) * context_len + shifted_col,
            mask=valid & (source < 0) & (prior_row >= 0),
            other=eos,
        )
        retained = tl.where(prior_row >= 0, shifted_old, shifted_seed)
        consumed = tl.load(ids + begin + source, mask=valid & (source >= 0), other=eos)
        tail = tl.where(source >= 0, consumed, retained)
        tl.store(following + row * context_len + col, tail, mask=valid)
    else:
        tl.store(context + row * context_len + col, eos, mask=valid)
        tl.store(following + row * context_len + col, eos, mask=valid)


def update_ple_history(
    ids, starts, metadata, previous, following, context, num_reqs, eos
):
    """Read the old suffix before updating it, even when batch slots swap."""
    if previous.data_ptr() == following.data_ptr():
        raise ValueError("PLE history updates require distinct input/output buffers")
    _update_ple_history[(context.shape[0],)](
        ids,
        starts,
        metadata,
        previous,
        following,
        context,
        num_reqs,
        context.shape[1],
        eos,
        BLOCK=triton.next_power_of_2(context.shape[1]),
        num_warps=4,
    )

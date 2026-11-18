# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""Framework token bookkeeping for padded speculative requests."""

from vllm.triton_utils import tl, triton


@triton.jit
def prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,
    discard_request_mask_ptr,
    backup_next_token_ids_ptr,
    next_token_ids_ptr,
    valid_sampled_tokens_count_ptr,
    vocab_size,
    num_sampled_tokens_per_req,
    num_reqs,
    stride_sampled_token_ids,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    """Select the last valid token and count accepted positions, without scores."""
    row = tl.program_id(0)
    if row >= num_reqs:
        return
    next_token = tl.load(backup_next_token_ids_ptr + row)
    valid_count = tl.full((), 0, tl.int32)
    if not tl.load(discard_request_mask_ptr + row):
        offsets = tl.arange(0, BLOCK_SIZE_TOKENS)
        in_row = offsets < num_sampled_tokens_per_req
        tokens = tl.load(
            sampled_token_ids_ptr + row * stride_sampled_token_ids + offsets,
            mask=in_row,
            other=-1,
        )
        valid = (tokens != -1) & (tokens < vocab_size) & in_row
        # Some compiler backends reduce a bool to int32 rather than uint32.
        # Both branches publish the upstream output buffer's signed dtype.
        valid_count = tl.sum(valid.to(tl.int32))
        if valid_count > 0:
            last = tl.max(tl.where(valid, offsets, -1))
            next_token = tl.sum(tl.where(offsets == last, tokens, 0)).to(tl.int32)
    tl.store(next_token_ids_ptr + row, next_token)
    tl.store(valid_sampled_tokens_count_ptr + row, valid_count)

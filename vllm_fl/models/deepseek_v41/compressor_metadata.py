# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2 request/slot metadata only; FP32 compression values remain in FGV."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _metadata_kernel(
    POS,
    PADDING,
    START,
    MAPPING,
    SLOTS,
    REQUESTS,
    QUERY,
    N,
    NREQS,
    MAXREQS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    end = tl.load(START + NREQS)
    query_value = tl.load(START + index, index <= NREQS, other=0)
    query_value = tl.where(index <= NREQS, query_value, end)
    tl.store(QUERY + index, query_value, index <= MAXREQS)
    valid = (index < N) & (index < end)
    padded = tl.load(PADDING + index, index < N, other=True)
    valid &= ~padded
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), NREQS, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        active = lo < hi
        mid = (lo + hi) // 2
        stop = tl.load(START + mid + 1, active, other=0)
        right = index >= stop
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    request = tl.minimum(lo, NREQS - 1).to(tl.int64)
    state = tl.load(MAPPING + request, valid, other=-1).to(tl.int64)
    position = tl.load(POS + index, index < N, other=-1).to(tl.int64)
    valid &= (state >= 0) & (state < MAXREQS) & (position >= 0)
    slot = state * CAPACITY + position % CAPACITY
    tl.store(SLOTS + index, tl.where(valid, slot, -1), index < N)
    tl.store(REQUESTS + index, tl.where(valid, request, -1), index < N)


def prepare_compressor_metadata(
    positions,
    is_padding,
    query_start_loc,
    idx_mapping,
    num_reqs,
    capacity,
    state_slots,
    token_to_req,
    query_output,
):
    """Fill stable ring slots and packed request indices from native V2 metadata.

    Slots use stable RequestState indices, while token_to_req addresses packed
    query boundaries. Padding or unknown request slots produce -1. Query output
    has max_requests+1 entries with repeated terminal boundaries after num_reqs.
    Caller owns monotone query boundaries and one contiguous chunk per request.
    All operations copy/construct framework metadata; no model values are read.
    """
    n = positions.numel()
    if not isinstance(num_reqs, int) or isinstance(num_reqs, bool) or num_reqs < 0:
        raise ValueError("num_reqs must be a nonnegative integer")
    if (
        not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity <= 0
        or capacity % 2
    ):
        raise ValueError("ring capacity must be a positive even integer")
    if query_output.ndim != 1 or query_output.numel() < 2:
        raise ValueError("query output must allocate positive request capacity")
    maximum = query_output.numel() - 1
    if num_reqs > maximum:
        raise ValueError("request batch exceeds ring capacity")
    if (
        positions.ndim != 1
        or is_padding.shape != (n,)
        or is_padding.dtype != torch.bool
    ):
        raise ValueError("positions/padding must be aligned vectors")
    if query_start_loc.ndim != 1 or query_start_loc.numel() < num_reqs + 1:
        raise ValueError("query boundaries are too short")
    if idx_mapping.ndim != 1 or idx_mapping.numel() < num_reqs:
        raise ValueError("request mapping is too short")
    if state_slots.shape != (n,) or state_slots.dtype != torch.int64:
        raise ValueError("state slots must be INT64 [tokens]")
    if token_to_req.shape != (n,) or token_to_req.dtype != torch.int32:
        raise ValueError("token requests must be INT32 [tokens]")
    if query_output.dtype != torch.int32:
        raise ValueError("query output must be INT32")
    inputs = [positions, query_start_loc, idx_mapping]
    if any(x.dtype not in (torch.int32, torch.int64) for x in inputs):
        raise ValueError("native request metadata must be INT32/INT64")
    tensors = inputs + [is_padding, state_slots, token_to_req, query_output]
    if positions.device.type != "cuda" or any(
        x.device != positions.device or not x.is_contiguous() for x in tensors
    ):
        raise ValueError(
            "all metadata must be contiguous on one CUDA-compatible device"
        )
    outputs = [state_slots, token_to_req, query_output]
    for i, value in enumerate(outputs):
        other = inputs + [is_padding] + outputs[:i] + outputs[i + 1 :]
        if value.numel() and any(
            x.numel()
            and x.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            for x in other
        ):
            raise ValueError("metadata outputs must not alias inputs or each other")
    if not num_reqs:
        state_slots.fill_(-1)
        token_to_req.fill_(-1)
        query_output.zero_()
        return
    with torch.cuda.device(positions.device):
        _metadata_kernel[(triton.cdiv(max(n, maximum + 1), 128),)](
            positions,
            is_padding,
            query_start_loc,
            idx_mapping,
            state_slots,
            token_to_req,
            query_output,
            n,
            num_reqs,
            maximum,
            capacity,
            128,
        )

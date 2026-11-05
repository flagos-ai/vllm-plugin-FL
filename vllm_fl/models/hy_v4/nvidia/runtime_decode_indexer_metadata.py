# SPDX-License-Identifier: Apache-2.0
"""Complete decode metadata chain in one GPU launch, with caller-owned outputs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_TOKENS": tokens}, num_warps=warps, num_stages=1)
        for tokens in (16, 32, 64)
        for warps in (4, 8)
    ],
    key=["B", "N", "COLS", "COUNTS_I32", "SEQ_I32", "CS", "SR", "SC", "PADDED"],
    strategy="default",
)
@triton.jit
def _decode_indexer_metadata_kernel(
    Counts,
    Seq,
    Rows,
    Lengths,
    B: tl.constexpr,
    N: tl.constexpr,
    COLS: tl.constexpr,
    CS: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    COUNTS_I32: tl.constexpr,
    SEQ_I32: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    request = tl.program_id(0)
    requests = tl.arange(0, BLOCK_B)
    counts = tl.load(Counts + requests * CS, requests < B, other=0).to(tl.int64)
    inclusive = tl.cumsum(counts, 0)
    exclusive = inclusive - counts
    start = tl.sum(tl.where(requests == request, exclusive, 0), 0)
    count = tl.load(Counts + request * CS).to(tl.int64)
    offsets = tl.arange(0, BLOCK_TOKENS).to(tl.int64)

    if COLS <= 1:
        sequence = tl.load(Seq + request * SR)
        # Preserve the old subtraction's dtype boundary before the I64 step
        # promotes it. Do not replace this with an all-I64 algebraic shortcut.
        if COUNTS_I32 and SEQ_I32:
            base_length = (
                (sequence.to(tl.int32) - count.to(tl.int32)).to(tl.int32).to(tl.int64)
            )
        else:
            base_length = sequence.to(tl.int64) - count

    # Same complete path for one/many requests and one/many tokens per request.
    # Zero-count requests perform no stores. Prefix intervals are disjoint.
    # Under counts>=0 and sum(counts)==N<=1024 this loop bound fits I32.
    # Prefix and address arithmetic remain I64.
    for first in tl.range(0, count.to(tl.int32), BLOCK_TOKENS):
        step = first + offsets
        token = start + step
        valid = (step < count) & (token < N) & (token >= 0)
        if COLS > 1:
            length = tl.load(
                Seq + request * SR + step * SC, valid & (step < COLS), other=0
            )
        else:
            length = base_length + step + 1
        tl.store(Rows + token, request.to(tl.int64), valid)
        tl.store(Lengths + token, length, valid)

    if PADDED:
        # N is output capacity P in this mode; the logical token count is the
        # GPU sum of counts. Complete prefix math above is unchanged.
        total = tl.sum(counts, 0)
        # Cover all tail rows even for B=1, P=1024. No cross-program overlap:
        # each program owns its striped tail blocks, all beyond the prefix.
        for first in tl.range(request * BLOCK_TOKENS, N, B * BLOCK_TOKENS):
            token = first + offsets
            tail = (token >= total) & (token < N)
            tl.store(Rows + token, 0, tail)
            tl.store(Lengths + token, 0, tail)


def _span(tensor):
    if tensor.numel() == 0:
        return None
    last = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    first = tensor.data_ptr()
    return first, first + (last + 1) * tensor.element_size()


def _overlap(first, second):
    return (
        first is not None
        and second is not None
        and max(first[0], second[0]) < min(first[1], second[1])
    )


def decode_indexer_metadata(
    counts, seq_lens, rows_out, lengths_out, n: int, *, padded: bool = False
):
    """Exact valid-input equivalent of the frozen adapter metadata chain.

    Device-value contract: counts>=0, sum(counts)==n; wide-2D seq has enough
    columns for every count. No host read/sync checks these device values.
    In padded=True mode n is P, sum(counts)=N may be <=P, and tail N:P
    is fully initialized to safe row0/length0. Existing mode is unchanged.
    Capacity above n is not logical padding and is never written. Outputs
    must not alias one another or either input. No torch compute fallback.
    """
    if type(padded) is not bool:
        raise TypeError("padded metadata mode must be a bool")
    tensors = (counts, seq_lens, rows_out, lengths_out)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("decode metadata requires four tensors")
    if type(n) is not int or not 0 <= n <= 1024:
        raise ValueError("logical decode token count must be in [0,1024]")
    if counts.ndim != 1 or not 0 <= counts.shape[0] <= 1024:
        raise ValueError("counts must be 1D with at most 1024 entries")
    b = counts.shape[0]
    if counts.dtype not in (torch.int32, torch.int64) or seq_lens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise NotImplementedError(
            "only int32/int64 counts and sequence lengths are supported"
        )
    if (
        seq_lens.ndim not in (1, 2)
        or seq_lens.shape[0] != b
        or (seq_lens.ndim == 2 and seq_lens.shape[1] == 0 and n > 0)
    ):
        raise ValueError("sequence lengths must have shape (B,), (B,1), or (B,C)")
    if counts.device.type != "cuda" or any(
        value.device != counts.device for value in tensors
    ):
        raise NotImplementedError(
            "all metadata and outputs must be on the same CUDA device"
        )
    if any(stride < 0 for value in tensors for stride in value.stride()):
        raise NotImplementedError("negative strides are unsupported")
    cols = seq_lens.shape[1] if seq_lens.ndim == 2 else 1
    expected_length_dtype = seq_lens.dtype if cols > 1 else torch.int64
    if rows_out.dtype != torch.int64 or lengths_out.dtype != expected_length_dtype:
        raise ValueError("output dtypes must match original rows/lengths promotion")
    if any(
        value.ndim != 1 or value.shape[0] < n or value.stride(0) != 1
        for value in (rows_out, lengths_out)
    ):
        raise ValueError(
            "caller-owned outputs must be contiguous 1D buffers with capacity>=n"
        )
    spans = [_span(value) for value in tensors]
    if _overlap(spans[2], spans[3]) or any(
        _overlap(output, source) for output in spans[2:] for source in spans[:2]
    ):
        raise ValueError("outputs must not overlap each other or the inputs")
    if b == 0 and n != 0:
        raise ValueError("empty counts require logical n=0")
    if n:
        with torch.cuda.device(counts.device):
            _decode_indexer_metadata_kernel[(b,)](
                counts,
                seq_lens,
                rows_out,
                lengths_out,
                B=b,
                N=n,
                COLS=cols,
                CS=counts.stride(0),
                SR=seq_lens.stride(0),
                SC=seq_lens.stride(1) if seq_lens.ndim == 2 else 0,
                COUNTS_I32=counts.dtype == torch.int32,
                SEQ_I32=seq_lens.dtype == torch.int32,
                PADDED=padded,
                BLOCK_B=triton.next_power_of_2(b),
            )
    return rows_out[:n], lengths_out[:n]

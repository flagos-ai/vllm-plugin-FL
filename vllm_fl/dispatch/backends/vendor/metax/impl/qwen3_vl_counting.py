# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _find_first_keep(
    cumulative_ptr,
    top_p_ptr,
    first_keep_ptr,
    n: tl.constexpr,
    m: tl.constexpr,
    block_rows: tl.constexpr,
):
    rows = tl.arange(0, block_rows)
    valid = rows < m
    cutoff = 1.0 - tl.load(top_p_ptr + rows, mask=valid, other=1.0)
    lo = tl.zeros((block_rows,), dtype=tl.int32)
    hi = tl.full((block_rows,), n - 1, dtype=tl.int32)
    for _ in range(18):
        mid = (lo + hi) // 2
        value = tl.load(
            cumulative_ptr + rows * n + mid, mask=valid, other=1.0
        )
        remove = (value <= cutoff) & (mid < n - 1)
        lo = tl.where(remove, mid + 1, lo)
        hi = tl.where(remove, hi, mid)
    tl.store(first_keep_ptr + rows, lo, mask=valid)


@triton.jit
def _threshold_counts(
    logits_ptr,
    sorted_logits_ptr,
    first_keep_ptr,
    less_counts_ptr,
    equal_counts_ptr,
    n: tl.constexpr,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n: tl.constexpr = tl.cdiv(n, block_n)
    row = pid // grid_n
    cols = (pid % grid_n) * block_n + tl.arange(0, block_n)
    valid = cols < n
    first_keep = tl.load(first_keep_ptr + row)
    threshold = tl.load(sorted_logits_ptr + row * n + first_keep)
    values = tl.load(logits_ptr + row * n + cols, mask=valid, other=0)
    values = values.to(tl.bfloat16).to(tl.float32)
    count_offset = row * grid_n + pid % grid_n
    tl.store(
        less_counts_ptr + count_offset,
        tl.sum(((values < threshold) & valid).to(tl.int32), axis=0),
    )
    tl.store(
        equal_counts_ptr + count_offset,
        tl.sum(((values == threshold) & valid).to(tl.int32), axis=0),
    )


@triton.jit
def _stable_threshold_mask(
    logits_ptr,
    sorted_logits_ptr,
    first_keep_ptr,
    less_counts_ptr,
    equal_counts_ptr,
    n: tl.constexpr,
    block_n: tl.constexpr,
    count_block: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n: tl.constexpr = tl.cdiv(n, block_n)
    chunk = pid % grid_n
    row = pid // grid_n
    cols = chunk * block_n + tl.arange(0, block_n)
    valid = cols < n
    first_keep = tl.load(first_keep_ptr + row)
    threshold = tl.load(sorted_logits_ptr + row * n + first_keep)

    count_cols = tl.arange(0, count_block)
    count_valid = count_cols < grid_n
    count_offsets = row * grid_n + count_cols
    less_counts = tl.load(less_counts_ptr + count_offsets, mask=count_valid, other=0)
    equal_counts = tl.load(equal_counts_ptr + count_offsets, mask=count_valid, other=0)
    total_less = tl.sum(less_counts, axis=0)
    prior_equal = tl.sum(
        tl.where(count_valid & (count_cols < chunk), equal_counts, 0), axis=0
    )
    tie_remove_count = first_keep - total_less

    offsets = row * n + cols
    original_values = tl.load(logits_ptr + offsets, mask=valid, other=0)
    values = original_values.to(tl.bfloat16).to(tl.float32)
    is_equal = (values == threshold) & valid
    local_equal_rank = tl.cumsum(is_equal.to(tl.int32), axis=0) - 1
    remove_equal = is_equal & (
        prior_equal + local_equal_rank < tie_remove_count
    )
    remove = (values < threshold) | remove_equal
    tl.store(
        logits_ptr + offsets,
        tl.where(remove, -float("inf"), original_values),
        mask=valid,
    )


def _ordered_bf16_keys(logits: torch.Tensor) -> torch.Tensor:
    raw = logits.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(
        (raw & 0x8000) != 0,
        (~raw) & 0xFFFF,
        raw ^ 0x8000,
    ).to(torch.int64)


def _counting_sorted_values(logits: torch.Tensor) -> torch.Tensor:
    m, n = logits.shape
    keys = _ordered_bf16_keys(logits)
    counts = torch.zeros((m, 65536), dtype=torch.int32, device=logits.device)
    counts.scatter_add_(1, keys, torch.ones_like(keys, dtype=torch.int32))
    prefix = counts.cumsum(-1)
    positions = torch.arange(n, device=logits.device, dtype=torch.int32)
    positions = positions.unsqueeze(0).expand(m, n).contiguous()
    sorted_keys = torch.searchsorted(
        prefix, positions, right=True, out_int32=True
    )
    raw = torch.where(
        sorted_keys < 0x8000,
        (~sorted_keys) & 0xFFFF,
        sorted_keys ^ 0x8000,
    ).to(torch.int32)
    return (raw << 16).view(torch.float32)


def apply_bf16_top_k_top_p(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    # Approximate selection on the BF16 grid. Compare thresholds on the same
    # grid, but preserve original FP32 values for surviving tokens.
    # BF16 selection is approximate, not a losslessness claim.
    sorted_logits = _counting_sorted_values(logits)
    n = logits.shape[1]
    top_k_index = n - top_k.to(torch.int64)
    top_k_threshold = sorted_logits.gather(1, top_k_index.unsqueeze(1))
    sorted_logits.masked_fill_(sorted_logits < top_k_threshold, -float("inf"))

    cumulative = sorted_logits.softmax(dim=-1)
    torch.cumsum(cumulative, dim=-1, out=cumulative)
    first_keep = torch.empty(
        (logits.shape[0],), dtype=torch.int32, device=logits.device
    )
    block_rows = triton.next_power_of_2(logits.shape[0])
    _find_first_keep[(1,)](
        cumulative,
        top_p,
        first_keep,
        n=n,
        m=logits.shape[0],
        block_rows=block_rows,
    )

    block_n = 256
    grid_n = triton.cdiv(n, block_n)
    less_counts = torch.empty(
        (logits.shape[0], grid_n), dtype=torch.int32, device=logits.device
    )
    equal_counts = torch.empty_like(less_counts)
    grid = (logits.shape[0] * grid_n,)
    _threshold_counts[grid](
        logits,
        sorted_logits,
        first_keep,
        less_counts,
        equal_counts,
        n=n,
        block_n=block_n,
    )
    _stable_threshold_mask[grid](
        logits,
        sorted_logits,
        first_keep,
        less_counts,
        equal_counts,
        n=n,
        block_n=block_n,
        count_block=1024,
    )
    return logits

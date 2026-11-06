# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool index mapping and top-k expansion metadata for the GLM sparse indexer.

The framework owns integer slot/page/token mapping; cache compression and
FP8 numerical operators are public FlagGems-vllm APIs.
"""

from __future__ import annotations

import torch


def compute_pooled_write_locs(
    page_table_64: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Map logical pooled-K ids to physical flat cache slots.

    ``pool_size`` consecutive tokens share one pool slot that lives at the
    *first* token page of each page-group. ``page_table_64`` maps token pages
    to physical block ids; we gather the block id of each pool's page-group
    and add the in-block pool offset.
    """
    assert page_table_64.ndim == 1
    pool_ids = pool_ids.to(torch.int64)
    block_size = 64  # indexer cache page size (matches sglang hard-code)
    pool_page_group = torch.div(pool_ids, block_size, rounding_mode="floor")
    token_page_row = pool_page_group * pool_size
    packed_page = page_table_64.index_select(0, token_page_row.to(torch.int64))
    return packed_page.to(torch.int64) * block_size + torch.remainder(
        pool_ids, block_size
    )


def build_pooled_page_table(
    page_table: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Build a pool-granular page table by taking every ``pool_size``-th
    token-page column (one pool maps to ``pool_size`` token pages).

    Uses gather (not strided slicing) so the result is always a fresh
    row-major tensor — some downstream kernels require stride(-1) == 1.
    """
    block_size = page_table.shape[-1]
    assert (
        block_size % pool_size == 0
    ), f"pool_size ({pool_size}) must divide page columns ({block_size})"
    idx = torch.arange(0, block_size, pool_size, device=page_table.device)
    return page_table[..., idx].contiguous()


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    """Number of pools to select so that expanding yields ``topk`` tokens."""
    assert topk % pool_size == 0
    return topk // pool_size


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor."""
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after expanded history tokens.

    ``index_kpool_always_select_tail`` keeps the (incomplete) trailing pool so
    the most recent tokens are always attended to.
    """
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool

    # tail tokens: [pool_len*pool_size, seq_len) for each row.
    pool_len = pool_lens.to(torch.int32)
    tail_start = pool_len * pool_size
    seq_len = seq_lens.to(torch.int32)
    tail_count = seq_len - tail_start  # in [0, pool_size)

    cols = torch.arange(out_cols, device=topk_result.device)[None, :]
    history_len = n_cols
    is_history = cols < history_len
    tail_off = cols - history_len
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])

    # safe_hist must be per-row [rows, out_cols] so the gather reads each row's
    # OWN history. cols is [1, out_cols]; if used directly, gather (which does
    # NOT broadcast the index) would read only row 0 of topk_result, making every
    # query inherit row 0's history (empty for the first token) and lose all its
    # selected tokens — only the per-row tail would survive. This only manifests
    # for multi-row sparse PREFILL (decode has 1 row, so it reads its own row 0).
    safe_hist = torch.minimum(cols, torch.full_like(cols, n_cols - 1)).expand(
        rows, out_cols
    )
    history_val = torch.gather(topk_result, 1, safe_hist)

    tail_raw = tail_start[:, None] + tail_off
    tail_val = tail_raw.to(torch.int32)
    if page_table is not None:
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail_val = torch.gather(page_table, 1, safe_tail).to(torch.int32)
    elif topk_offsets is not None:
        tail_val = (tail_raw + topk_offsets.to(torch.int64).unsqueeze(1)).to(
            torch.int32
        )

    out = torch.where(is_history, history_val, -1)
    out = torch.where(is_tail, tail_val, out)
    return out

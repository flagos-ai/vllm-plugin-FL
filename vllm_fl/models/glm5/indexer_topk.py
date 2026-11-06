# SPDX-License-Identifier: Apache-2.0
"""GLM row metadata adapters over existing PyTorch TopK operations."""

import torch


def _select(logits, starts, ends, indices, num_rows, stride0, stride1, top_k):
    if (stride0, stride1) != logits.stride():
        raise ValueError("TopK logits strides do not match the tensor view")
    if indices.shape[0] < num_rows or indices.shape[1] < top_k:
        raise ValueError("TopK output buffer is too small")
    columns = logits.shape[1]
    count = min(top_k, columns)
    indices[:num_rows].fill_(-1)
    if not count:
        return
    positions = torch.arange(columns, device=logits.device)
    # Bound temporary masks independently of the batch size.
    for row in range(0, num_rows, 32):
        stop = min(row + 32, num_rows)
        start = starts[row:stop, None]
        end = ends[row:stop, None]
        valid = (positions[None, :] >= start) & (positions[None, :] < end)
        scores = logits[row:stop].masked_fill(~valid, -torch.inf)
        selected = torch.topk(scores, count, dim=1, sorted=False).indices
        selected_valid = valid.gather(1, selected)
        # Valid -inf values can tie with padding. Fill any invalid choices
        # from unselected valid columns, retaining uniqueness and row bounds.
        remaining = valid.scatter(1, selected, False)
        replacements = torch.topk(remaining.to(torch.int32), count, dim=1).indices
        rank = (~selected_valid).cumsum(1) - 1
        chosen = torch.where(
            selected_valid, selected, replacements.gather(1, rank.clamp_min(0))
        )
        chosen_valid = selected_valid | (rank < remaining.sum(1, keepdim=True))
        relative = torch.where(chosen_valid, chosen - start, columns)
        # Canonical index order makes the subsequent sparse reduction stable.
        relative = relative.sort(dim=1).values
        indices[row:stop, :count].copy_(torch.where(relative < columns, relative, -1))


def topk_prefill(logits, starts, ends, indices, num_rows, stride0, stride1, top_k):
    return _select(logits, starts, ends, indices, num_rows, stride0, stride1, top_k)


def topk_decode(
    logits,
    next_n,
    seq_lens,
    indices,
    num_rows,
    stride0,
    stride1,
    top_k,
    *,
    max_seq_len=None,
):
    del max_seq_len
    rows = torch.arange(num_rows, device=logits.device)
    ends = seq_lens.reshape(-1)[rows // next_n] - next_n + rows % next_n + 1
    return _select(
        logits, torch.zeros_like(ends), ends, indices, num_rows, stride0, stride1, top_k
    )

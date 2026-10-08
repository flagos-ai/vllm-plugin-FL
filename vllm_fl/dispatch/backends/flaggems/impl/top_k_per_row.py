# SPDX-License-Identifier: Apache-2.0
"""FlagGems adapters for row-wise top-k capabilities."""

from __future__ import annotations

import torch


def _normalize_decode_lengths(
    seq_lens: torch.Tensor, num_rows: int, next_n: int
) -> tuple[torch.Tensor, int]:
    if next_n <= 0 or num_rows % next_n:
        raise ValueError("next_n must be positive and divide the number of rows")
    batch_size = num_rows // next_n
    if seq_lens.ndim == 1:
        if seq_lens.shape[0] != batch_size:
            raise ValueError("1D seq_lens must have one final length per request")
        return seq_lens.contiguous(), next_n
    if seq_lens.ndim == 2:
        if seq_lens.shape != (batch_size, next_n):
            raise ValueError("2D seq_lens must have shape [batch, next_n]")
        # The FlagGems ABI derives each row length from one final request
        # length. Use next_n=1 for explicit per-row lengths so callers are not
        # required to satisfy that stronger, provider-specific assumption.
        return seq_lens.reshape(-1).contiguous(), 1
    raise ValueError("seq_lens must be 1D or 2D")


def _validate_common(logits: torch.Tensor, indices: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("Row-wise top-k expects FP32 logits [rows, width]")
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("Row-wise top-k expects INT32 indices [rows, top_k]")
    if logits.shape[0] != indices.shape[0]:
        raise ValueError("logits and indices must have the same number of rows")
    if logits.device != indices.device:
        raise ValueError("logits and indices must share a device")


def top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Select request-relative top-k indices for prefill rows."""
    from flag_gems.fused import top_k_per_row_prefill as flaggems_topk

    _validate_common(logits, indices)
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        raise ValueError("row_starts and row_ends must be 1D")
    if row_starts.shape[0] != logits.shape[0] or row_ends.shape != row_starts.shape:
        raise ValueError("prefill ranges must contain one start/end per row")
    if row_starts.dtype != torch.int32 or row_ends.dtype != torch.int32:
        raise TypeError("prefill row ranges must be int32")
    if logits.device != row_starts.device or logits.device != row_ends.device:
        raise ValueError("logits, row ranges and indices must share a device")
    flaggems_topk(
        logits,
        row_starts.contiguous(),
        row_ends.contiguous(),
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        indices.shape[1],
    )


def top_k_per_row_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    *,
    next_n: int,
) -> None:
    """Select top-k indices within each decode row's valid token range."""
    from flag_gems.fused import top_k_per_row_decode as flaggems_topk

    _validate_common(logits, indices)
    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must be int32")
    if logits.device != seq_lens.device:
        raise ValueError("logits, seq_lens and indices must share a device")
    kernel_seq_lens, kernel_next_n = _normalize_decode_lengths(
        seq_lens, logits.shape[0], next_n
    )
    flaggems_topk(
        logits,
        kernel_next_n,
        kernel_seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        indices.shape[1],
    )


__all__ = ["top_k_per_row_decode", "top_k_per_row_prefill"]

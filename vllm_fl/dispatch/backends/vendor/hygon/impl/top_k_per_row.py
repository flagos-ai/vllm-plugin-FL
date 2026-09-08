# Copyright (c) 2026 BAAI. All rights reserved.

"""GLM-Hygon prefill and decode Top-K helpers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_outside_row_kernel(
    logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride_row,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    mask = offsets < width
    values = tl.load(
        logits_ptr + row * stride_row + offsets,
        mask=mask,
        other=-float("inf"),
    )
    values = tl.where(
        (offsets >= row_start) & (offsets < row_end),
        values,
        -float("inf"),
    )
    tl.store(
        logits_ptr + row * stride_row + offsets,
        values,
        mask=mask,
    )


@triton.jit
def _store_relative_indices_kernel(
    selected_ptr,
    output_ptr,
    row_starts_ptr,
    row_ends_ptr,
    selected_stride,
    output_stride,
    effective_k: tl.constexpr,
    topk_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    selected_mask = offsets < effective_k
    absolute_indices = tl.load(
        selected_ptr + row * selected_stride + offsets,
        mask=selected_mask,
        other=-1,
    )
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    valid = (
        selected_mask
        & (absolute_indices >= row_start)
        & (absolute_indices < row_end)
    )
    relative_indices = tl.where(
        valid,
        absolute_indices - row_start,
        -1,
    ).to(tl.int32)
    tl.store(
        output_ptr + row * output_stride + offsets,
        relative_indices,
        mask=offsets < topk_tokens,
    )


@triton.jit
def _fill_all_valid_relative_indices_kernel(
    output_ptr,
    row_starts_ptr,
    row_ends_ptr,
    output_stride,
    topk_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Emit every request-local index for a full-selection row."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    valid_length = tl.maximum(row_end - row_start, 0)
    values = tl.where(
        offsets < valid_length,
        offsets,
        -1,
    ).to(tl.int32)
    tl.store(
        output_ptr + row * output_stride + offsets,
        values,
        mask=offsets < topk_tokens,
    )


@triton.jit
def _fill_all_valid_decode_indices_kernel(
    output_ptr,
    seq_lens_ptr,
    output_stride,
    NEXT_N: tl.constexpr,
    SEQ_LENS_2D: tl.constexpr,
    topk_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Match vLLM's decode shortcut when a row has at most Top-K items."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    if SEQ_LENS_2D:
        valid_length = tl.load(seq_lens_ptr + row)
    else:
        batch_idx = row // NEXT_N
        next_n_idx = row % NEXT_N
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        valid_length = tl.maximum(
            seq_len - NEXT_N + next_n_idx + 1,
            0,
        )

    values = tl.where(
        offsets < valid_length,
        offsets,
        -1,
    ).to(tl.int32)
    tl.store(
        output_ptr + row * output_stride + offsets,
        values,
        mask=offsets < topk_tokens,
    )


@triton.jit
def _mask_outside_decode_row_kernel(
    logits_ptr,
    seq_lens_ptr,
    stride_row,
    NEXT_N: tl.constexpr,
    SEQ_LENS_2D: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Mask positions outside each decode row's request-local length."""

    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)

    if SEQ_LENS_2D:
        valid_length = tl.load(seq_lens_ptr + row)
    else:
        batch_idx = row // NEXT_N
        next_n_idx = row % NEXT_N
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        valid_length = tl.maximum(
            seq_len - NEXT_N + next_n_idx + 1,
            0,
        )

    in_width = offsets < width
    values = tl.load(
        logits_ptr + row * stride_row + offsets,
        mask=in_width,
        other=-float("inf"),
    )
    values = tl.where(
        offsets < valid_length,
        values,
        -float("inf"),
    )
    tl.store(
        logits_ptr + row * stride_row + offsets,
        values,
        mask=in_width,
    )


@triton.jit
def _store_decode_selected_indices_kernel(
    selected_ptr,
    output_ptr,
    seq_lens_ptr,
    selected_stride,
    output_stride,
    NEXT_N: tl.constexpr,
    SEQ_LENS_2D: tl.constexpr,
    effective_k: tl.constexpr,
    topk_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Store valid request-local decode indices and pad the rest with -1."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    if SEQ_LENS_2D:
        valid_length = tl.load(seq_lens_ptr + row)
    else:
        batch_idx = row // NEXT_N
        next_n_idx = row % NEXT_N
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        valid_length = tl.maximum(
            seq_len - NEXT_N + next_n_idx + 1,
            0,
        )

    selected_mask = offsets < effective_k
    selected = tl.load(
        selected_ptr + row * selected_stride + offsets,
        mask=selected_mask,
        other=-1,
    )
    valid = (
        selected_mask
        & (selected >= 0)
        & (selected < valid_length)
    )
    values = tl.where(valid, selected, -1).to(tl.int32)
    tl.store(
        output_ptr + row * output_stride + offsets,
        values,
        mask=offsets < topk_tokens,
    )


def glm_hygon_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """Select request-local positions and pad unused slots with ``-1``.

    The FlagGems implementation raises when ``topk_tokens > logits.shape[1]``.
    GLM commonly has K=2048 with fewer candidates during initial prefill. Select
    only
    ``min(K, width)`` candidates, keep valid entries first, convert them to
    request-local positions, and fill the unused output slots with -1.
    """

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor")
    if indices.ndim != 2:
        raise ValueError("indices must be a 2D tensor")
    if logits.dtype != torch.float32:
        raise TypeError("logits must be float32")
    if indices.dtype != torch.int32:
        raise TypeError("indices must be int32")
    if row_starts.dtype != torch.int32 or row_ends.dtype != torch.int32:
        raise TypeError("row starts/ends must be int32")
    if stride1 != 1 or logits.stride(1) != 1:
        raise ValueError("logits must be contiguous in its last dimension")
    if num_rows <= 0 or topk_tokens <= 0:
        return
    if logits.shape[0] < num_rows or indices.shape[0] < num_rows:
        raise ValueError("num_rows exceeds input/output rows")
    if indices.shape[1] < topk_tokens:
        raise ValueError("indices width is smaller than topk_tokens")

    width = logits.shape[1]
    effective_k = min(topk_tokens, width)
    if effective_k <= 0:
        indices[:num_rows, :topk_tokens].fill_(-1)
        return

    # This is a sufficient chunk-wide fast path for vLLM's per-row rule
    # ``row_end - row_start <= topK``: if the whole logits workspace fits in K,
    # every row fits as well. Every causal position must then be selected, so
    # emit request-local indices directly. Besides being cheaper, this avoids
    # FlagGems' radix TopK, whose small-width path currently fails HCU LLVM
    # translation on BW1000.
    if width <= topk_tokens:
        output_block = triton.next_power_of_2(topk_tokens)
        _fill_all_valid_relative_indices_kernel[(num_rows,)](
            indices,
            row_starts,
            row_ends,
            indices.stride(0),
            topk_tokens=topk_tokens,
            BLOCK=output_block,
            num_warps=8,
            num_stages=1,
        )
        return

    mask_block = 1024
    _mask_outside_row_kernel[
        (num_rows, triton.cdiv(width, mask_block))
    ](
        logits,
        row_starts,
        row_ends,
        stride0,
        width=width,
        BLOCK=mask_block,
        num_warps=4,
        num_stages=1,
    )

    # sorted=True is intentional: all finite valid scores precede the masked
    # -inf entries, so the output remains compact before its -1 padding.
    _, selected = torch.topk(
        logits[:num_rows],
        effective_k,
        dim=1,
        largest=True,
        sorted=True,
    )

    output_block = triton.next_power_of_2(topk_tokens)
    _store_relative_indices_kernel[(num_rows,)](
        selected,
        indices,
        row_starts,
        row_ends,
        selected.stride(0),
        indices.stride(0),
        effective_k=effective_k,
        topk_tokens=topk_tokens,
        BLOCK=output_block,
        num_warps=8,
        num_stages=1,
    )


def glm_hygon_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
    max_seq_len: int,
) -> None:
    """Select decode indices with vLLM-compatible short-row semantics."""

    if logits.ndim != 2 or indices.ndim != 2:
        raise ValueError("logits and indices must be 2D tensors")
    if seq_lens.ndim not in (1, 2):
        raise ValueError("seq_lens must be a 1D or 2D tensor")
    if logits.dtype != torch.float32:
        raise TypeError("logits must be float32")
    if indices.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise TypeError("indices and seq_lens must be int32")
    if stride1 != 1 or logits.stride(1) != 1:
        raise ValueError("logits must be contiguous in its last dimension")
    if num_rows <= 0 or topk_tokens <= 0:
        return
    if indices.shape[0] < num_rows or indices.shape[1] < topk_tokens:
        raise ValueError("indices is smaller than the requested output")

    # vLLM's CUDA topKPerRowJob directly emits all valid indices whenever
    # rowLen <= topK. max_seq_len is metadata, so this avoids a GPU-to-CPU sync
    # on every decode step while proving that the condition holds for all rows.
    if max_seq_len <= topk_tokens:
        output_block = triton.next_power_of_2(topk_tokens)
        _fill_all_valid_decode_indices_kernel[(num_rows,)](
            indices,
            seq_lens,
            indices.stride(0),
            NEXT_N=next_n,
            SEQ_LENS_2D=seq_lens.ndim == 2,
            topk_tokens=topk_tokens,
            BLOCK=output_block,
            num_warps=8,
            num_stages=1,
        )
        return

    # FlagGems' decode selector asserts num_rows == 1, which is not true for
    # vLLM's batched decode and CUDAGraph warmup. Use the same batched masking
    # and selection strategy as the Hygon prefill implementation instead.
    width = logits.shape[1]
    effective_k = min(topk_tokens, width)
    if effective_k <= 0:
        indices[:num_rows, :topk_tokens].fill_(-1)
        return

    mask_block = 1024
    _mask_outside_decode_row_kernel[
        (num_rows, triton.cdiv(width, mask_block))
    ](
        logits,
        seq_lens,
        stride0,
        NEXT_N=next_n,
        SEQ_LENS_2D=seq_lens.ndim == 2,
        width=width,
        BLOCK=mask_block,
        num_warps=4,
        num_stages=1,
    )

    _, selected = torch.topk(
        logits[:num_rows],
        effective_k,
        dim=1,
        largest=True,
        sorted=True,
    )

    output_block = triton.next_power_of_2(topk_tokens)
    _store_decode_selected_indices_kernel[(num_rows,)](
        selected,
        indices,
        seq_lens,
        selected.stride(0),
        indices.stride(0),
        NEXT_N=next_n,
        SEQ_LENS_2D=seq_lens.ndim == 2,
        effective_k=effective_k,
        topk_tokens=topk_tokens,
        BLOCK=output_block,
        num_warps=8,
        num_stages=1,
    )

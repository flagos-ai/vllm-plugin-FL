# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA metadata/selection and graph workspace ownership; numerical ops live in FlagGems-vllm."""

from __future__ import annotations

import torch
from flaggems_vllm import (
    qsa_mqa_paged,
    qsa_store_cache_rows,
    qsa_store_kv_cache_rows,
    qsa_compress_groups_with_ratio,
    qsa_compress_norm_mrope_store_groups,
    qsa_sparse_paged_attention as _qsa_sparse_paged_attention,
    qsa_sparse_split_count,
)
from vllm.triton_utils import HAS_TRITON, tl, triton

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024

def _is_triton_device(tensor: torch.Tensor) -> bool:
    """Treat every non-host accelerator supported by FlagTree as eligible."""

    return HAS_TRITON and tensor.device.type not in ("cpu", "meta")



def qsa_forward_metadata_triton_supported(device: torch.device) -> bool:
    """Return whether this process can launch the vendor-neutral Triton path."""

    return HAS_TRITON and device.type not in ("cpu", "meta")



@triton.jit
def _build_qsa_forward_metadata_kernel(
    token_to_req_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    common_slot_mapping_ptr,
    logical_positions_ptr,
    qsa_slot_mapping_ptr,
    token_to_req_stride,
    query_start_loc_stride,
    seq_lens_stride,
    block_table_req_stride,
    block_table_block_stride,
    common_slot_mapping_stride,
    logical_positions_stride,
    qsa_slot_mapping_stride,
    NUM_ROWS: tl.constexpr,
    NUM_REQS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    STORAGE_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build all replay-sensitive QSA metadata without host-visible state."""

    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_mask = rows < NUM_ROWS

    # query_start_loc is monotonic.  The request for a mapped token is the
    # rightmost request whose start offset is <= the token row.  Each loop load
    # is scalar and broadcast across the block, avoiding the repeat_interleave
    # ATen chain while also handling zero-length requests exactly.
    requests = tl.zeros((BLOCK,), dtype=tl.int32)
    for request_index in tl.static_range(0, NUM_REQS):
        request_start = tl.load(
            query_start_loc_ptr + request_index * query_start_loc_stride
        ).to(tl.int64)
        requests = tl.where(rows >= request_start, request_index, requests)

    total_query_tokens = tl.load(
        query_start_loc_ptr + NUM_REQS * query_start_loc_stride
    ).to(tl.int64)
    mapped = row_mask & (rows < total_query_tokens)
    stored_requests = tl.where(mapped, requests, 0)
    tl.store(
        token_to_req_ptr + rows * token_to_req_stride,
        stored_requests,
        mask=row_mask,
    )
    safe_requests = requests.to(tl.int64)

    query_starts = tl.load(
        query_start_loc_ptr + safe_requests * query_start_loc_stride,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    query_ends = tl.load(
        query_start_loc_ptr + (safe_requests + 1) * query_start_loc_stride,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    sequence_lengths = tl.load(
        seq_lens_ptr + safe_requests * seq_lens_stride,
        mask=row_mask,
        other=0,
    ).to(tl.int64)

    logical_positions = sequence_lengths - (query_ends - query_starts) + (
        rows - query_starts
    )
    stored_positions = tl.where(mapped, logical_positions, -1)
    tl.store(
        logical_positions_ptr + rows * logical_positions_stride,
        stored_positions,
        mask=row_mask,
    )

    common_slots = tl.load(
        common_slot_mapping_ptr + rows * common_slot_mapping_stride,
        mask=row_mask,
        other=-1,
    ).to(tl.int64)
    nonnegative_positions = tl.maximum(logical_positions, 0)
    compressed_positions = nonnegative_positions // COMPRESS_RATIO
    logical_blocks = compressed_positions // STORAGE_BLOCK_SIZE
    block_valid = (logical_blocks >= 0) & (logical_blocks < TABLE_WIDTH)
    safe_blocks = tl.maximum(0, tl.minimum(logical_blocks, TABLE_WIDTH - 1))
    physical_blocks = tl.load(
        block_table_ptr
        + safe_requests * block_table_req_stride
        + safe_blocks * block_table_block_stride,
        mask=mapped & block_valid,
        other=-1,
    ).to(tl.int64)

    complete_group = ((logical_positions + 1) % COMPRESS_RATIO) == 0
    slot_valid = (
        mapped
        & (common_slots >= 0)
        & (logical_positions >= 0)
        & complete_group
        & block_valid
        & (physical_blocks >= 0)
    )
    compressed_slots = (
        physical_blocks * STORAGE_BLOCK_SIZE
        + compressed_positions % STORAGE_BLOCK_SIZE
    )
    tl.store(
        qsa_slot_mapping_ptr + rows * qsa_slot_mapping_stride,
        tl.where(slot_valid, compressed_slots, -1),
        mask=row_mask,
    )



def build_qsa_forward_metadata(
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    common_slot_mapping: torch.Tensor,
    logical_positions: torch.Tensor,
    qsa_slot_mapping: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
) -> bool:
    """Try the graph-safe QSA metadata builder.

    ``token_to_req`` is an output, not an input: the same launch derives it
    directly from ``query_start_loc`` and then produces logical positions and
    compressed slots.  Returns ``False`` without touching the outputs when
    Triton is unavailable or the device is not supported, allowing callers to
    execute the pure-Torch reference path.  ``compress_ratio == 1`` deliberately
    never enters this kernel because that cache aliases vLLM's common slots.
    """

    if compress_ratio == 1 or not _is_triton_device(token_to_req):
        return False
    if storage_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA metadata block size and compression ratio must be positive")
    rows = token_to_req.numel()
    if (
        token_to_req.ndim != 1
        or query_start_loc.ndim != 1
        or seq_lens.ndim != 1
        or block_table.ndim != 2
        or common_slot_mapping.shape != (rows,)
        or logical_positions.shape != (rows,)
        or qsa_slot_mapping.shape != (rows,)
    ):
        raise ValueError("QSA metadata tensors have incompatible shapes")
    if query_start_loc.numel() != seq_lens.numel() + 1:
        raise ValueError("QSA query_start_loc must contain one terminal offset")
    if logical_positions.dtype != torch.int64 or qsa_slot_mapping.dtype != torch.int64:
        raise ValueError("QSA metadata outputs must use int64")
    if token_to_req.dtype != torch.int32:
        raise ValueError("QSA token_to_req output must use int32")
    tensors = (
        query_start_loc,
        seq_lens,
        block_table,
        common_slot_mapping,
        logical_positions,
        qsa_slot_mapping,
    )
    if any(tensor.device != token_to_req.device for tensor in tensors):
        raise ValueError("QSA metadata tensors must be on the same device")
    if rows == 0:
        return True
    if seq_lens.numel() == 0 or block_table.shape[1] == 0:
        return False

    block = 256
    _build_qsa_forward_metadata_kernel[(triton.cdiv(rows, block),)](
        token_to_req,
        query_start_loc,
        seq_lens,
        block_table,
        common_slot_mapping,
        logical_positions,
        qsa_slot_mapping,
        token_to_req.stride(0),
        query_start_loc.stride(0),
        seq_lens.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        common_slot_mapping.stride(0),
        logical_positions.stride(0),
        qsa_slot_mapping.stride(0),
        NUM_ROWS=rows,
        NUM_REQS=seq_lens.numel(),
        TABLE_WIDTH=block_table.shape[1],
        STORAGE_BLOCK_SIZE=storage_block_size,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=block,
        num_warps=4,
    )
    return True



@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )



def _qsa_get_split_workspace(
    q: torch.Tensor,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return fixed-shape worker workspace, or eager-only direct-test buffers."""

    shape_output = (q.shape[0], q.shape[1], num_splits, q.shape[2])
    shape_stats = (q.shape[0], q.shape[1], num_splits)
    # Never ask the global manager to grow while a graph is being captured.
    # Model-owned bucket caches pass an explicit workspace on this path.
    if q.device.type == "cuda":
        try:
            if torch.cuda.is_current_stream_capturing():
                return None
        except (AssertionError, RuntimeError):
            return None
    # Never allocate shape-dependent workspace during CUDA graph capture.  The
    # single-kernel fallback remains graph-safe if a worker did not reserve the
    # split bucket before capture.
    return (
        torch.empty(shape_output, dtype=torch.float32, device=q.device),
        torch.empty(shape_stats, dtype=torch.float32, device=q.device),
        torch.empty(shape_stats, dtype=torch.float32, device=q.device),
    )



def qsa_prepare_split_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    topk: int,
    workspace_cache: dict[
        tuple[int, int, int, int, int, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] | None = None,
) -> tuple[
    int,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    """Prepare one fixed bucket without resizing another graph's workspace.

    A QSA layer owns ``workspace_cache`` so graph captures for different row
    or TopK buckets retain distinct tensor addresses.  The helper is called on
    the eager warmup path; a cache miss during capture deliberately returns no
    workspace; the caller then uses the single kernel.
    """

    num_splits = qsa_sparse_split_count(q, k_cache, topk)
    if num_splits <= 1:
        return 1, None
    key = (
        q.shape[0],
        q.shape[1],
        k_cache.shape[2],
        q.shape[2],
        topk,
        num_splits,
    )
    if workspace_cache is not None and key in workspace_cache:
        return num_splits, workspace_cache[key]
    if q.device.type == "cuda":
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except (AssertionError, RuntimeError):
            return num_splits, None
        if capturing:
            return num_splits, None
    workspace = (
        torch.empty(
            (q.shape[0], q.shape[1], num_splits, q.shape[2]),
            dtype=torch.float32,
            device=q.device,
        ),
        torch.empty(
            (q.shape[0], q.shape[1], num_splits),
            dtype=torch.float32,
            device=q.device,
        ),
        torch.empty(
            (q.shape[0], q.shape[1], num_splits),
            dtype=torch.float32,
            device=q.device,
        ),
    )
    if workspace_cache is not None:
        workspace_cache[key] = workspace
    return num_splits, workspace



def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the incomplete causal tail."""

    if not _is_triton_device(block_indices):
        raise RuntimeError("QSA index expansion requires an accelerator Triton backend")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out



def _qsa_deterministic_block_topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
) -> torch.Tensor:
    """Select by (score descending, logical index ascending), then visit in order.

    Native cooperative/persistent TopK emits an unordered set. Above the QSA
    budget its order varies between identical launches, changing the BF16
    attention reduction. Ties at the cutoff can also change *membership*, so
    sorting that set alone is insufficient. Stable score sorting provides an
    exact tie break without perturbing scores; logical ordering then fixes the
    attention reduction order. All shapes are static and graph-capturable.
    """
    if logits.ndim != 2 or visible_blocks.shape != (logits.shape[0],):
        raise ValueError("QSA scores and visible-block counts have incompatible shapes")
    if block_topk <= 0:
        raise ValueError("QSA block top-k must be positive")
    rows, columns = logits.shape
    width = min(block_topk, columns)
    ranked = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :width]
    # Mask padding before canonicalizing: otherwise -1 would sort ahead of
    # real blocks, and expansion consumes only the first visible ranks.
    ranked = ranked.masked_fill(ranked >= visible_blocks[:, None], columns)
    ordered = ranked.sort(dim=-1).values
    result = torch.full((rows, block_topk), -1, dtype=torch.int32, device=logits.device)
    result[:, :width] = ordered.masked_fill(ordered == columns, -1).to(torch.int32)
    return result



def qsa_select_all_paged_tokens(
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_model_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select every visible block when the worker's context fits the budget.

    With at most ``token_topk`` tokens, score ranking cannot exclude any
    visible block. Canonical block order is just arange. Reuse the existing
    expansion kernel for causal tails, padding and invalid request rows.
    ``max_model_len`` is the immutable serving limit, never a capture-time
    observation of a short batch in a worker that can later serve long input.
    """
    if not 0 < max_model_len <= token_topk:
        raise ValueError("Selecting all QSA tokens requires a context within budget")
    if compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    blocks = torch.arange(
        token_topk // compress_ratio, dtype=torch.int32, device=query_positions.device
    ).expand(query_positions.numel(), -1)
    return expand_qsa_block_indices(
        blocks,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out,
    )



def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score, select, and expand QSA indices without host synchronization."""

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    columns = page_table.shape[1] * k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    # Budget scores plus stable-sort values/indices and temporary storage,
    # rather than only the FP32 logits (long mixed prefills can be large).
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 32, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged(
            q[row_slice],
            k_cache,
            page_table,
            token_to_req[row_slice],
            query_positions[row_slice],
            sequence_lengths,
            compress_ratio,
        )
        blocks = _qsa_deterministic_block_topk(logits, visible_blocks, block_topk)
        expand_qsa_block_indices(
            blocks,
            query_positions[row_slice],
            sequence_lengths,
            token_to_req[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out



def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    split_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Call the library with a model-owned or eager-only split workspace."""
    if split_workspace is None:
        splits = qsa_sparse_split_count(q, k_cache, logical_indices.shape[1])
        if splits > 1:
            split_workspace = _qsa_get_split_workspace(q, splits)
    return _qsa_sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, softmax_scale=softmax_scale, out=out, gate=gate, split_workspace=split_workspace)

__all__ = [
    "build_qsa_forward_metadata",
    "expand_qsa_block_indices",
    "qsa_compress_groups_with_ratio",
    "qsa_compress_norm_mrope_store_groups",
    "qsa_mqa_paged",
    "qsa_forward_metadata_triton_supported",
    "qsa_select_paged_tokens",
    "qsa_select_all_paged_tokens",
    "qsa_prepare_split_workspace",
    "qsa_sparse_split_count",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "qsa_store_kv_cache_rows",
]

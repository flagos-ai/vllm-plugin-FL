# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor Triton kernels for the Qwen3.8-Flash-Next QSA path."""

from __future__ import annotations

import math
import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

try:
    from vllm.v1.worker.workspace import current_workspace_manager
except ImportError:  # newer/alternate vLLM workspace APIs use local buffers
    current_workspace_manager = None

from ..nvidia_fast_paths import (
    has_native_topk,
    native_topk,
    qsa_sparse_triton_launch_config,
)

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
_QSA_MQA_DOT_ENABLED = os.environ.get("QWEN4_QSA_MQA_DOT", "1") != "0"
_QSA_SPLIT_ALLOWED = (1, 2, 4, 8, 16, 32)


def _qsa_split_required() -> bool:
    """Whether a selected split bucket must be present during capture.

    The normal worker path keeps a graph-safe single-kernel fallback when a
    bucket was not warmed before capture.  Validation and all-on serving can
    set this gate to prove that the captured graph really contains the split
    and merge launches instead of silently measuring the fallback.
    """

    return os.environ.get("QWEN4_QSA_SPLIT_REQUIRE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    # physical_page * block stride can overflow int32 for large caches.
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


@triton.jit
def _qsa_mqa_paged_dot_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    """Tensor-core/MFMA QSA score path expressed with portable ``tl.dot``."""

    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[:, None] * stride_q_head
        + dims[None, :] * stride_q_dim,
        mask=(heads[:, None] < NUM_HEADS) & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    keys = tl.load(
        k_cache_ptr
        + safe_page[None, :] * stride_cache_block
        + page_offset[None, :] * stride_cache_token
        + dims[:, None] * stride_cache_dim,
        mask=(dims[:, None] < HEAD_DIM) & valid[None, :],
        other=0.0,
    )
    dots = tl.dot(query, keys)
    score = tl.sum(tl.maximum(dots, 0.0), axis=0) / score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


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


@triton.jit
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    gate_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_row,
    stride_indices_column,
    stride_table_req,
    stride_table_page,
    stride_gate_row,
    stride_gate_head,
    stride_gate_dim,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    num_cache_blocks,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_GATE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(token_to_req_ptr + row)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    for start in tl.range(0, TOPK, BLOCK_N):
        columns = start + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        logical_page = tl.maximum(logical_token, 0) // PAGE_SIZE
        page_offset = tl.maximum(logical_token, 0) % PAGE_SIZE
        valid = (
            (row < num_rows)
            & (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    output = tl.where(
        normalizer[:, None] > 0,
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    if APPLY_GATE:
        gate = tl.load(
            gate_ptr
            + row * stride_gate_row
            + (first_head + head_offsets[:, None]) * stride_gate_head
            + dim_offsets[None, :] * stride_gate_dim,
            mask=(row < num_rows)
            & (head_offsets[:, None] < GROUP_SIZE)
            & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        # Preserve the unfused BF16 rounding points: attention is stored as
        # BF16, sigmoid returns BF16 for a BF16 gate, then the product rounds
        # to BF16 before the original output buffer is consumed by o_proj.
        output = (
            output.to(tl.bfloat16)
            * tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
        ).to(tl.bfloat16)
    tl.store(
        output_ptr
        + row * stride_output_row
        + (first_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(row < num_rows)
        & (head_offsets[:, None] < GROUP_SIZE)
        & (dim_offsets[None, :] < HEAD_DIM),
    )


@triton.jit
def _qsa_sparse_paged_gqa_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_row,
    stride_indices_column,
    stride_table_req,
    stride_table_page,
    stride_partial_output_row,
    stride_partial_output_head,
    stride_partial_output_split,
    stride_partial_output_dim,
    stride_partial_stat_row,
    stride_partial_stat_head,
    stride_partial_stat_split,
    num_rows,
    num_cache_blocks,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    SPLIT_TOPK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Compute one contiguous TopK split into FP32 partials.

    The split axis is deliberately part of the launch grid, so every program
    locates its own token range from ``program_id(2)``.  No host-side index
    expansion or reduction metadata is needed during graph replay.
    """

    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE)
        & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    # Preserve the single-kernel path's BF16 query scaling/rounding point.
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)
    split_start = split * SPLIT_TOPK

    for start in tl.range(0, SPLIT_TOPK, BLOCK_N):
        columns = split_start + start + column_offsets
        valid_column = (columns < TOPK) & (start + column_offsets < SPLIT_TOPK)
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns * stride_indices_column,
            mask=valid_column,
            other=-1,
        )
        logical_page = tl.maximum(logical_token, 0) // PAGE_SIZE
        page_offset = tl.maximum(logical_token, 0) % PAGE_SIZE
        valid = (
            (row < num_rows)
            & (request >= 0)
            & (request < num_requests)
            & valid_column
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
        safe_logical_page = tl.minimum(
            logical_page, PAGE_TABLE_WIDTH - 1
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + safe_logical_page * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    query_heads = first_head + head_offsets
    stat_offsets = (
        row * stride_partial_stat_row
        + query_heads * stride_partial_stat_head
        + split * stride_partial_stat_split
    )
    stat_mask = (row < num_rows) & (head_offsets < GROUP_SIZE)
    tl.store(partial_max_ptr + stat_offsets, max_value, mask=stat_mask)
    tl.store(partial_sum_ptr + stat_offsets, normalizer, mask=stat_mask)
    output_offsets = (
        row * stride_partial_output_row
        + query_heads[:, None] * stride_partial_output_head
        + split * stride_partial_output_split
        + dim_offsets[None, :] * stride_partial_output_dim
    )
    tl.store(
        partial_output_ptr + output_offsets,
        accumulator,
        mask=stat_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM),
    )


@triton.jit
def _qsa_sparse_paged_gqa_split_reduce_kernel(
    partial_output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    gate_ptr,
    output_ptr,
    stride_partial_output_row,
    stride_partial_output_head,
    stride_partial_output_split,
    stride_partial_output_dim,
    stride_partial_stat_row,
    stride_partial_stat_head,
    stride_partial_stat_split,
    stride_gate_row,
    stride_gate_head,
    stride_gate_dim,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_GATE: tl.constexpr,
) -> None:
    """Stable FP32 merge and BF16 output-gate epilogue."""

    row = tl.program_id(0)
    query_head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_S)
    dim_offsets = tl.arange(0, BLOCK_D)
    split_mask = split_offsets < NUM_SPLITS
    stat_offsets = (
        row * stride_partial_stat_row
        + query_head * stride_partial_stat_head
        + split_offsets * stride_partial_stat_split
    )
    partial_max = tl.load(
        partial_max_ptr + stat_offsets, mask=split_mask, other=-1.0e20
    )
    partial_sum = tl.load(partial_sum_ptr + stat_offsets, mask=split_mask, other=0.0)
    global_max = tl.max(partial_max, axis=0)
    split_scale = tl.math.exp2(partial_max - global_max)
    denominator = tl.sum(partial_sum * split_scale, axis=0)
    partial_offsets = (
        row * stride_partial_output_row
        + query_head * stride_partial_output_head
        + split_offsets[:, None] * stride_partial_output_split
        + dim_offsets[None, :] * stride_partial_output_dim
    )
    partial_output = tl.load(
        partial_output_ptr + partial_offsets,
        mask=split_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    numerator = tl.sum(partial_output * split_scale[:, None], axis=0)
    output = tl.where(denominator > 0, numerator / denominator, 0.0)
    if APPLY_GATE:
        gate = tl.load(
            gate_ptr
            + row * stride_gate_row
            + query_head * stride_gate_head
            + dim_offsets * stride_gate_dim,
            mask=(row < num_rows)
            & (query_head < NUM_QUERY_HEADS)
            & (dim_offsets < HEAD_DIM),
            other=0.0,
        )
        # Match the existing single-kernel observable BF16 rounding points.
        output = (
            output.to(tl.bfloat16)
            * tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
        ).to(tl.bfloat16)
    tl.store(
        output_ptr
        + row * stride_output_row
        + query_head * stride_output_head
        + dim_offsets * stride_output_dim,
        output,
        mask=(row < num_rows)
        & (query_head < NUM_QUERY_HEADS)
        & (dim_offsets < HEAD_DIM),
    )


def _qsa_sparse_split_count(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    topk: int,
) -> int:
    """Select a graph-static split count for the measured H100 c64 shape.

    The default is deliberately narrow: only SM90 ``[rows, 3, 256]`` over a
    single KV head and a sufficiently wide selection use split=8.  Shape or
    architecture misses stay on the proven single CTA path.  An environment
    override is useful for the short validation matrix without changing model
    code; ``1`` disables the split path.
    """

    try:
        requested = int(os.environ.get("QWEN4_QSA_SPLIT_TOPK", "8"))
    except ValueError:
        requested = 1
    if requested not in _QSA_SPLIT_ALLOWED or requested == 1:
        return 1
    if q.device.type != "cuda" or q.shape[0] <= 0:
        return 1
    if q.shape[1] != 3 or k_cache.shape[2] != 1 or q.shape[2] != 256:
        return 1
    if q.shape[0] > 64 or topk < 512:
        return 1
    try:
        # The measured H100 selector is the only selector currently approved
        # for split dispatch.  Avoid introducing a second architecture ABI.
        block_n, num_warps, num_stages = qsa_sparse_triton_launch_config(q.device)
        if (block_n, num_warps, num_stages) != (64, 8, 3):
            return 1
    except (AssertionError, RuntimeError):
        return 1
    num_tiles = triton.cdiv(topk, 64)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    return min(requested, max_useful_splits)


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
    if current_workspace_manager is not None:
        try:
            return current_workspace_manager().get_simultaneous(
                (shape_output, torch.float32),
                (shape_stats, torch.float32),
                (shape_stats, torch.float32),
            )
        except (AssertionError, RuntimeError, TypeError):
            pass
    # Never allocate shape-dependent workspace during CUDA graph capture.  The
    # single-kernel fallback remains graph-safe if a worker did not reserve the
    # split bucket before capture.
    return (
        torch.empty(shape_output, dtype=torch.float32, device=q.device),
        torch.empty(shape_stats, dtype=torch.float32, device=q.device),
        torch.empty(shape_stats, dtype=torch.float32, device=q.device),
    )


def qsa_sparse_split_count(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    topk: int,
) -> int:
    """Public shape selector used by the model-owned workspace cache."""

    return _qsa_sparse_split_count(q, k_cache, topk)


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
    workspace and lets the caller use the single-kernel fallback unless
    ``QWEN4_QSA_SPLIT_REQUIRE=1`` is set for all-on validation.
    """

    num_splits = _qsa_sparse_split_count(q, k_cache, topk)
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
        except (AssertionError, RuntimeError) as exc:
            if _qsa_split_required():
                raise RuntimeError(
                    "QSA could not determine CUDA graph capture state while "
                    "QWEN4_QSA_SPLIT_REQUIRE is enabled"
                ) from exc
            return num_splits, None
        if capturing:
            if _qsa_split_required():
                raise RuntimeError(
                    "QSA split workspace was not warmed before CUDA graph "
                    "capture; disable QWEN4_QSA_SPLIT_REQUIRE or warm the "
                    "exact rows/TopK bucket first"
                )
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


def _qsa_sparse_paged_attention_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    partial_output: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    *,
    num_splits: int,
    softmax_scale: float | None,
    out: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    """Run graph-safe split-TopK sparse attention with caller-owned partials."""

    if num_splits not in _QSA_SPLIT_ALLOWED[1:]:
        raise ValueError("QSA split count must be a power of two from 2 through 32")
    expected_output = (q.shape[0], q.shape[1], num_splits, q.shape[2])
    expected_stats = (q.shape[0], q.shape[1], num_splits)
    if partial_output.shape != expected_output or partial_output.dtype != torch.float32:
        raise ValueError("split QSA partial output workspace has an invalid layout")
    if partial_max.shape != expected_stats or partial_max.dtype != torch.float32:
        raise ValueError("split QSA partial max workspace has an invalid layout")
    if partial_sum.shape != expected_stats or partial_sum.dtype != torch.float32:
        raise ValueError("split QSA partial sum workspace has an invalid layout")
    if out.shape != q.shape:
        raise ValueError("split QSA output must match its query")
    if partial_output.device != q.device or partial_max.device != q.device:
        raise ValueError("split QSA workspace must share the query device")
    if partial_sum.device != q.device:
        raise ValueError("split QSA workspace must share the query device")

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = max(8, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    block_n, num_warps, num_stages = qsa_sparse_triton_launch_config(q.device)
    split_topk = triton.cdiv(logical_indices.shape[1], num_splits)
    _qsa_sparse_paged_gqa_split_kernel[
        (q.shape[0], k_cache.shape[2], num_splits)
    ](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_max,
        partial_sum,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        SPLIT_TOPK=split_topk,
        NUM_SPLITS=num_splits,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _qsa_sparse_paged_gqa_split_reduce_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_max,
        partial_sum,
        gate if gate is not None else out,
        out,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        gate.stride(2) if gate is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        HEAD_DIM=q.shape[2],
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=block_d,
        APPLY_GATE=gate is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _store_qsa_kv_rows_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    k_rows_ptr,
    v_rows_ptr,
    stride_k_cache_block,
    stride_k_cache_token,
    stride_k_cache_head,
    stride_k_cache_dim,
    stride_v_cache_block,
    stride_v_cache_token,
    stride_v_cache_head,
    stride_v_cache_dim,
    stride_k_rows_row,
    stride_k_rows_head,
    stride_k_rows_dim,
    stride_v_rows_row,
    stride_v_rows_head,
    stride_v_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Store K and V together while preserving arbitrary cache strides."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    k_values = tl.load(
        k_rows_ptr
        + row * stride_k_rows_row
        + head * stride_k_rows_head
        + dims * stride_k_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    v_values = tl.load(
        v_rows_ptr
        + row * stride_v_rows_row
        + head * stride_v_rows_head
        + dims * stride_v_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    tl.store(
        k_cache_ptr
        + block * stride_k_cache_block
        + token * stride_k_cache_token
        + head * stride_k_cache_head
        + dims * stride_k_cache_dim,
        k_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )
    tl.store(
        v_cache_ptr
        + block * stride_v_cache_block
        + token * stride_v_cache_token
        + head * stride_v_cache_head
        + dims * stride_v_cache_dim,
        v_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    rope_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_raw_table_req,
    stride_raw_table_page,
    stride_rope_table_req,
    stride_rope_table_page,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_raw_requests,
    num_rope_requests,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    ROPE_TABLE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_raw_requests)
        & (request < num_rope_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if valid_row:
        for group_offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - group_offset)
            logical_page = position // RAW_PAGE_SIZE
            page_offset = position % RAW_PAGE_SIZE
            valid = logical_page < RAW_TABLE_WIDTH
            physical_page = tl.load(
                raw_table_ptr
                + request * stride_raw_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1) * stride_raw_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            # physical_page * block stride can overflow int32 for large caches.
            values = tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            accumulator += values

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        rope_logical_page = first_position // ROPE_PAGE_SIZE
        rope_page_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_logical_page < ROPE_TABLE_WIDTH)
        rope_physical_page = tl.load(
            rope_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_rope_requests - 1)
            * stride_rope_table_req
            + tl.minimum(rope_logical_page, ROPE_TABLE_WIDTH - 1)
            * stride_rope_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (rope_physical_page < num_rope_blocks)
        rope_values = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64) * stride_rope_block
            + rope_page_offset * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_rope & (position_dims < 3),
            other=0,
        )
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            rope_values,
            mask=(row < num_rows) & (position_dims < 3),
        )
    else:
        first_position = tl.where(valid_row, first_position, 0)
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            first_position,
            mask=(row < num_rows) & (position_dims < 3),
        )


@triton.jit
def _compress_norm_mrope_store_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    norm_weight_ptr,
    cos_sin_cache_ptr,
    compressed_cache_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_table_req,
    stride_table_page,
    stride_cos_row,
    stride_cos_dim,
    stride_compressed_block,
    stride_compressed_token,
    stride_compressed_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_compressed_blocks,
    num_requests,
    num_cos_rows,
    norm_eps,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MROPE_SECTION_T: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    LOAD_MROPE_POSITIONS: tl.constexpr,
) -> None:
    """Pool, Gemma-normalize, MRoPE, and store one compressed QSA key."""

    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
        & (compressed_slot < num_compressed_blocks * COMPRESSED_PAGE_SIZE)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if valid_row:
        for group_offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - group_offset)
            logical_page = position // RAW_PAGE_SIZE
            page_offset = position % RAW_PAGE_SIZE
            valid = logical_page < RAW_TABLE_WIDTH
            physical_page = tl.load(
                raw_table_ptr
                + request * stride_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1) * stride_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            accumulator += tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)

    # Match the unfused BF16 materialization before Gemma RMSNorm.
    pooled = (accumulator / COMPRESS_RATIO).to(tl.bfloat16)
    pooled_fp32 = pooled.to(tl.float32)
    variance = tl.sum(pooled_fp32 * pooled_fp32, axis=0) / HEAD_DIM
    weight = tl.load(
        norm_weight_ptr + dims,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    normalized = (
        pooled_fp32 * tl.rsqrt(variance + norm_eps) * (weight + 1.0)
    ).to(tl.bfloat16)

    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_MROPE_POSITIONS:
        rope_page = first_position // ROPE_PAGE_SIZE
        rope_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_page < RAW_TABLE_WIDTH)
        rope_physical_page = tl.load(
            raw_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1)
            * stride_table_req
            + tl.minimum(rope_page, RAW_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (
            rope_physical_page < num_rope_blocks
        )
        axis_offsets = tl.arange(0, 4)
        axis_positions = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64) * stride_rope_block
            + rope_offset * stride_rope_token
            + axis_offsets * stride_rope_dim,
            mask=valid_rope & (axis_offsets < 3),
            other=0,
        )
        time_position = tl.max(tl.where(axis_offsets == 0, axis_positions, 0))
        height_position = tl.max(tl.where(axis_offsets == 1, axis_positions, 0))
        width_position = tl.max(tl.where(axis_offsets == 2, axis_positions, 0))
    else:
        time_position = first_position
        height_position = first_position
        width_position = first_position

    # HEAD_DIM=128 and ROTARY_DIM=64 in the production checkpoint.  Split the
    # normalized vector without dynamic local indexing: first head half holds
    # all rotary channels, second head half is the pass-through tail.
    head_pairs = tl.permute(tl.reshape(normalized, (2, BLOCK_D // 2)), (1, 0))
    rotary_values, pass_values = tl.split(head_pairs)
    rotary_pairs = tl.permute(
        tl.reshape(rotary_values, (2, ROTARY_DIM // 2)), (1, 0)
    )
    first_half, second_half = tl.split(rotary_pairs)
    frequencies = tl.arange(0, ROTARY_DIM // 2)
    if MROPE_INTERLEAVED:
        use_height = ((frequencies % 3) == 1) & (
            frequencies < 3 * MROPE_SECTION_H
        )
        use_width = ((frequencies % 3) == 2) & (
            frequencies < 3 * MROPE_SECTION_W
        )
    else:
        height_start = MROPE_SECTION_T
        width_start = height_start + MROPE_SECTION_H
        use_height = (frequencies >= height_start) & (frequencies < width_start)
        use_width = (frequencies >= width_start) & (
            frequencies < width_start + MROPE_SECTION_W
        )
    rope_positions = tl.where(
        use_height,
        height_position,
        tl.where(use_width, width_position, time_position),
    )
    valid_position = (rope_positions >= 0) & (rope_positions < num_cos_rows)
    safe_positions = tl.minimum(tl.maximum(rope_positions, 0), num_cos_rows - 1)
    cos = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + frequencies * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + (ROTARY_DIM // 2 + frequencies) * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    rotated_first = (first_half * cos - second_half * sin).to(tl.bfloat16)
    rotated_second = (second_half * cos + first_half * sin).to(tl.bfloat16)

    compressed_block = tl.maximum(compressed_slot, 0) // COMPRESSED_PAGE_SIZE
    compressed_token = tl.maximum(compressed_slot, 0) % COMPRESSED_PAGE_SIZE
    compressed_base = (
        compressed_cache_ptr
        + compressed_block.to(tl.int64) * stride_compressed_block
        + compressed_token * stride_compressed_token
    )
    tl.store(
        compressed_base + frequencies * stride_compressed_dim,
        rotated_first,
        mask=valid_row & valid_position,
    )
    tl.store(
        compressed_base
        + (ROTARY_DIM // 2 + frequencies) * stride_compressed_dim,
        rotated_second,
        mask=valid_row & valid_position,
    )
    pass_offsets = tl.arange(0, BLOCK_D // 2)
    tl.store(
        compressed_base + (ROTARY_DIM + pass_offsets) * stride_compressed_dim,
        pass_values,
        mask=valid_row & ((ROTARY_DIM + pass_offsets) < HEAD_DIM),
    )
def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def _use_qsa_mqa_dot(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Gate the portable dot path where it wins without hurting single-row decode."""

    return (
        _QSA_MQA_DOT_ENABLED
        and q.shape[0] >= 8
        and q.shape[1:] == (4, 128)
        and q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
    )


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    _validate_mqa(q)
    if not _is_triton_device(q):
        raise RuntimeError("paged QSA scoring requires an accelerator Triton backend")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    block_n = 32
    use_dot = _use_qsa_mqa_dot(q, k_cache)
    kernel = _qsa_mqa_paged_dot_kernel if use_dot else _qsa_mqa_paged_kernel
    dot_kwargs = {"BLOCK_H": 16, "num_stages": 2} if use_dot else {}
    kernel[(q.shape[0], triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
        **dot_kwargs,
    )
    return logits, visible_blocks


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
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    chunk_rows = min(rows, rows_per_chunk)
    use_native_topk = has_native_topk() and block_topk in (512, 1024, 2048)
    blocks_buffer: torch.Tensor | None = None
    topk_workspace: torch.Tensor | None = None
    if use_native_topk:
        if current_workspace_manager is not None:
            try:
                blocks_buffer, topk_workspace = (
                    current_workspace_manager().get_simultaneous(
                        ((chunk_rows, block_topk), torch.int32),
                        ((_TOPK_WORKSPACE_BYTES,), torch.uint8),
                    )
                )
            except (AssertionError, RuntimeError):
                # Direct operator tests do not install a worker workspace.
                # Fixed-shape allocations remain graph-capturable.
                blocks_buffer = None
        if blocks_buffer is None or topk_workspace is None:
            blocks_buffer = torch.empty(
                (chunk_rows, block_topk), dtype=torch.int32, device=q.device
            )
            topk_workspace = torch.empty(
                (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
            )
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
        if use_native_topk:
            assert blocks_buffer is not None and topk_workspace is not None
            blocks = blocks_buffer[: row_end - row_start]
            native_topk(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                columns,
            )
        else:
            # Cross-vendor dispatcher entry point: FlagGems or the vendor
            # runtime can provide TopK. Sorting keeps finite visible blocks
            # ahead of the -inf padding consumed by expansion.
            _, blocks = torch.topk(
                logits,
                block_topk,
                dim=-1,
                largest=True,
                sorted=True,
            )
            del visible_blocks
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
    """Run sparse GQA directly over paged BF16 K/V caches.

    ``split_workspace`` is optional so existing callers keep the ABI.  A
    worker-provided workspace is preferred; direct eager tests may omit it and
    receive a temporary buffer, while CUDA graph capture falls back to the
    single kernel if no fixed bucket was reserved.
    """

    if not _is_triton_device(q):
        raise RuntimeError(
            "paged QSA sparse attention requires an accelerator Triton backend"
        )
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    if scale <= 0:
        raise ValueError("QSA softmax scale must be positive")
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    if gate is not None and gate.shape != q.shape:
        raise ValueError("QSA output gate must match its query")
    if gate is not None and gate.dtype != q.dtype:
        raise ValueError("QSA output gate must match its query dtype")
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    # Triton accepts an 8-row dot tile on the supported accelerator backend.
    # TP8 executes three Q heads per replicated KV head, so a minimum of 16
    # spends most of the M tile on padding.  Eight retains Triton's matrix-dot
    # lowering while reducing that waste and is neutral to the device vendor.
    block_m = max(8, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    block_n, num_warps, num_stages = qsa_sparse_triton_launch_config(q.device)
    num_splits = _qsa_sparse_split_count(q, k_cache, logical_indices.shape[1])
    if num_splits > 1:
        if split_workspace is None:
            split_workspace = _qsa_get_split_workspace(q, num_splits)
            if split_workspace is None and _qsa_split_required():
                raise RuntimeError(
                    "QSA split workspace is unavailable for the selected shape; "
                    "warm the exact bucket before CUDA graph capture"
                )
        if split_workspace is not None:
            return _qsa_sparse_paged_attention_split(
                q,
                k_cache,
                v_cache,
                logical_indices,
                block_table,
                token_to_req,
                *split_workspace,
                num_splits=num_splits,
                softmax_scale=scale,
                out=out,
                gate=gate,
            )
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], k_cache.shape[2])](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        gate if gate is not None else out,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        gate.stride(2) if gate is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        APPLY_GATE=gate is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not _is_triton_device(cache):
        raise RuntimeError("QSA cache stores require an accelerator Triton backend")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_store_kv_cache_rows(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Store paired K/V rows with one cross-vendor Triton launch."""

    if not _is_triton_device(k_cache):
        raise RuntimeError("QSA K/V cache stores require an accelerator Triton backend")
    if (
        k_cache.ndim != 4
        or v_cache.shape != k_cache.shape
        or key.ndim != 3
        or value.shape != key.shape
    ):
        raise ValueError("QSA K/V cache store received invalid tensor shapes")
    if key.shape != (slot_mapping.numel(), k_cache.shape[2], k_cache.shape[3]):
        raise ValueError("QSA K/V rows and cache geometry are incompatible")
    if key.dtype != k_cache.dtype or value.dtype != v_cache.dtype:
        raise ValueError("QSA K/V rows and caches must have matching dtypes")
    if not all(k_cache.shape):
        raise ValueError("QSA K/V cache dimensions must be nonzero")
    if not key.shape[0]:
        return
    _store_qsa_kv_rows_kernel[(key.shape[0], key.shape[1])](
        k_cache,
        v_cache,
        slot_mapping,
        key,
        value,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key.shape[0],
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=key.shape[1],
        HEAD_DIM=key.shape[2],
        BLOCK_D=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool raw-key groups and load their packed or derived positions."""

    if not _is_triton_device(raw_cache):
        raise RuntimeError("QSA compression requires an accelerator Triton backend")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if raw_cache.ndim != 4 or raw_cache.shape[2] != 1:
        raise ValueError("QSA raw cache has an invalid shape")
    if raw_block_table.ndim != 2:
        raise ValueError("QSA raw compression block table must be rank two")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (not all(raw_cache.shape) or not all(raw_block_table.shape)):
        raise ValueError("QSA raw cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_cache.shape[3]), dtype=raw_cache.dtype, device=raw_cache.device
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_cache.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = raw_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        raw_block_table.shape[0],
        raw_block_table.shape[0],
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        ROPE_TABLE_WIDTH=raw_block_table.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_cache.shape[3],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_cache.shape[3]),
        num_warps=4,
    )
    return pooled, first_positions


def qsa_compress_norm_mrope_store_groups(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compressed_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
    norm_eps: float,
    rotary_dim: int,
    mrope_section: tuple[int, int, int],
    mrope_interleaved: bool,
    rope_cache: torch.Tensor | None = None,
) -> None:
    """Fuse QSA compression, Gemma norm, Neox MRoPE, and cache insert."""

    if not _is_triton_device(raw_cache):
        raise RuntimeError("fused QSA compression requires accelerator Triton")
    rows = token_to_req.numel()
    if raw_cache.ndim != 4 or raw_cache.shape[2] != 1 or not all(raw_cache.shape):
        raise ValueError("fused QSA compression received an invalid raw cache")
    head_dim = raw_cache.shape[3]
    if head_dim != 128 or rotary_dim != 64:
        raise ValueError("fused QSA compression requires head_dim=128, rotary_dim=64")
    if (
        compressed_cache.ndim != 4
        or compressed_cache.shape[2:] != (1, head_dim)
        or not all(compressed_cache.shape)
        or compressed_cache.device != raw_cache.device
        or compressed_cache.dtype != raw_cache.dtype
    ):
        raise ValueError("fused QSA compressed cache must match the raw key width")
    if raw_block_table.ndim != 2 or not all(raw_block_table.shape):
        raise ValueError("fused QSA compression requires a nonempty block table")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("fused QSA compression metadata must match token rows")
    if (
        norm_weight.shape != (head_dim,)
        or norm_weight.dtype != raw_cache.dtype
        or norm_weight.device != raw_cache.device
        or norm_weight.stride(0) != 1
    ):
        raise ValueError("fused QSA compression norm weight is incompatible")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or not cos_sin_cache.shape[0]
        or cos_sin_cache.device != raw_cache.device
        or cos_sin_cache.dtype != raw_cache.dtype
    ):
        raise ValueError("fused QSA compression received an invalid RoPE cache")
    if sum(mrope_section) != rotary_dim // 2:
        raise ValueError("fused QSA MRoPE sections must cover half the rotary dim")
    if compress_ratio <= 0 or norm_eps <= 0:
        raise ValueError("fused QSA compression ratio and epsilon must be positive")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
        or rope_cache.device != raw_cache.device
    ):
        raise ValueError("fused QSA packed MRoPE cache is invalid")
    if not rows:
        return
    if rope_cache is None:
        rope_cache = raw_cache
        load_mrope_positions = False
    else:
        load_mrope_positions = True
    _compress_norm_mrope_store_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        norm_weight,
        cos_sin_cache,
        compressed_cache,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        compressed_cache.stride(3),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        compressed_cache.shape[0],
        raw_block_table.shape[0],
        cos_sin_cache.shape[0],
        float(norm_eps),
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        COMPRESSED_PAGE_SIZE=compressed_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        MROPE_SECTION_T=mrope_section[0],
        MROPE_SECTION_H=mrope_section[1],
        MROPE_SECTION_W=mrope_section[2],
        MROPE_INTERLEAVED=mrope_interleaved,
        LOAD_MROPE_POSITIONS=load_mrope_positions,
        num_warps=4,
    )


__all__ = [
    "build_qsa_forward_metadata",
    "expand_qsa_block_indices",
    "qsa_compress_groups_with_ratio",
    "qsa_compress_norm_mrope_store_groups",
    "qsa_mqa_paged",
    "qsa_forward_metadata_triton_supported",
    "qsa_select_paged_tokens",
    "qsa_prepare_split_workspace",
    "qsa_sparse_split_count",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "qsa_store_kv_cache_rows",
]

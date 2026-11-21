# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""V4.1 request/cache addresses only; no KV values or score selection (plugin:A3)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["valid_rows"])
def _pad_metadata_kernel(
    SLOTS,
    POSITIONS,
    LENGTHS,
    REQUESTS,
    SWA,
    TABLES,
    valid_rows,
    SWA_WIDTH: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row < valid_rows:
        return
    tl.store(SLOTS + row, -1)
    tl.store(POSITIONS + row, 0)
    tl.store(LENGTHS + row, 0)
    tl.store(REQUESTS + row, -1)
    columns = tl.arange(0, BLOCK)
    if SWA_WIDTH:
        tl.store(SWA + row * SWA_WIDTH + columns, -1, columns < SWA_WIDTH)
    if TABLE_WIDTH:
        tl.store(TABLES + row * TABLE_WIDTH + columns, -1, columns < TABLE_WIDTH)


def pad_cache_metadata(slots, positions, lengths, requests, swa, tables, valid_rows):
    """Initialize only unused caller-owned address rows with one fixed-shape launch.

    Always launch, even with no padding, so every rank loads this metadata
    kernel during startup rather than at a new per-DP request length.
    """
    swa_width = 0 if swa is None else swa.shape[-1]
    table_width = 0 if tables is None else tables.shape[-1]
    _pad_metadata_kernel[(slots.numel(),)](
        slots,
        positions,
        lengths,
        requests,
        swa,
        tables,
        valid_rows,
        swa_width,
        table_width,
        triton.next_power_of_2(max(swa_width, table_width, 1)),
        num_warps=1,
        num_stages=1,
    )


@triton.jit
def _cache_metadata_kernel(
    POS,
    QUERY,
    SEQ,
    REQUESTS,
    TABLE,
    ORIGINAL_SLOTS,
    SLOTS,
    GROUP_POS,
    LENGTHS,
    SWA,
    TOKEN_TABLES,
    ACTUAL,
    NREQS,
    NBLOCKS,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COLUMNS: tl.constexpr,
    SWA_STRIDE: tl.constexpr,
    TOKEN_TABLE_STRIDE: tl.constexpr,
    LOGICAL_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_POS: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_TABLES: tl.constexpr,
    TABLE_TILE: tl.constexpr,
    CAUSAL: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    SWA_TILE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    request = tl.load(REQUESTS + token).to(tl.int64)
    raw_slot = tl.load(ORIGINAL_SLOTS + token)
    valid = (token < ACTUAL) & (raw_slot >= 0) & (request >= 0) & (request < NREQS)
    begin = tl.load(QUERY + request, valid, other=0).to(tl.int64)
    end = tl.load(QUERY + request + 1, valid, other=0).to(tl.int64)
    valid &= (token >= begin) & (token < end)
    if HAS_POS:
        position = tl.load(POS + token).to(tl.int64)
    else:
        seq_len = tl.load(SEQ + request, valid, other=0).to(tl.int64)
        position = seq_len - (end - begin) + token - begin
    valid &= (position >= 0) & (position < TABLE_COLUMNS * LOGICAL_BLOCK)
    column = position // LOGICAL_BLOCK
    page = tl.load(TABLE + request * TABLE_STRIDE + column, valid, other=-1).to(
        tl.int64
    )
    valid &= (page >= 0) & (page < NBLOCKS)
    storage_block: tl.constexpr = LOGICAL_BLOCK // RATIO
    physical = page * storage_block + (position % LOGICAL_BLOCK) // RATIO
    closed = (position + 1) % RATIO == 0
    tl.store(SLOTS + token, tl.where(valid & closed, physical, -1))
    tl.store(GROUP_POS + token, tl.where(valid, position // RATIO * RATIO, 0))
    tl.store(LENGTHS + token, tl.where(valid, (position + 1) // RATIO, 0))
    if HAS_SWA:
        offset = tl.arange(0, SWA_TILE)
        seq_len = tl.load(SEQ + request, valid, other=0).to(tl.int64)
        first_query = seq_len - (end - begin)
        start = (
            tl.maximum(position - 127, 0)
            if CAUSAL
            else tl.maximum(first_query - 128, 0)
        )
        logical = start + offset
        take = valid & (offset < SWA_WIDTH)
        take &= (logical <= position) if CAUSAL else (logical < seq_len)
        block = tl.load(
            TABLE + request * TABLE_STRIDE + logical // LOGICAL_BLOCK,
            take,
            other=-1,
        ).to(tl.int64)
        take &= (block >= 0) & (block < NBLOCKS)
        index = block * LOGICAL_BLOCK + logical % LOGICAL_BLOCK
        tl.store(
            SWA + token * SWA_STRIDE + offset,
            tl.where(take, index, -1),
            offset < SWA_WIDTH,
        )
    if HAS_TABLES:
        cols = tl.arange(0, TABLE_TILE)
        block = tl.load(
            TABLE + request * TABLE_STRIDE + cols,
            valid & (cols < TABLE_COLUMNS),
            other=-1,
        )
        block = tl.where((block >= 0) & (block < NBLOCKS), block, -1)
        tl.store(
            TOKEN_TABLES + token * TOKEN_TABLE_STRIDE + cols,
            block,
            cols < TABLE_COLUMNS,
        )


@triton.jit
def _physical_indices_kernel(
    LOGICAL,
    REQUESTS,
    LENGTHS,
    TABLE,
    OUT,
    K,
    NREQS,
    NBLOCKS,
    LOGICAL_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COLUMNS: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    STORAGE_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    request = tl.load(REQUESTS + token).to(tl.int64)
    length = tl.load(LENGTHS + token).to(tl.int64)
    logical = tl.load(LOGICAL + token * LOGICAL_STRIDE + col, col < K, other=-1).to(
        tl.int64
    )
    block_col = logical // STORAGE_BLOCK
    valid = (request >= 0) & (request < NREQS) & (logical >= 0) & (logical < length)
    valid &= (col < K) & (block_col < TABLE_COLUMNS)
    page = tl.load(TABLE + request * TABLE_STRIDE + block_col, valid, other=-1).to(
        tl.int64
    )
    valid &= (page >= 0) & (page < NBLOCKS)
    physical = page * STORAGE_BLOCK + logical % STORAGE_BLOCK
    tl.store(OUT + token * OUT_STRIDE + col, tl.where(valid, physical, -1), col < K)


def _scalar(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _vector(value, length, dtypes, device, name):
    if not isinstance(value, torch.Tensor) or value.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    if value.dtype not in dtypes or value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} has incompatible dtype/device/layout")


def _no_alias(outputs, inputs):
    for index, output in enumerate(outputs):
        if output is None or not output.numel():
            continue
        others = inputs + outputs[:index] + outputs[index + 1 :]
        if any(
            other is not None
            and other.numel()
            and output.untyped_storage().data_ptr()
            == other.untyped_storage().data_ptr()
            for other in others
        ):
            raise ValueError(
                "metadata output buffers must not alias any input or each other"
            )


def prepare_cache_metadata(
    positions,
    query_start_loc,
    seq_lens,
    token_to_req,
    block_table,
    original_slots,
    num_actual_tokens,
    num_reqs,
    num_blocks,
    compress_ratio,
    logical_block_size,
    slot_out,
    group_positions_out,
    lengths_out,
    *,
    swa_indices_out=None,
    token_tables_out=None,
    causal=True,
):
    """Fill stable caller-owned cache addresses, grouped positions and context lengths.

    Logical blocks are 64 original tokens; compression is 1 or 2. A ratio-2
    output slot is valid only at a closed (odd) position. Original slot values
    are used only for padding validity, never as pre-compressed addresses.
    Bounds, request ordering and positions come from the framework. Invalid
    padding/request/position/page entries produce slot/index -1 and length/pos
    zero. positions=None derives positions from exact GPU seq_lens and query
    boundaries. Optional SWA [N,1,128] is left-packed; per-token block tables
    copy request metadata only. No KV values, sorting, CPU sync or state reset.
    Inputs are contiguous vectors except padded-row tables. Every output entry
    is overwritten; outputs do not alias inputs. All tensors share one device.
    """
    for value, name in [
        (num_actual_tokens, "num_actual_tokens"),
        (num_reqs, "num_reqs"),
        (num_blocks, "num_blocks"),
    ]:
        _scalar(value, name)
    _scalar(logical_block_size, "logical_block_size", 1)
    if logical_block_size != 64:
        raise ValueError("V4.1 cache metadata requires logical_block_size=64")
    if (
        isinstance(compress_ratio, bool)
        or not isinstance(compress_ratio, int)
        or compress_ratio not in (1, 2)
    ):
        raise ValueError("compress_ratio must be integer 1 or 2")
    if not isinstance(original_slots, torch.Tensor) or original_slots.ndim != 1:
        raise ValueError("original_slots must be a vector")
    n, device = original_slots.numel(), original_slots.device
    if device.type != "cuda" or num_actual_tokens > n:
        raise ValueError(
            "metadata requires an accelerator and actual tokens <= padded capacity"
        )
    ints = (torch.int32, torch.int64)
    _vector(original_slots, n, ints, device, "original_slots")
    _vector(token_to_req, n, ints, device, "token_to_req")
    _vector(query_start_loc, num_reqs + 1, ints, device, "query_start_loc")
    _vector(seq_lens, num_reqs, ints, device, "seq_lens")
    if positions is not None:
        _vector(positions, n, ints, device, "positions")
    if (
        not isinstance(block_table, torch.Tensor)
        or block_table.ndim != 2
        or block_table.shape[0] != num_reqs
        or block_table.shape[1] <= 0
        or block_table.stride(1) != 1
        or block_table.stride(0) < block_table.shape[1]
        or block_table.dtype not in ints
        or block_table.device != device
        or block_table.shape[1] * 64 >= 2**31
        or num_blocks * 64 >= 2**31
    ):
        raise ValueError(
            "block_table must be a padded-row integer matrix on the input device"
        )
    _vector(slot_out, n, (torch.int64,), device, "slot_out")
    _vector(group_positions_out, n, (torch.int64,), device, "group_positions_out")
    _vector(lengths_out, n, (torch.int32,), device, "lengths_out")
    if swa_indices_out is not None:
        if (
            not isinstance(swa_indices_out, torch.Tensor)
            or compress_ratio != 1
            or swa_indices_out.ndim != 3
            or swa_indices_out.shape[:2] != (n, 1)
            or swa_indices_out.shape[-1] != (128 if causal else 192)
        ):
            raise ValueError(
                "SWA metadata requires ratio 1 and 128 causal or 192 padded draft indices"
            )
        if (
            swa_indices_out.dtype != torch.int32
            or swa_indices_out.device != device
            or swa_indices_out.stride(-1) != 1
            or swa_indices_out.stride(0) < swa_indices_out.shape[-1]
        ):
            raise ValueError(
                "SWA output requires inner-contiguous INT32 on the input device"
            )
    if token_tables_out is not None and (
        not isinstance(token_tables_out, torch.Tensor)
        or token_tables_out.shape != (n, block_table.shape[1])
        or token_tables_out.dtype != torch.int32
        or (token_tables_out.device != device)
        or (token_tables_out.stride(1) != 1)
        or (token_tables_out.stride(0) < block_table.shape[1])
    ):
        raise ValueError(
            "per-token table output must be padded-row INT32 [N,table_columns]"
        )
    inputs = [
        positions,
        query_start_loc,
        seq_lens,
        token_to_req,
        block_table,
        original_slots,
    ]
    outputs = [
        slot_out,
        group_positions_out,
        lengths_out,
        swa_indices_out,
        token_tables_out,
    ]
    _no_alias(outputs, inputs)
    if n:
        with torch.cuda.device(device):
            _cache_metadata_kernel[(n,)](
                positions if positions is not None else original_slots,
                query_start_loc,
                seq_lens,
                token_to_req,
                block_table,
                original_slots,
                slot_out,
                group_positions_out,
                lengths_out,
                swa_indices_out if swa_indices_out is not None else slot_out,
                token_tables_out if token_tables_out is not None else slot_out,
                num_actual_tokens,
                num_reqs,
                num_blocks,
                block_table.stride(0),
                block_table.shape[1],
                swa_indices_out.stride(0) if swa_indices_out is not None else 1,
                token_tables_out.stride(0) if token_tables_out is not None else 1,
                logical_block_size,
                compress_ratio,
                positions is not None,
                swa_indices_out is not None,
                token_tables_out is not None,
                triton.next_power_of_2(block_table.shape[1]),
                causal,
                swa_indices_out.shape[-1] if swa_indices_out is not None else 128,
                (
                    triton.next_power_of_2(swa_indices_out.shape[-1])
                    if swa_indices_out is not None
                    else 128
                ),
            )
    return slot_out


def logical_to_physical_indices(
    logical_indices,
    token_to_req,
    lengths,
    block_table,
    num_blocks,
    storage_block_size,
    output,
):
    """Map selected logical compressed positions to physical cache indices only.

    Invalid selected/request/page IDs and positions beyond the causal length
    produce -1. Input/output are padded-row INT32 matrices, vectors contiguous.
    No score read or top-k computation; the caller owns all buffers.
    """
    _scalar(num_blocks, "num_blocks")
    _scalar(storage_block_size, "storage_block_size", 1)
    if storage_block_size not in (32, 64):
        raise ValueError("physical cache pages must contain 32 or 64 entries")
    if not isinstance(logical_indices, torch.Tensor) or logical_indices.ndim != 2:
        raise ValueError("logical_indices must be a matrix")
    n, k = logical_indices.shape
    device = logical_indices.device
    if (
        device.type != "cuda"
        or logical_indices.dtype != torch.int32
        or logical_indices.stride(1) != 1
        or logical_indices.stride(0) < k
    ):
        raise ValueError(
            "logical_indices must be a padded-row accelerator INT32 matrix"
        )
    _vector(token_to_req, n, (torch.int32, torch.int64), device, "token_to_req")
    _vector(lengths, n, (torch.int32,), device, "lengths")
    if (
        not isinstance(block_table, torch.Tensor)
        or block_table.ndim != 2
        or block_table.shape[1] <= 0
        or block_table.dtype != torch.int32
        or block_table.device != device
        or block_table.stride(1) != 1
        or block_table.stride(0) < block_table.shape[1]
    ):
        raise ValueError(
            "block_table must be a padded-row INT32 matrix on the input device"
        )
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != (n, k)
        or output.dtype != torch.int32
        or output.device != device
        or output.stride(1) != 1
        or output.stride(0) < k
    ):
        raise ValueError("output must match the INT32 index matrix shape/device")
    if num_blocks * storage_block_size >= 2**31:
        raise ValueError("physical cache indices must fit INT32")
    _no_alias([output], [logical_indices, token_to_req, lengths, block_table])
    if n and k:
        with torch.cuda.device(device):
            _physical_indices_kernel[(n, triton.cdiv(k, 256))](
                logical_indices,
                token_to_req,
                lengths,
                block_table,
                output,
                k,
                block_table.shape[0],
                num_blocks,
                logical_indices.stride(0),
                block_table.stride(0),
                block_table.shape[1],
                output.stride(0),
                storage_block_size,
                256,
            )
    return output

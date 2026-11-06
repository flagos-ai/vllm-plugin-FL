# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Triton producers for Hy4 request, position and slot metadata.

The caller owns scheduling, buffers and execution leases. These producers use
portable integer operations without vendor APIs or architecture-specific code.
"""

from __future__ import annotations

from flaggems_vllm.utils import libentry

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm.v1.worker.block_table import SlotMappingMode


@triton.jit(do_not_specialize=["max_num_tokens"])
def _compute_slot_mapping_graph_kernel(
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        num_tokens = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        # Device boundary changes on every replay, including shrinking batches.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    if start_idx == end_idx:
        for i in range(0, block_table_stride, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                block_table_ptr + req_idx * block_table_stride + offsets,
                NULL_BLOCK_ID,
                mask=offsets < block_table_stride,
            )
    if not MAP_SLOTS:
        return

    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        virtual_block_indices = pos // virtual_block_size
        virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
        slot_offsets = local_block_offsets % block_size
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit
def _compute_num_computed_tokens_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    num_computed_tokens_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
    tl.store(
        num_computed_tokens_ptr + offsets,
        seq_len - (query_end - query_start),
        mask=mask,
    )


def compute_common_attention_metadata(
    block_table, num_reqs, query_start_loc, positions, seq_lens, num_computed_tokens
):
    # Group iteration occurs during eager fallback/warm/capture only.
    for group in block_table.block_tables:
        mode = group.slot_mapping_mode
        if mode not in (SlotMappingMode.TOKEN_TO_KV_SLOT, SlotMappingMode.NONE):
            raise ValueError("unsupported slot mapping mode")
        _compute_slot_mapping_graph_kernel[(num_reqs + 1,)](
            group.max_num_batched_tokens,
            query_start_loc,
            positions,
            group.block_table.gpu,
            group.block_table.gpu.stride(0),
            group.block_size,
            group.slot_mapping.gpu,
            KV_CACHE_BLOCK_SIZE=group.kv_cache_block_size,
            BLOCKS_PER_KV_BLOCK=group.blocks_per_kv_block,
            TOTAL_CP_WORLD_SIZE=group.dcp_world_size,
            TOTAL_CP_RANK=group.dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=group.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            NULL_BLOCK_ID=NULL_BLOCK_ID,
            MAP_SLOTS=mode == SlotMappingMode.TOKEN_TO_KV_SLOT,
            BLOCK_SIZE=1024,
        )
    _compute_num_computed_tokens_kernel[(triton.cdiv(num_reqs, 256),)](
        query_start_loc,
        seq_lens,
        num_computed_tokens,
        num_reqs=num_reqs,
        BLOCK_SIZE=256,
    )


@triton.jit
def _prepare_positions_seq_lens_kernel(
    query_start_loc, computed, positions, seq_lens, BLOCK_SIZE: tl.constexpr
):
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    if start == end:
        tl.store(seq_lens + req, 0)
        return
    base = tl.load(computed + req).to(tl.int64)
    tl.store(seq_lens + req, base + end - start)
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        tl.store(positions + token, base + token - start, mask=token < end)


def compute_step_common_attention_metadata(
    block_table,
    num_reqs,
    query_start_loc,
    positions,
    seq_lens,
    num_computed_tokens,
    mrope_state=None,
    mrope_output=None,
):
    _prepare_positions_seq_lens_kernel[(num_reqs,)](
        query_start_loc, num_computed_tokens, positions, seq_lens, BLOCK_SIZE=256
    )
    compute_common_attention_metadata(
        block_table, num_reqs, query_start_loc, positions, seq_lens, num_computed_tokens
    )

    if mrope_state is not None:
        decode_mrope_kernel[(num_reqs,)](
            query_start_loc,
            num_computed_tokens,
            mrope_state,
            mrope_output,
            ROW_STRIDE=mrope_output.stride(0),
            COL_STRIDE=mrope_output.stride(1),
            BLOCK_SIZE=256,
        )


@triton.jit
def decode_mrope_kernel(
    query_start_loc,
    computed_tokens,
    state,
    output,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.load(state) == 0:
        return
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    # Empty padded rows must not read stale request delta/count values.
    if start == end:
        return
    base = tl.load(computed_tokens + req).to(tl.int64) + tl.load(state + 1 + req).to(
        tl.int64
    )
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        value = base + token - start
        for axis in range(3):
            tl.store(
                output + axis * ROW_STRIDE + token * COL_STRIDE, value, mask=token < end
            )


@triton.jit
def _compute_req_id_per_token_kernel(
    query_start_loc,
    output,
    counts,
    NUM_REQUESTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    if req == NUM_REQUESTS:
        # Native fill_(0) semantics for every padded token, including shrink.
        for i in range(start, CAPACITY, BLOCK_SIZE):
            token = i + tl.arange(0, BLOCK_SIZE)
            tl.store(output + token, 0, mask=token < CAPACITY)
        return
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    tl.store(counts + req, end - start)
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        tl.store(output + token, req, mask=token < end)


def populate_request_metadata(query_start_loc, request_ids, request_counts):
    """No allocation: fixed extents and current monotonic QSL are caller contracts."""
    num_reqs = request_counts.numel()
    _compute_req_id_per_token_kernel[(num_reqs + 1,)](
        query_start_loc,
        request_ids,
        request_counts,
        NUM_REQUESTS=num_reqs,
        CAPACITY=request_ids.numel(),
        BLOCK_SIZE=256,
    )


@libentry()
@triton.jit(do_not_specialize=["NMAP", "B"])
def _async_input_correct_kernel(
    P,
    NC,
    AC,
    VC,
    PP,
    PD,
    NMAP,
    B,
    CAP: tl.constexpr,
    VC_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.arange(0, BLOCK)
    prev = tl.load(P + 2 * NMAP + r, r < B, -1).to(tl.int32)
    cpu = tl.load(P + 2 * NMAP + B + r, r < B, 0).to(tl.int32)
    g = tl.maximum(prev, 0)
    drafts = tl.load(P + 2 * NMAP + 2 * B + g, (r < B) & (g < CAP), 0).to(tl.int32)
    old = tl.load(NC + g, (r < B) & (g < CAP), 0)
    valid = tl.load(VC + g, (r < B) & (g < VC_SIZE), 0).to(tl.int32)
    accepted = tl.load(AC + r, r < B, 0)
    all_drafts = tl.load(P + 2 * NMAP + 2 * B + r, r < CAP, 0).to(tl.int32)
    # Every old gather completes before any in-place permutation write, also
    # for cycles spanning warps. One CTA is intentional, not an occupancy tune.
    tl.debug_barrier()
    participating = (prev >= 0) & (drafts > 0)
    tl.store(NC + r, tl.where(participating, old + valid, cpu), r < B)
    tl.store(AC + r, tl.where(participating, valid, accepted), r < B)
    tl.store(PP + r, prev, r < B)
    tl.store(PD + r, all_drafts, r < CAP)


@libentry()
@triton.jit(do_not_specialize=["NMAP", "NSAMPLE"])
def _async_input_gather_kernel(
    P,
    OUT,
    SAMPLE,
    DRAFT,
    NMAP,
    NSAMPLE,
    SAMPLE_S0: tl.constexpr,
    DRAFT_S0: tl.constexpr,
    DRAFT_S1: tl.constexpr,
    DRAFT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = i < NMAP
    dst = tl.load(P + i, active, 0).to(tl.int32)
    src = tl.load(P + NMAP + i, active, 0).to(tl.int32)
    a = tl.load(SAMPLE + src * SAMPLE_S0, active & (i < NSAMPLE), 0)
    b = tl.load(
        DRAFT + (src // DRAFT_K) * DRAFT_S0 + (src % DRAFT_K) * DRAFT_S1,
        active & (i >= NSAMPLE),
        0,
    ).to(tl.int32)
    tl.store(OUT + dst, tl.where(i < NSAMPLE, a, b), active)


def async_input_correct(
    packet,
    num_computed,
    num_accepted,
    valid_counts,
    prev_positions,
    prev_drafts,
    *,
    num_maps,
    num_reqs,
    prev_capacity,
):
    """Exact native correction; packet indices must be CPU-validated."""
    _async_input_correct_kernel[(1,)](
        packet,
        num_computed,
        num_accepted,
        valid_counts,
        prev_positions,
        prev_drafts,
        num_maps,
        num_reqs,
        prev_capacity,
        valid_counts.numel(),
        triton.next_power_of_2(prev_capacity),
        num_warps=4,
    )


def async_input_gather(
    packet, out_ids, sampled_ids, draft_ids, *, num_maps, num_sampled
):
    """Sources are disjoint from out_ids; destination indices are unique."""
    if not num_maps:
        return
    _async_input_gather_kernel[(triton.cdiv(num_maps, 128),)](
        packet,
        out_ids,
        sampled_ids,
        draft_ids,
        num_maps,
        num_sampled,
        sampled_ids.stride(0),
        draft_ids.stride(0),
        draft_ids.stride(1),
        draft_ids.shape[1],
        128,
        num_warps=4,
    )


def _compute_step_common_with_request_ids(
    block_table,
    num_reqs,
    query_start_loc,
    positions,
    seq_lens,
    num_computed_tokens,
    request_ids,
    request_counts,
    *extra_tensors,
):
    compute_step_common_attention_metadata(
        block_table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        *extra_tensors,
    )
    populate_request_metadata(query_start_loc, request_ids, request_counts)

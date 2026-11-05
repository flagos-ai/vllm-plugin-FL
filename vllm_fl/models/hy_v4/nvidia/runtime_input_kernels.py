# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native V1 async input maps: integer correction and gather/cast/scatter.

Caller constructs valid packet indices on CPU. No allocation or Torch compute.
The fixed one-CTA correction preserves every old-state read before writes.
"""

import torch
import triton
import triton.language as tl
from flaggems_vllm.utils import libentry


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


def _vector(x, dtype, minimum, device):
    if (
        not torch.is_tensor(x)
        or not x.is_cuda
        or x.device != device
        or x.dtype != dtype
        or x.ndim != 1
        or x.stride(0) != 1
        or x.numel() < minimum
    ):
        raise NotImplementedError("unsupported async staging vector ABI")


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
    if not (0 <= num_maps <= 2**31 - 1 and 1 <= num_reqs <= prev_capacity <= 1024):
        raise NotImplementedError("unsupported async staging extent")
    dev = packet.device
    _vector(packet, torch.int64, 2 * num_maps + 2 * num_reqs + prev_capacity, dev)
    for x, n in (
        (num_computed, prev_capacity),
        (num_accepted, num_reqs),
        (prev_drafts, prev_capacity),
        (valid_counts, 1),
    ):
        _vector(x, torch.int32, n, dev)
    _vector(prev_positions, torch.int64, num_reqs, dev)
    pointers = [
        x.untyped_storage().data_ptr()
        for x in (
            packet,
            num_computed,
            num_accepted,
            valid_counts,
            prev_positions,
            prev_drafts,
        )
    ]
    if len(set(pointers)) != len(pointers):
        raise NotImplementedError("async correction buffer alias")
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
    if not (0 <= num_sampled <= num_maps <= 2**31 - 1):
        raise NotImplementedError("unsupported map counts")
    dev = packet.device
    _vector(packet, torch.int64, 2 * num_maps, dev)
    _vector(out_ids, torch.int32, 1, dev)
    for x in (sampled_ids, draft_ids):
        if (
            not x.is_cuda
            or x.device != dev
            or x.ndim != 2
            or x.shape[1] < 1
            or min(x.stride()) < 0
            or x.dtype not in (torch.int32, torch.int64)
        ):
            raise NotImplementedError("unsupported async staging source ABI")
        if x.untyped_storage().data_ptr() == out_ids.untyped_storage().data_ptr():
            raise NotImplementedError("input map source/destination alias")
    if sampled_ids.dtype != torch.int32:
        raise NotImplementedError("native scatter sample source must be int32")
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


__all__ = ["async_input_correct", "async_input_gather"]

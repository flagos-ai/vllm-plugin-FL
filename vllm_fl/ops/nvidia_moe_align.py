# Copyright (c) 2026 BAAI. All rights reserved.

"""Graph-safe MoE alignment for NVIDIA EMPTY wheels with invalid routes.

Follow vLLM 0.32's native alignment contract: filter global IDs, map to local
experts before counting, and keep the original flattened route indices. The
FlagGems alignment API does not expose this invalid-route/pre-map contract.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _local_experts(
    Ids, Map, rows, N: tl.constexpr, E: tl.constexpr, HAS_MAP: tl.constexpr
):
    expert = tl.load(Ids + rows, rows < N, other=-1)
    valid = (rows < N) & (expert >= 0) & (expert < E)
    # Mask before looking up the map: padded -1 IDs must never index it.
    if HAS_MAP:
        expert = tl.load(Map + tl.where(valid, expert, 0), valid, other=-1)
        valid = valid & (expert >= 0) & (expert < E)
    return tl.where(valid, expert, 0).to(tl.int32), valid


@triton.jit
def _count_or_scatter(
    Ids,
    Map,
    Counts,
    Starts,
    Sorted,
    N: tl.constexpr,
    E: tl.constexpr,
    HAS_MAP: tl.constexpr,
    SCATTER: tl.constexpr,
    TILE: tl.constexpr,
):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    expert, valid = _local_experts(Ids, Map, rows, N, E, HAS_MAP)
    rank = tl.atomic_add(Counts + expert, 1, valid, sem="relaxed")
    if SCATTER:
        start = tl.load(Starts + expert, valid, other=0)
        tl.store(Sorted + start + rank, rows, valid)


@triton.jit
def _prefix(
    Counts,
    Starts,
    Ends,
    PostPad,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    experts = tl.arange(0, TILE)
    counts = tl.load(Counts + experts, experts < E, other=0)
    padded = tl.cdiv(counts, BLOCK) * BLOCK
    ends = tl.cumsum(padded)
    tl.store(Starts + experts, ends - padded, experts < E)
    tl.store(Ends + experts, ends, experts < E)
    tl.store(PostPad, tl.sum(padded))
    # Reuse the histogram as the per-expert scatter cursor.
    tl.store(Counts + experts, 0, experts < E)


@triton.jit
def _fill_expert_blocks(Starts, Ends, Experts, BLOCK: tl.constexpr, TILE: tl.constexpr):
    expert = tl.program_id(0)
    start = tl.load(Starts + expert) // BLOCK
    end = tl.load(Ends + expert) // BLOCK
    for offset in range(start, end, TILE):
        blocks = offset + tl.arange(0, TILE)
        tl.store(Experts + blocks, expert, blocks < end)


def moe_align_block_size_nvidia(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align only valid routes without host reads or dynamic-sized allocations."""
    if not topk_ids.is_cuda or topk_ids.dtype not in (
        torch.int32,
        torch.int64,
        torch.uint32,
    ):
        raise ValueError("MoE routes must be CUDA integer IDs")
    if block_size <= 0 or num_experts <= 0:
        raise ValueError("MoE block size and expert count must be positive")
    if expert_map is not None and (
        expert_map.device != topk_ids.device
        or expert_map.shape != (num_experts,)
        or expert_map.dtype != torch.int32
    ):
        raise ValueError("MoE expert map must be an int32 vector on the route device")

    numel = topk_ids.numel()
    capacity = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = triton.cdiv(capacity, block_size) * block_size
    if numel < num_experts:
        capacity = min(numel * block_size, capacity)
    if capacity >= 2**31:
        raise ValueError("MoE aligned route indices must fit in int32")
    device = topk_ids.device
    sorted_ids = torch.full((capacity,), numel, dtype=torch.int32, device=device)
    expert_ids = torch.full(
        (triton.cdiv(capacity, block_size),), -1, dtype=torch.int32, device=device
    )
    post_pad = torch.zeros((1,), dtype=torch.int32, device=device)
    if numel == 0:
        return sorted_ids, expert_ids, post_pad

    ids = topk_ids.contiguous()
    mapping = expert_map.contiguous() if expert_map is not None else ids
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=device)
    starts = torch.empty_like(counts)
    ends = torch.empty_like(counts)
    grid = (triton.cdiv(numel, 256),)
    args = (ids, mapping, counts, starts, sorted_ids)
    _count_or_scatter[grid](
        *args, numel, num_experts, expert_map is not None, False, 256
    )
    _prefix[(1,)](
        counts,
        starts,
        ends,
        post_pad,
        num_experts,
        block_size,
        triton.next_power_of_2(num_experts),
    )
    _fill_expert_blocks[(num_experts,)](starts, ends, expert_ids, block_size, 128)
    _count_or_scatter[grid](
        *args, numel, num_experts, expert_map is not None, True, 256
    )
    return sorted_ids, expert_ids, post_pad

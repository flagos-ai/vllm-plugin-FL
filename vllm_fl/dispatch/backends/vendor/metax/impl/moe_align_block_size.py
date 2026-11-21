# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounds-safe MetaX MoE alignment for vLLM 0.32 padding expert IDs.

The four-stage histogram/prefix/scatter algorithm follows FlagGems 5.5's
``fused/moe_align_block_size.py``. Its original histogram and scatter index
through negative padding IDs. Keep the algorithm in the plugin and mask IDs
before either indexed access, as vLLM 0.32's CUDA align implementation does.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _count_and_initialize(
    topk_ids,
    expert_map,
    counts,
    sorted_ids,
    expert_ids,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    SORTED_CAPACITY: tl.constexpr,
    EXPERT_CAPACITY: tl.constexpr,
    ROUTES_PER_PARTITION: tl.constexpr,
    SORTED_PER_PARTITION: tl.constexpr,
    EXPERTS_PER_PARTITION: tl.constexpr,
    HAS_MAP: tl.constexpr,
    IGNORE_NONLOCAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    for start in range(0, SORTED_PER_PARTITION, BLOCK):
        local = start + lanes
        offsets = pid * SORTED_PER_PARTITION + local
        mask = (local < SORTED_PER_PARTITION) & (offsets < SORTED_CAPACITY)
        tl.store(sorted_ids + offsets, NUM_ROUTES, mask=mask)
    for start in range(0, EXPERTS_PER_PARTITION, BLOCK):
        local = start + lanes
        offsets = pid * EXPERTS_PER_PARTITION + local
        mask = (local < EXPERTS_PER_PARTITION) & (offsets < EXPERT_CAPACITY)
        tl.store(expert_ids + offsets, -1, mask=mask)
    for start in range(0, ROUTES_PER_PARTITION, BLOCK):
        local = start + lanes
        offsets = pid * ROUTES_PER_PARTITION + local
        mask = (local < ROUTES_PER_PARTITION) & (offsets < NUM_ROUTES)
        experts = tl.load(topk_ids + offsets, mask=mask, other=-1).to(tl.int64)
        valid = mask & (experts >= 0) & (experts < NUM_EXPERTS)
        if HAS_MAP and IGNORE_NONLOCAL:
            local_experts = tl.load(expert_map + experts, mask=valid, other=-1)
            valid = valid & (local_experts >= 0)
        tl.atomic_add(counts + (pid + 1) * NUM_EXPERTS + experts, 1, mask=valid)


@triton.jit
def _prefix_partition_counts(counts, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    partitions = tl.arange(0, BLOCK) + 1
    mask = partitions <= NUM_EXPERTS
    values = tl.load(counts + partitions * NUM_EXPERTS + expert, mask=mask, other=0)
    prefix = tl.cumsum(values, axis=0)
    tl.store(counts + partitions * NUM_EXPERTS + expert, prefix, mask=mask)


@triton.jit
def _align_expert_counts(
    counts,
    cumsum,
    num_post_pad,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    experts = tl.arange(0, BLOCK)
    mask = experts < NUM_EXPERTS
    totals = tl.load(counts + NUM_EXPERTS * NUM_EXPERTS + experts, mask=mask, other=0)
    aligned = tl.cdiv(totals, BLOCK_SIZE) * BLOCK_SIZE
    tl.store(cumsum + experts + 1, tl.cumsum(aligned, axis=0), mask=mask)
    tl.store(num_post_pad, tl.sum(aligned, axis=0))


@triton.jit
def _scatter_routes(
    topk_ids,
    expert_map,
    counts,
    cumsum,
    sorted_ids,
    expert_ids,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROUTES_PER_PARTITION: tl.constexpr,
    HAS_MAP: tl.constexpr,
    IGNORE_NONLOCAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = tl.load(cumsum + pid)
    end = tl.load(cumsum + pid + 1)
    block_expert = pid
    if HAS_MAP:
        block_expert = tl.load(expert_map + pid).to(tl.int32)
    for offset in range(start, end, BLOCK_SIZE):
        tl.store(expert_ids + offset // BLOCK_SIZE, block_expert)

    lanes = tl.arange(0, BLOCK)
    for base in range(0, ROUTES_PER_PARTITION, BLOCK):
        local = base + lanes
        offsets = pid * ROUTES_PER_PARTITION + local
        mask = (local < ROUTES_PER_PARTITION) & (offsets < NUM_ROUTES)
        experts = tl.load(topk_ids + offsets, mask=mask, other=-1).to(tl.int64)
        valid = mask & (experts >= 0) & (experts < NUM_EXPERTS)
        if HAS_MAP and IGNORE_NONLOCAL:
            local_experts = tl.load(expert_map + experts, mask=valid, other=-1)
            valid = valid & (local_experts >= 0)
        ranks = tl.atomic_add(counts + pid * NUM_EXPERTS + experts, 1, mask=valid)
        starts = tl.load(cumsum + experts, mask=valid, other=0)
        tl.store(sorted_ids + starts + ranks, offsets, mask=valid)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align valid global routes; optionally exclude nonlocal mapped experts."""
    if block_size <= 0 or num_experts <= 0:
        raise ValueError("MoE block_size and num_experts must be positive")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("MoE expert IDs must be int32 or int64")
    topk_ids = topk_ids.contiguous()
    if expert_map is not None:
        if expert_map.dtype not in (torch.int32, torch.int64):
            raise TypeError("expert_map must be int32 or int64")
        if expert_map.device != topk_ids.device:
            raise ValueError("expert_map and expert IDs must use the same device")
        if expert_map.numel() < num_experts:
            raise ValueError("expert_map must cover every global expert")
        expert_map = expert_map.contiguous()
    num_routes = topk_ids.numel()
    capacity = num_routes + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = triton.cdiv(capacity, block_size) * block_size
    if num_routes < num_experts:
        capacity = min(num_routes * block_size, capacity)
    num_blocks = triton.cdiv(capacity, block_size)
    sorted_ids = torch.empty(capacity, device=topk_ids.device, dtype=torch.int32)
    expert_ids = torch.empty(num_blocks, device=topk_ids.device, dtype=torch.int32)
    num_post_pad = torch.zeros(1, device=topk_ids.device, dtype=torch.int32)
    if num_routes == 0:
        return sorted_ids, expert_ids, num_post_pad
    counts = torch.zeros(
        (num_experts + 1, num_experts), device=topk_ids.device, dtype=torch.int32
    )
    cumsum = torch.zeros(num_experts + 1, device=topk_ids.device, dtype=torch.int32)
    partition_size = triton.cdiv(num_routes, num_experts)
    common = dict(num_warps=4, num_stages=1)
    _count_and_initialize[(num_experts,)](
        topk_ids,
        expert_map,
        counts,
        sorted_ids,
        expert_ids,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        SORTED_CAPACITY=capacity,
        EXPERT_CAPACITY=num_blocks,
        ROUTES_PER_PARTITION=partition_size,
        SORTED_PER_PARTITION=triton.cdiv(capacity, num_experts),
        EXPERTS_PER_PARTITION=triton.cdiv(num_blocks, num_experts),
        HAS_MAP=expert_map is not None,
        IGNORE_NONLOCAL=ignore_invalid_experts,
        BLOCK=256,
        **common,
    )
    expert_block = triton.next_power_of_2(num_experts)
    _prefix_partition_counts[(num_experts,)](
        counts, NUM_EXPERTS=num_experts, BLOCK=expert_block, **common
    )
    _align_expert_counts[(1,)](
        counts,
        cumsum,
        num_post_pad,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE=block_size,
        BLOCK=expert_block,
        **common,
    )
    _scatter_routes[(num_experts,)](
        topk_ids,
        expert_map,
        counts,
        cumsum,
        sorted_ids,
        expert_ids,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE=block_size,
        ROUTES_PER_PARTITION=partition_size,
        HAS_MAP=expert_map is not None,
        IGNORE_NONLOCAL=ignore_invalid_experts,
        BLOCK=256,
        **common,
    )
    return sorted_ids, expert_ids, num_post_pad


@triton.jit
def _zero_invalid_routes(
    output,
    topk_ids,
    expert_map,
    NUM_ELEMENTS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    OUT_S2: tl.constexpr,
    IDS_S0: tl.constexpr,
    IDS_S1: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUM_ELEMENTS
    routes, columns = offsets // WIDTH, offsets % WIDTH
    rows, slots = routes // TOP_K, routes % TOP_K
    experts = tl.load(
        topk_ids + rows * IDS_S0 + slots * IDS_S1, mask=mask, other=-1
    ).to(tl.int64)
    valid = mask & (experts >= 0) & (experts < NUM_EXPERTS)
    if HAS_MAP:
        local_experts = tl.load(expert_map + experts, mask=valid, other=-1)
        valid = valid & (local_experts >= 0)
    tl.store(
        output + rows * OUT_S0 + slots * OUT_S1 + columns * OUT_S2,
        0,
        mask=mask & ~valid,
    )


def zero_invalid_expert_outputs(
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> None:
    """Clear skipped GEMM slots before reduction, preserving every valid slot."""
    if output.ndim != 3 or topk_ids.ndim != 2 or output.shape[:2] != topk_ids.shape:
        raise ValueError("MoE output must have shape [tokens, top_k, hidden]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("MoE expert IDs must be int32 or int64")
    if output.numel() == 0:
        return
    if expert_map is not None:
        if expert_map.dtype not in (torch.int32, torch.int64):
            raise TypeError("expert_map must be int32 or int64")
        if expert_map.device != topk_ids.device or expert_map.numel() < num_experts:
            raise ValueError("expert_map must cover every expert on the IDs' device")
        expert_map = expert_map.contiguous()
    _zero_invalid_routes[(triton.cdiv(output.numel(), 512),)](
        output,
        topk_ids,
        expert_map,
        NUM_ELEMENTS=output.numel(),
        WIDTH=output.shape[2],
        TOP_K=topk_ids.shape[1],
        NUM_EXPERTS=num_experts,
        OUT_S0=output.stride(0),
        OUT_S1=output.stride(1),
        OUT_S2=output.stride(2),
        IDS_S0=topk_ids.stride(0),
        IDS_S1=topk_ids.stride(1),
        HAS_MAP=expert_map is not None,
        BLOCK=512,
        num_warps=4,
        num_stages=1,
    )

# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference fused-MoE operator implementations using PyTorch.

The flag_gems (flagtree-compiled) MoE kernels compute wrong values on
TX8110 without raising, so these ops are routed to pure-torch fallbacks
that are numerically fine on txda (see config/tsingmicro.yaml).
"""

from __future__ import annotations

import torch

from vllm.utils.math_utils import round_up


def moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch port of flag_gems' moe_align_block_size_triton semantics.

    Allocation math matches flaggems (numel + num_experts*(block_size-1),
    round_up, numel<num_experts cap). Sorting matches the tsingmicro triton
    kernel: a stable sort of the flattened topk_ids by expert groups tokens
    per expert in original (token-major) order, and block-aligned counts make
    whole blocks belong to exactly one expert. Padding sentinel is `numel` in
    sorted_ids and -1 in expert_ids, matching the triton kernel and vllm csrc.
    """
    numel = topk_ids.numel()
    flat = topk_ids.reshape(-1)

    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if numel < num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)

    sorted_ids = torch.full(
        (max_num_tokens_padded,), numel, dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

    if numel == 0:
        num_tokens_post_pad[0] = 0
        return sorted_ids, torch.full(
            (0,), -1, dtype=torch.int32, device=topk_ids.device
        ), num_tokens_post_pad

    # Order tokens by expert (stable: original order within each expert),
    # then drop invalid ids to padding, exactly like the triton kernel's
    # `valid = mask & (expert_id < num_experts)` count filter.
    order = torch.argsort(flat, stable=True)
    expert_of_sorted = flat[order]
    if ignore_invalid_experts:
        keep = (expert_of_sorted >= 0) & (expert_of_sorted < num_experts)
        order = order[keep]
        expert_of_sorted = expert_of_sorted[keep]
    num_valid = order.numel()

    # cdiv(counts, bs)*bs per expert, then exclusive-scan into offsets.
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    ones = torch.ones(num_valid, dtype=torch.int32, device=topk_ids.device)
    counts.scatter_add_(0, expert_of_sorted, ones)  # bincount absent on txda
    aligned = ((counts + block_size - 1) // block_size) * block_size
    offsets = aligned.cumsum(0, dtype=torch.int64)
    starts = offsets - aligned

    total = int(offsets[-1].item())
    num_tokens_post_pad[0] = total
    max_num_m_blocks = (total + block_size - 1) // block_size
    expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )

    # Scatter tokens into each expert's aligned span: the triton kernel
    # reads tokens at block-aligned offsets, so per-expert padding must be
    # materialized in the output (a plain prefix fill shifts later experts).
    # starts[e] is the padded offset of expert e; within-expert tokens stay
    # in original (stable) order, so position = starts[e] + within-index.
    starts = (offsets - aligned).to(torch.int64)
    run_start = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    present = counts > 0
    run_start[present] = counts.cumsum(0)[present] - counts[present]
    within = torch.arange(num_valid, device=topk_ids.device) - run_start[expert_of_sorted]
    positions = starts[expert_of_sorted] + within
    sorted_ids[positions] = order.to(torch.int32)
    # Each block belongs to the expert that owns its padded span. Aligned
    # counts are multiples of block_size, so a block never straddles experts
    # and expert_of_padded[block_start] is unambiguous; there is no partial
    # final block. (Indexing the valid-only list by padded positions, as an
    # earlier version did, labels the wrong blocks whenever an expert's
    # aligned span exceeds one block.)
    expert_of_padded = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=topk_ids.device), aligned
    )
    block_starts = torch.arange(max_num_m_blocks, device=topk_ids.device) * block_size
    expert_ids[:] = expert_of_padded[block_starts]

    if expert_map is not None:
        # Padding blocks (-1) are never read by the moe kernel; map only
        # real experts so -1 cannot wrap to a valid map entry.
        expert_ids = torch.where(
            expert_ids == -1, -1, expert_map[expert_ids.clamp(min=0)]
        )

    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_sum_torch(inp: torch.Tensor, out: torch.Tensor) -> None:
    """Sum over the intermediate-cache hidden dim, written into out in-place."""
    reduced = inp.float().sum(dim=1)
    out.copy_(reduced.to(dtype=out.dtype))


def topk_softmax_torch(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-row softmax -> topk, matching vllm csrc topk_softmax.

    Fills the three pre-allocated output tensors in-place; weights are the
    raw (post-softmax) scores at the selected experts.
    """
    scores = torch.softmax(gating_output.float(), dim=-1)
    vals, idx = torch.topk(scores, k=topk_weights.size(-1), dim=-1, sorted=False)
    if renormalize:
        vals = vals / vals.sum(-1, keepdim=True)
    topk_weights.copy_(vals)
    topk_indices.copy_(idx.to(topk_indices.dtype))
    token_expert_indices.copy_(idx.to(torch.int32))
    return topk_weights, topk_indices


def grouped_topk_torch(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor | None,
    scoring_func: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch replica of flag_gems' grouped_topk semantics.

    Group score = top-2 sum of (score+bias) within the group (flag_gems
    max1+max2 convention), groups selected by that score, then top-k by
    selection score (score+bias, -inf outside selected groups). Output
    weights are the RAW processed scores (no bias), float32, matching
    flag_gems so the F (flagtree) and T (triton) paths route identically.
    """
    scores = scores.float()
    M, num_experts = scores.shape
    assert num_experts % n_group == 0
    assert scoring_func in (0, 1)

    if bias is None:
        bias = torch.zeros(num_experts, dtype=scores.dtype, device=scores.device)
    else:
        bias = bias.to(dtype=scores.dtype, device=scores.device).reshape(-1)
        assert bias.numel() == num_experts

    scores_processed = torch.sigmoid(scores) if scoring_func == 1 else scores
    scored = scores_processed + bias

    # Top-2 sum of the scored values within each group (flag_gems convention).
    group_scores = scored.view(M, n_group, -1).topk(2, dim=-1).values.sum(-1)
    # NaN group scores count as non-finite (kernel converts them to -inf).
    finite = torch.isfinite(group_scores)
    group_scores = torch.where(
        finite, group_scores, torch.full_like(group_scores, -float("inf"))
    )

    # Select topk_group groups, top-k by scored value inside them, output the
    # RAW processed score (no bias). Rows whose max group score is -inf fall
    # back to the flag_gems default (1/topk, 0..topk-1), applied after
    # renormalize/scaling so defaults stay unscaled.
    group_idx = group_scores.topk(topk_group, dim=-1, sorted=False).indices
    mask = torch.zeros(M, n_group, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, group_idx, True)
    mask = mask.repeat_interleave(num_experts // n_group, dim=1)
    selection = torch.where(mask, scored, torch.full_like(scored, -float("inf")))
    topk_values, topk_indices = selection.topk(topk, dim=-1, sorted=False)
    topk_weights = scores_processed.gather(-1, topk_indices)

    if renormalize:
        topk_weights = (
            topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
        ) * routed_scaling_factor
    else:
        topk_weights = topk_weights * routed_scaling_factor

    if_proceed = group_scores.max(-1).values != -float("inf")
    default_vals = torch.full(
        (M, topk), 1.0 / topk, dtype=torch.float32, device=scores.device
    )
    default_idx = torch.arange(topk, dtype=torch.int32, device=scores.device)
    default_idx = default_idx.expand(M, -1)
    topk_weights = torch.where(
        if_proceed.unsqueeze(-1), topk_weights, default_vals
    )
    topk_indices = torch.where(
        if_proceed.unsqueeze(-1), topk_indices.to(torch.int32), default_idx
    )

    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)

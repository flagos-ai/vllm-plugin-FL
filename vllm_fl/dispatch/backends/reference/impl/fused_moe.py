# Copyright (c) 2026 BAAI. All rights reserved.

"""Native PyTorch reference implementations for MoE helper operators."""

from __future__ import annotations

import torch


def moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_experts < 0:
        raise ValueError("num_experts must be non-negative")
    if expert_map is not None and expert_map.numel() < num_experts:
        raise ValueError("expert_map must contain every global expert")

    num_routes = topk_ids.numel()
    max_num_tokens_padded = num_routes + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = (
            (max_num_tokens_padded + block_size - 1) // block_size * block_size
        )
    if num_routes < num_experts:
        max_num_tokens_padded = min(
            num_routes * block_size, max_num_tokens_padded
        )

    sentinel = num_routes
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        sentinel,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (
        max_num_tokens_padded + block_size - 1
    ) // block_size
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_padded = torch.zeros(
        (1,), dtype=torch.int32, device=topk_ids.device
    )

    flat_ids = topk_ids.reshape(-1)
    token_offset = 0
    block_offset = 0
    for global_expert in range(num_experts):
        local_expert = global_expert
        if expert_map is not None:
            local_expert = int(expert_map[global_expert].item())
            if ignore_invalid_experts and local_expert < 0:
                continue

        route_indices = torch.nonzero(
            flat_ids == global_expert, as_tuple=False
        ).flatten()
        route_count = route_indices.numel()
        if route_count == 0:
            continue

        padded_count = (
            (route_count + block_size - 1) // block_size * block_size
        )
        sorted_token_ids[token_offset : token_offset + route_count].copy_(
            route_indices.to(torch.int32)
        )
        num_blocks = padded_count // block_size
        output_expert = local_expert if expert_map is not None else global_expert
        expert_ids[block_offset : block_offset + num_blocks] = output_expert
        token_offset += padded_count
        block_offset += num_blocks

    num_tokens_post_padded.fill_(token_offset)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def moe_sum_torch(inp: torch.Tensor, out: torch.Tensor) -> None:
    if inp.ndim != out.ndim + 1:
        raise ValueError(
            "moe_sum expects input shape [tokens, topk, ...] and output "
            "shape [tokens, ...]"
        )
    if inp.shape[0] != out.shape[0] or inp.shape[2:] != out.shape[1:]:
        raise ValueError(
            f"Incompatible moe_sum shapes: input={tuple(inp.shape)}, "
            f"output={tuple(out.shape)}"
        )
    torch.sum(inp, dim=1, out=out)


def topk_softmax_torch(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gating_output.ndim != 2:
        raise ValueError("gating_output must be a 2D tensor")
    if topk_weights.shape != topk_indices.shape:
        raise ValueError("topk_weights and topk_indices must have equal shapes")
    if token_expert_indices.shape != topk_weights.shape:
        raise ValueError("token_expert_indices must match topk_weights")

    num_tokens = gating_output.shape[0]
    topk = topk_weights.shape[1]
    if topk > gating_output.shape[1]:
        raise ValueError("topk cannot exceed the number of experts")

    probabilities = torch.softmax(gating_output.float(), dim=-1)
    weights, indices = torch.topk(
        probabilities, k=topk, dim=-1, largest=True, sorted=True
    )
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    topk_weights.copy_(weights.to(topk_weights.dtype))
    topk_indices.copy_(indices.to(topk_indices.dtype))
    source_rows = torch.arange(
        num_tokens, device=gating_output.device, dtype=torch.int32
    ).unsqueeze(1) + (
        torch.arange(topk, device=gating_output.device, dtype=torch.int32)
        * num_tokens
    ).unsqueeze(0)
    token_expert_indices.copy_(source_rows)
    return topk_weights, topk_indices


def grouped_topk_torch(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D tensor")
    num_tokens, num_experts = scores.shape
    if n_group <= 0 or num_experts % n_group:
        raise ValueError("num_experts must be divisible by a positive n_group")
    if not 0 < topk_group <= n_group:
        raise ValueError("topk_group must be in [1, n_group]")
    if not 0 < topk <= num_experts:
        raise ValueError("topk must be in [1, num_experts]")
    if scoring_func not in (0, 1):
        raise ValueError("scoring_func must be 0 (none) or 1 (sigmoid)")

    bias = bias.flatten()
    if bias.numel() != num_experts:
        raise ValueError("bias length must match num_experts")

    routing_scores = scores.float()
    if scoring_func == 1:
        routing_scores = routing_scores.sigmoid()
    original_scores = routing_scores
    biased_scores = routing_scores + bias.float().unsqueeze(0)
    experts_per_group = num_experts // n_group

    group_scores = biased_scores.view(
        num_tokens, n_group, experts_per_group
    ).topk(min(2, experts_per_group), dim=-1, sorted=True).values.sum(dim=-1)
    selected_groups = torch.topk(
        group_scores, k=topk_group, dim=-1, sorted=True
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(
        num_tokens, n_group, experts_per_group
    ).reshape(num_tokens, num_experts)

    selection_scores = biased_scores.masked_fill(~expert_mask, float("-inf"))
    topk_indices = torch.topk(
        selection_scores, k=topk, dim=-1, sorted=True
    ).indices
    topk_values = original_scores.gather(1, topk_indices)
    if renormalize:
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True)
    topk_values = topk_values * routed_scaling_factor
    return topk_values.to(torch.float32), topk_indices.to(torch.int32)

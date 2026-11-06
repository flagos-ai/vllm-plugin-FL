# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon implementation of vLLM's raw ``moe_align_block_size`` op."""

from __future__ import annotations

import torch


def moe_align_block_size_hygon_out(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """Fill preallocated outputs with vllm_hcu's categorized LightOp API."""
    from lightop.moe import moe_align_block_size_out

    moe_align_block_size_out(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
        is_ep=False,
        is_fuse_fill=True,
    )


__all__ = ["moe_align_block_size_hygon_out"]

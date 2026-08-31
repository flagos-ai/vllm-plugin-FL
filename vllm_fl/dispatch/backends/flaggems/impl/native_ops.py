# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems implementations for vLLM native out-parameter ABIs."""

from __future__ import annotations

import torch


def silu_and_mul_out_flaggems(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """Fill vLLM's preallocated output with the existing FlagGems kernel."""
    from flag_gems.modules.activation import gems_silu_and_mul

    hidden_size = input.shape[-1] // 2
    # The current FlagGems API returns a tensor and has no documented out ABI.
    # Keep the unavoidable final copy isolated to this native adapter.
    output.copy_(
        gems_silu_and_mul(
            input[..., :hidden_size],
            input[..., hidden_size:],
        )
    )


def moe_align_block_size_out_flaggems(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """Call FlagGems' raw kernel with vLLM's preallocated outputs."""
    if expert_map is not None:
        raise NotImplementedError(
            "FlagGems moe_align_block_size does not support expert_map"
        )

    from flag_gems import moe_align_block_size_triton

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )


__all__ = [
    "moe_align_block_size_out_flaggems",
    "silu_and_mul_out_flaggems",
]

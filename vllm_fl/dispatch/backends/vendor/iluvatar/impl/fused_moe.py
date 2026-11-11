# Copyright (c) 2026 BAAI. All rights reserved.
"""CoreX adapters for the pinned FlagGems-vllm MoE provider."""


def moe_align_block_size(
    topk_ids,
    block_size,
    num_experts,
    expert_map=None,
    pad_sorted_ids=False,
    ignore_invalid_experts=False,
):
    # CoreX exposes TLE imports, but cannot launch the cooperative grid barrier
    # used by the generic provider. Use its supported four-stage entry point,
    # rather than mutating HAS_TLE globally or changing the pinned dependency.
    # ignore_invalid_experts follows the existing FlagGems adapter contract.
    from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle

    return moe_align_block_size_no_tle(
        topk_ids, block_size, num_experts, expert_map, pad_sorted_ids
    )

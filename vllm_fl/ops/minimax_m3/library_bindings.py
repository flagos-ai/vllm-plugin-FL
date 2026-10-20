# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""M3 library bindings; source package and numerical preparation are explicit."""

import flag_gems
import torch


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle

    return moe_align_block_size_no_tle(
        topk_ids, block_size, num_experts, expert_map, pad_sorted_ids
    )


def invoke_fused_moe_triton_kernel(*args, **kwargs):
    from flaggems_vllm.ops.fused_moe import invoke_fused_moe_triton_kernel as invoke

    return invoke(*args, **kwargs)


def moe_sum(inp, out):
    from flaggems_vllm.ops.moe_sum import moe_sum as ordered_sum

    return ordered_sum(inp, out, reduction="sequential")


def grouped_topk(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    from flaggems_vllm.ops.grouped_topk import grouped_topk as select

    if scoring_func not in (0, 1):
        raise ValueError("Expected identity or sigmoid scoring")
    # Preserve the accepted materialized sigmoid and dtype before expert selection.
    if scoring_func == 1:
        scores = flag_gems.sigmoid(scores.float()).to(scores.dtype)
    return select(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func=0,
    )

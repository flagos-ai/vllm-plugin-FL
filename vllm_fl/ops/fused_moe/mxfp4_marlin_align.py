# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the existing four-stage aligner without speculative TLE launches.

On the pinned stack, TLE launch failures can invalidate CUDA capture before
its fallback executes. The four-stage path preserves the original alignment
contract and is also the eager fallback used after those TLE failures.
"""

import importlib


def align_mxfp4_tokens(gems, topk_ids, num_experts, policy):
    if (
        topk_ids.is_cuda
        and hasattr(gems, "_MXFP4AlignMode")
        and policy.align_mode == gems._MXFP4AlignMode.default
    ):
        align_module = importlib.import_module("flaggems_vllm.ops.moe_align_block_size")
        return align_module.moe_align_block_size_no_tle(
            topk_ids, block_size=policy.block_m, num_experts=num_experts
        )
    # Keep the policy's singleton/small-grouped paths; CPU tests may inject
    # a dependency with the same alignment contract.
    return gems._align_mxfp4_tokens(
        topk_ids, num_experts, policy.block_m, policy.align_mode
    )

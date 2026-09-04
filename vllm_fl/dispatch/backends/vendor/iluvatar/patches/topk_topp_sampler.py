# Copyright (c) 2026 BAAI. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

# vLLM's Triton topk_topp kernel fails to compile on iluvatar's corex Triton
# fork (triton < 3.3): the ternary-search kernel divides uint32 by int32,
# which the fork rejects for mixed signedness. Patch apply_top_k_top_p to
# always use the PyTorch fallback (same shape as the metax patch).
# TODO: remove once the minimum iluvatar triton is >= 3.3.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton

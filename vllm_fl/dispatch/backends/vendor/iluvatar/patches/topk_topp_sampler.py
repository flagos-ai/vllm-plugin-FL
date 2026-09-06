# Copyright (c) 2026 BAAI. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

# vLLM's Triton topk_topp kernel fails to compile on iluvatar's corex triton
# 3.1 fork (4.4.0 T path): the ternary-search kernel divides uint32 by int32,
# which the fork rejects for mixed signedness. Only that fork needs the
# PyTorch fallback — flagtree 3.6 (4.4.0 F) and corex 3.2 (4.5.0) compile the
# native kernel fine, so gate the patch on the triton version. Module
# __version__ is a bare number in this matrix; "3.1" uniquely identifies the
# corex 4.4.0 fork (flagtree reports 3.6, corex 4.5.0 reports 3.2).
# TODO: remove once the minimum iluvatar triton is >= 3.3.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


def _needs_pytorch_sampler() -> bool:
    """True only on the corex triton 3.1 fork (iluvatar 4.4.0 T path)."""
    try:
        import triton
    except Exception:
        return True  # unverifiable → keep the safe pytorch fallback
    return str(getattr(triton, "__version__", "")).startswith("3.1")


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton only on the
# corex 3.1 fork; leave the native path for flagtree 3.6 / corex 3.2.
if _needs_pytorch_sampler():
    topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton


# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). Patch apply_top_k_top_p so that
# top-p-only batches use FlagGems' sort-free top_p_mask_ kernel, and every
# other case (top-k present, unsupported layout, older FlagGems) keeps the
# PyTorch sort fallback.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from flag_gems.fused import top_p_mask_
except ImportError:  # FlagGems without the sort-free top-p op
    top_p_mask_ = None

# Mechanism marker: number of batches filtered by the sort-free kernel.
SORT_FREE_TOP_P_CALLS = 0


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    global SORT_FREE_TOP_P_CALLS
    if p is None and k is None:
        return logits
    if (
        k is None
        and top_p_mask_ is not None
        and logits.dim() == 2
        and logits.stride(1) == 1
    ):
        if SORT_FREE_TOP_P_CALLS == 0:
            logger.info("Sort-free top-p active: flag_gems.fused.top_p_mask_")
        SORT_FREE_TOP_P_CALLS += 1
        return top_p_mask_(logits, p)
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips vLLM's Triton kernel
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton


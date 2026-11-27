# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). Patch apply_top_k_top_p so it
# never takes that path: top-p-only batches (the default sampling of models
# whose generation_config sets top_p) use FlagGems' sort-free top_p_mask_;
# everything else keeps the PyTorch fallback.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    if k is None and logits.dtype == torch.float32 and logits.stride(-1) == 1:
        # Sorting the whole vocabulary per row dominated decode steps; the
        # FlagGems kernel finds the nucleus cut-off without sorting.
        from flag_gems.fused import top_p_mask_

        return top_p_mask_(logits, p)
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton

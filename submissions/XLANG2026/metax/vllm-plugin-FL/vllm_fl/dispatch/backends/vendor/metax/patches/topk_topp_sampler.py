# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). FlagGems now provides
# MetaX-compatible kernels:
#   - flag_gems.fused.top_k_top_p: sort-free top-k/top-p logits filtering
#   - flag_gems.fused.gumbel_max_sample: single-pass Gumbel-Max sampling that
#     replaces the eager softmax -> exponential-noise -> div -> argmax tail
# Both keep the PyTorch path as a safety net.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

try:
    from flag_gems.fused.top_k_top_p import (
        apply_top_k_top_p as _apply_top_k_top_p_flaggems,
    )
except Exception:  # FlagGems without the fused kernel: stay on PyTorch
    _apply_top_k_top_p_flaggems = None

try:
    from flag_gems.fused.gumbel_max_sample import (
        gumbel_max_sample as _gumbel_max_sample_flaggems,
    )
except Exception:
    _gumbel_max_sample_flaggems = None


def _apply_top_k_top_p_metax(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    if _apply_top_k_top_p_flaggems is not None:
        try:
            return _apply_top_k_top_p_flaggems(logits, k, p)
        except Exception:
            pass
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with the MetaX-compatible kernel
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_metax

_orig_forward_native = topk_topp_sampler.TopKTopPSampler.forward_native


def _forward_native_metax(self, logits, generators, k, p):
    """MetaX: fused Gumbel-Max tail instead of softmax+noise+div+argmax."""
    logits = topk_topp_sampler.apply_top_k_top_p(logits, k, p)
    logits_to_return = None
    if self.logprobs_mode == "processed_logits":
        logits_to_return = logits
    elif self.logprobs_mode == "processed_logprobs":
        logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
    if (
        _gumbel_max_sample_flaggems is not None
        and not generators
        and not self.use_fp64_gumbel
    ):
        try:
            return _gumbel_max_sample_flaggems(logits), logits_to_return
        except Exception:
            pass
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return (
        topk_topp_sampler.random_sample(probs, generators, self.use_fp64_gumbel),
        logits_to_return,
    )


topk_topp_sampler.TopKTopPSampler.forward_native = _forward_native_metax

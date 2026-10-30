# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# MetaX keeps vLLM's unsupported Triton filter disabled. Prefer the explicit
# FGV deterministic API when enabled; retain the original PyTorch fallback.
import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

from vllm_fl import envs as fl_envs


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        import flaggems_vllm

        apply_filter = getattr(flaggems_vllm, "apply_top_k_top_p", None)
        if apply_filter is not None:
            return apply_filter(logits, k, p)
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton

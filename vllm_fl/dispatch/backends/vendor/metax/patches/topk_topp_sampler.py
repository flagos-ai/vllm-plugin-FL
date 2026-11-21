# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). Patch apply_top_k_top_p
# to always use the PyTorch fallback.
# TODO: remove once FlagGems provides a MetaX-compatible topk_topp kernel.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


from vllm_fl.patches.topk_topp_sampler import patch_vllm_topk_topp_sampler

if not patch_vllm_topk_topp_sampler():
    def _apply_top_k_top_p_no_triton(
        logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
    ) -> torch.Tensor:
        if p is None and k is None:
            return logits
        return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)

    topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton

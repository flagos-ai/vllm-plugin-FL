# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX sampling: native fallback with scoped BF16-grid filtering."""

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.v1.sample.ops import topk_topp_sampler


def _use_counting_filter() -> bool:
    """Keep approximate filtering within the validated model/hardware scope."""
    config = get_current_vllm_config_or_none()
    if config is None or config.model_config is None:
        return False
    model = config.model_config
    hf = model.hf_config
    if hf.model_type != "qwen3_vl_moe" or model.dtype != torch.bfloat16:
        return False
    text = hf.text_config
    return (
        text.num_experts == 128
        and text.hidden_size == 2048
        and text.vocab_size == 151936
        and "C550" in torch.cuda.get_device_name().upper()
    )


def _supports_counting_filter(logits, k, p) -> bool:
    if k is None or p is None:
        return False
    return (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.shape == (64, 151936)
        and logits.is_contiguous()
        and k.shape == p.shape == (64,)
        and k.dtype in (torch.int32, torch.int64)
        and p.dtype == torch.float32
        and k.device == p.device == logits.device
        and k.is_contiguous() and p.is_contiguous()
    )


def _apply_top_k_top_p_no_triton(logits, k, p):
    if k is None and p is None:
        return logits
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


class MetaXTopKTopPSampler(topk_topp_sampler.TopKTopPSampler):
    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        super().__init__(logprobs_mode, use_fp64_gumbel)
        # Worker initialization supplies config; sampling need not retain it.
        self._use_counting = _use_counting_filter()

    def forward_native(self, logits, generators, k, p):
        if self._use_counting and _supports_counting_filter(logits, k, p):
            from ..impl.qwen3_vl_counting import apply_bf16_top_k_top_p
            logits = apply_bf16_top_k_top_p(logits, k, p)
            # Filtering is complete. Reuse upstream RNG and logprobs handling.
            return super().forward_native(logits, generators, None, None)
        return super().forward_native(logits, generators, k, p)


# Keep the existing MetaX fallback and adapt the sampler construction point.
# The upstream class and its methods remain unchanged.
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton
topk_topp_sampler.TopKTopPSampler = MetaXTopKTopPSampler

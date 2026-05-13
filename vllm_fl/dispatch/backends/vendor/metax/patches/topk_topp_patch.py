# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ---------------------------------------------------------------------------
# Patch: Replace the Triton-based topk/topp sampling kernel with a PyTorch
# native implementation on MetaX platform.
#
# The vLLM 0.19.0 `_topk_topp_kernel` in topk_topp_triton.py uses IR patterns
# that MetaX's Triton compiler cannot handle ("operand #0 does not dominate
# this use" during TTGIR PassManager). This patch provides an equivalent
# torch-native fallback so sampling works without --enforce-eager.
#
# Avoid torch.topk here. On MetaX, FlagGems may route torch.topk to a Triton
# kernel whose private memory request exceeds the default driver limit.
# ---------------------------------------------------------------------------

import torch

import vllm.v1.sample.ops.topk_topp_sampler as _topk_topp_sampler


def _apply_top_k_top_p_torch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Sort-based PyTorch implementation of combined top-k/top-p filtering.

    Semantics match the Triton kernel: for each row, mask out tokens that
    fall outside the top-k or top-p constraint by setting them to -inf.
    """
    if k is None and p is None:
        return logits

    vocab_size = logits.shape[-1]
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        k_long = k.to(torch.long)
        needs_topk = (k_long > 0) & (k_long < vocab_size)
        gather_idx = vocab_size - k_long.clamp(min=1, max=vocab_size)
        top_k_threshold = logits_sort.gather(1, gather_idx.unsqueeze(dim=1))
        top_k_mask = (logits_sort < top_k_threshold) & needs_topk.unsqueeze(dim=1)
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # Keep at least the largest token in each row.
        top_p_mask[:, -1] = False
        if p.ndim > 0:
            top_p_mask = top_p_mask & (p < 1.0).unsqueeze(dim=1)
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


# Monkey-patch: replace apply_top_k_top_p with torch-native version
_topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_torch

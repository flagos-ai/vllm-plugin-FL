# SPDX-License-Identifier: Apache-2.0
"""Preserve the native large-batch filter; specialize small K3 top-p sorts."""

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p as baseline_filter

from vllm_fl.dispatch import CachedOp

_stable_sort = CachedOp("kimi_k3_sampling_stable_sort")


def supports(logits, k, p):
    return (
        k is None
        and isinstance(p, torch.Tensor)
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.is_contiguous()
        and 1 <= logits.shape[0] < 8
        and logits.shape[1] == 163840
        and p.device == logits.device
        and p.dtype == torch.float32
        and p.shape == (logits.shape[0],)
        and p.is_contiguous()
    )


def apply_top_k_top_p(logits, k, p):
    if not supports(logits, k, p):
        return baseline_filter(logits, k, p)
    # Signed integer keys retain the pinned FG radix ordering, including zeros.
    bits = logits.view(torch.int32)
    keys = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key_values = torch.empty_like(keys)
    indices = torch.empty(keys.shape, dtype=torch.int64, device=keys.device)
    _stable_sort(keys, key_values, indices)
    values = logits.gather(-1, indices)
    probs = values.softmax(dim=-1)
    cdf = torch.cumsum(probs, dim=-1, out=probs)
    mask = cdf <= 1 - p.unsqueeze(1)
    mask[:, -1] = False
    values.masked_fill_(mask, -float("inf"))
    return logits.scatter_(dim=-1, index=indices, src=values)

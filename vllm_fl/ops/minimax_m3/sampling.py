# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""M3 filter composition; preserve FG ordering and the native sampling RNG."""
import torch

from vllm.logger import init_logger

from vllm_fl.dispatch import CachedOp, get_logger

logger = init_logger(get_logger(__name__).name)
_stable_sort = CachedOp("m3_sampling_stable_sort", retry_on_error=False)
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

def supports(logits, k, p):
    return (
        k is None and isinstance(p, torch.Tensor)
        and logits.is_cuda and logits.dtype == torch.float32
        and logits.ndim == 2 and logits.is_contiguous()
        and 1 <= logits.shape[0] <= 64 and logits.shape[1] == 200064
        and p.device == logits.device and p.dtype == torch.float32
        and p.shape == (logits.shape[0],) and p.is_contiguous()
    )

def apply_top_k_top_p(logits, k, p):
    if not supports(logits, k, p):
        return apply_top_k_top_p_pytorch(logits, k, p)
    logger.info_once("MiniMax M3 top-p filter uses native stable int32 sort")
    bits = logits.view(torch.int32)
    keys = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
    key_values = torch.empty_like(keys)
    indices = torch.empty(keys.shape, dtype=torch.int64, device=keys.device)
    # Signed bit keys preserve the pinned FG radix order, including signed zero.
    # Only sorting uses the native backend; existing ATen bindings handle the
    # key preparation, softmax/CDF and mask/scatter. RNG stays in vLLM.
    _stable_sort(keys, key_values, indices)
    values = logits.gather(-1, indices)
    probs = values.softmax(dim=-1)
    cdf = torch.cumsum(probs, dim=-1, out=probs)
    mask = cdf <= 1 - p.unsqueeze(1)
    mask[:, -1] = False
    values.masked_fill_(mask, -float("inf"))
    return logits.scatter_(dim=-1, index=indices, src=values)

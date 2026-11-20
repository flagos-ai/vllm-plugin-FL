# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""FG vision computation; vLLM classes and chunking stay in models/ops."""

import flag_gems
import torch


def vision_attention(query, key, value, cu_seqlens, max_seqlen, scale):
    assert query.ndim == 4 and query.shape[0] == 1 and query.dtype == torch.bfloat16
    assert key.shape == value.shape and key.shape[:2] == query.shape[:2]
    n = query.shape[1]
    cu = (
        cu_seqlens.to(device=query.device, dtype=torch.int32)
        if cu_seqlens is not None
        else torch.tensor([0, n], device=query.device, dtype=torch.int32)
    )
    maxlen = (
        int(max_seqlen.item())
        if isinstance(max_seqlen, torch.Tensor)
        else int(max_seqlen or n)
    )
    q = query.squeeze(0).contiguous()
    k = key.squeeze(0).contiguous()
    v = value.squeeze(0).contiguous()
    result = flag_gems.flash_attn_varlen_func(
        q,
        k,
        v,
        maxlen,
        cu,
        maxlen,
        cu,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
    )
    assert isinstance(result, torch.Tensor)
    return result.unsqueeze(0)


def _patch_embed(pixel_values, weight):
    import flag_gems

    assert pixel_values.ndim == 2 and weight.ndim == 5
    assert pixel_values.shape[1] == weight[0].numel()
    assert pixel_values.dtype == weight.dtype
    return flag_gems.mm(pixel_values, weight.flatten(1).t())

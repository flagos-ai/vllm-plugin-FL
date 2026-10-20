# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import flag_gems
import torch

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as _MMEncoderBase,
)

from vllm_fl.dispatch import CachedOp

vision_attention_op = CachedOp("m3_vision_attention")


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


class M3VisionAttentionFL(_MMEncoderBase):
    def __init__(self, num_heads, head_size, scale=None, num_kv_heads=None, prefix=""):
        torch.nn.Module.__init__(self)
        self.fp8_enabled = False
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.scale = head_size**-0.5 if scale is None else scale
        self.layer_name = prefix

    def forward(
        self, query, key, value, cu_seqlens=None, max_seqlen=None, sequence_lengths=None
    ):
        return vision_attention_op(
            query, key, value, cu_seqlens, max_seqlen, self.scale
        )

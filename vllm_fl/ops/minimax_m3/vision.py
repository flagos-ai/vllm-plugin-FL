# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as _MMEncoderBase,
)

from vllm_fl.dispatch import CachedOp

vision_attention_op = CachedOp("m3_vision_attention")


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

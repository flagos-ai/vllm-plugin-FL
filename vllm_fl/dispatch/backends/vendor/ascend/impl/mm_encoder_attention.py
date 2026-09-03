# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/v0.13.0/vllm_ascend/ops/triton/fused_gdn_gating.py
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch_npu
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention



class AscendMMEncoderAttention(MMEncoderAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MMEncoderAttention.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

    def reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def forward_oot(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor
        | None = None,  # Only used for Flash Attention
            sequence_lengths: torch.Tensor
        | None = None,  # Only used for FlashInfer CuDNN backend
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        # q, k, v: [b, s, head, head_dim] -> [b * s, head, head_dim]
        q, k, v = self.reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        # Pure-PyTorch scaled dot-product attention (matmul path).
        # Avoids _npu_flash_attention_unpad which can trigger DDR
        # out-of-range aicore errors on some model configurations.
        # q,k,v: [B*S, H, D] -> [B, H, S, D] for matmul
        q4d = q.view(bsz, -1, self.num_heads, self.head_size).transpose(1, 2)
        k4d = k.view(bsz, -1, self.num_heads, self.head_size).transpose(1, 2)
        v4d = v.view(bsz, -1, self.num_heads, self.head_size).transpose(1, 2)

        scale = self.head_size ** -0.5

        def _sdpa(qx, kx, vx):
            aw = torch.matmul(qx, kx.transpose(-2, -1)) * scale
            aw = torch.softmax(aw.float(), dim=-1).to(qx.dtype)
            return torch.matmul(aw, vx)

        if cu_seqlens is not None:
            # Block-diagonal (windowed) attention: vision encoders restrict
            # attention to within each cu_seqlens segment (window / image).
            # Computing full dense attention here scrambles spatial features
            # across window boundaries and corrupts image understanding.
            # Mirror vllm's torch_sdpa_wrapper: split along the sequence dim
            # and attend within each segment independently.
            lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            q_chunks = torch.split(q4d, lens, dim=2)
            k_chunks = torch.split(k4d, lens, dim=2)
            v_chunks = torch.split(v4d, lens, dim=2)
            outs = [
                _sdpa(qi, ki, vi)
                for qi, ki, vi in zip(q_chunks, k_chunks, v_chunks)
            ]
            context_layer = torch.cat(outs, dim=2)  # [B, H, S, D]
        else:
            context_layer = _sdpa(q4d, k4d, v4d)

        if is_reshaped:
            # [B, H, S, D] -> [B, S, H, D]
            context_layer = context_layer.transpose(1, 2).contiguous()
        else:
            # [B, H, S, D] -> [B, S, H*D]
            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.view(bsz, -1, self.num_heads * self.head_size)
        return context_layer


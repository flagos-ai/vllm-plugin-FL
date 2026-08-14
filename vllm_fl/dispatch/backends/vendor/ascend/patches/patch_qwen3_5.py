#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Copyright (c) 2026 BAAI. All rights reserved.
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
# mypy: ignore-errors

"""Ascend-specific patch for Qwen3.5/Qwen3.6 attention.

Replaces the upstream ``Qwen3NextAttention.forward`` with a fused
``triton_split_qkv_rmsnorm_mrope`` implementation on Ascend NPUs.
"""

import logging

import torch

from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

# Importing this module registers ``torch.ops.vllm.triton_split_qkv_rmsnorm_mrope``
# if it has not been registered by another backend (e.g. vllm-ascend).
from ..impl.linearnorm import split_qkv_rmsnorm_mrope  # noqa: F401

logger = logging.getLogger(__name__)


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)

        if "qwen3_5" in self.config.model_type:
            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)

            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
            )
        else:
            if self.attn_output_gate:
                q_gate, k, v = qkv.split(
                    [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
                )
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*orig_shape, -1)
                gate = gate.reshape(*orig_shape, -1)
            else:
                q, k, v = qkv.split(
                    [self.q_size, self.kv_size, self.kv_size], dim=-1
                )
                gate = None

            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.num_heads * self.head_dim
            )
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
                -1, self.num_kv_heads * self.head_dim
            )

            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


def patch_qwen3_5_attention() -> None:
    """Apply the Ascend Qwen3.5/Qwen3.6 attention patch."""
    Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
    logger.info("Patched Qwen3NextAttention for Ascend (split_qkv_rmsnorm_mrope)")

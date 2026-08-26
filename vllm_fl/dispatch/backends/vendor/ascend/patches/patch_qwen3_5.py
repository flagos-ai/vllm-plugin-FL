# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Qwen3.5/Qwen3.6 model patch from vLLM-Ascend 0.20.2."""

from __future__ import annotations

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models import qwen3_next as qwen3_next_module
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextSparseMoeBlock,
)

from vllm_fl.ascend_forward_context import _EXTRA_CTX


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
                original_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*original_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*original_shape, -1)
                gate = gate.reshape(*original_shape, -1)
            else:
                q, k, v = qkv.split(
                    [self.q_size, self.kv_size, self.kv_size], dim=-1
                )
            q = self.q_norm(
                q.view(-1, self.num_heads, self.head_dim)
            ).view(-1, self.num_heads * self.head_dim)
            k = self.k_norm(
                k.view(-1, self.num_kv_heads, self.head_dim)
            ).view(-1, self.num_kv_heads * self.head_dim)
            q, k = self.rotary_emb(positions, q, k)

        # The generic vLLM 0.20.2 Attention.forward builds its output from a
        # Python ``torch.Size``.  Under FULL_DECODE_ONLY eager-FX that size is
        # frozen to the one-token trace input, while prefill can later provide
        # 16K tokens.  BF16 Qwen3.5 has matching query/output widths, so derive
        # the buffer from the runtime query tensor and invoke the same unified
        # attention custom op directly.
        if self.attn.query_quant is not None:
            raise NotImplementedError(
                "Dynamic Qwen3.5 attention output currently supports BF16"
            )
        attn_output = torch.empty_like(q)
        torch.ops.vllm.unified_attention_with_output(
            q.reshape(-1, self.num_heads, self.head_dim),
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            v.reshape(-1, self.num_kv_heads, self.head_dim),
            attn_output.reshape(-1, self.num_heads, self.head_dim),
            self.attn.layer_name,
            kv_cache_dummy_dep=None,
        )
        if self.attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate)
        output[:], _ = self.o_proj(attn_output)


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

        if self.layer_idx == 0 and _EXTRA_CTX.flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            rows = (hidden_states.shape[0] + tp_size - 1) // tp_size
            attention_output = torch.empty(
                (rows, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            attention_output = torch.empty_like(hidden_states)

        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = attention_output

        if self.layer_scale:
            scale = self.attn_layer_scale.to(hidden_states.dtype)
            hidden_states = hidden_states * (
                scale[0] + 1 if hidden_states.ndim == 2 else scale + 1
            )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            scale = self.ffn_layer_scale.to(hidden_states.dtype)
            hidden_states = hidden_states * (
                scale[0] + 1 if hidden_states.ndim == 2 else scale + 1
            )
        return hidden_states, residual


def _dynamic_sparse_moe_forward(
    self: Qwen3NextSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Qwen3-Next MoE forward without a trace-time-frozen output view.

    In vLLM 0.20.2 the upstream implementation saves ``hidden_states.shape``
    as a Python tuple and uses it in the final ``view``.  The eager FX graph
    used by FULL_DECODE_ONLY is traced with one token, so that tuple becomes
    ``(1, hidden_size)`` and rejects a later chunked-prefill tensor.  The
    block's input and output are already two-dimensional, so the final view is
    unnecessary and can be removed entirely.
    """
    num_tokens = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

    if self.is_sequence_parallel:
        hidden_states = qwen3_next_module.sequence_parallel_chunk(
            hidden_states
        )

    if self.experts.is_internal_router:
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
        )
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

    if self.is_sequence_parallel:
        final_hidden_states = (
            qwen3_next_module.tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
        )
        final_hidden_states = final_hidden_states[:num_tokens]

    return final_hidden_states


def apply_qwen3_5_patch() -> None:
    Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
    Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
    Qwen3NextSparseMoeBlock.forward = _dynamic_sparse_moe_forward

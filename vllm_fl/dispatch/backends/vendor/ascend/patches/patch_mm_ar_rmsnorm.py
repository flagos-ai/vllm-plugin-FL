# Copyright (c) 2026 BAAI. All rights reserved.
"""Wire the mm+all-reduce+add+RMSNorm MC2 fusion into the *upstream* vLLM
Qwen3Next classes.

Qwen3.5/Qwen3.6 (vllm_fl/models/qwen3_5.py) imports Qwen3NextDecoderLayer /
Qwen3NextGatedDeltaNet / Qwen3NextAttention from
``vllm.model_executor.models.qwen3_next`` (upstream), NOT from the FL
vendored copy, so the eager fusion wiring must patch the upstream classes
(the first attempt only changed the vendored file and was never exercised
by the 35B path).

The patched forwards are behavior-identical to upstream unless
VLLM_FL_ENABLE_MM_AR_RMSNORM=1 (see
vllm_fl/dispatch/backends/vendor/ascend/impl/mm_allreduce_rmsnorm.py):
then the attention blocks return their pre-projection activation and the
decoder layer fuses projection + TP all-reduce + residual + RMSNorm above
MM_AR_RMSNORM_MIN_TOKENS tokens per step.
"""

import logging

import torch
from einops import rearrange
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
)

from ..impl.mm_allreduce_rmsnorm import (
    MM_AR_RMSNORM_MIN_TOKENS,
    fused_mm_allreduce_add_rmsnorm,
    mm_ar_rmsnorm_enabled,
)

logger = logging.getLogger(__name__)


def _gdn_forward(
    self,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
):
    """Upstream Qwen3NextGatedDeltaNet.forward + fusion early-return."""
    num_tokens = hidden_states.size(0)

    # Part 1: Input Projection
    projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
    projected_states_ba, _ = self.in_proj_ba(hidden_states)
    query, key, value, z, b, a = self.fix_query_key_value_ordering(
        projected_states_qkvz, projected_states_ba
    )
    query, key, value = map(
        lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
    )
    mixed_qkv = torch.cat((query, key, value), dim=-1)

    # Part 2: Core Attention (Custom Op)
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    torch.ops.vllm.gdn_attention_core(
        mixed_qkv,
        b,
        a,
        core_attn_out,
        self.prefix,
    )

    # Part 3: Output Projection
    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    if mm_ar_rmsnorm_enabled():
        # Skip out_proj: the caller fuses projection + all-reduce +
        # residual + RMSNorm via matmul_allreduce_add_rmsnorm.
        return core_attn_out.contiguous()
    output[:num_tokens], _ = self.out_proj(core_attn_out)
    return None


def _attention_forward(
    self,
    positions: torch.Tensor,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
):
    """Upstream Qwen3NextAttention.forward + fusion early-return."""
    qkv, _ = self.qkv_proj(hidden_states)

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
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

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

    if mm_ar_rmsnorm_enabled():
        # Skip o_proj: the caller fuses projection + all-reduce +
        # residual + RMSNorm via matmul_allreduce_add_rmsnorm.
        return attn_output.reshape(
            -1, self.num_heads * self.head_dim).contiguous()
    output[:], _ = self.o_proj(attn_output)
    return None


def _decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor = None,
    **kwargs: object,
):
    """Upstream Qwen3NextDecoderLayer.forward + fusion branch."""
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    self_attention_output = torch.empty_like(hidden_states)
    if self.layer_type == "linear_attention":
        attn_pre_proj = self.linear_attn(
            hidden_states=hidden_states,
            output=self_attention_output,
        )
    elif self.layer_type == "full_attention":
        attn_pre_proj = self.self_attn(
            hidden_states=hidden_states,
            output=self_attention_output,
            positions=positions,
        )
    else:
        raise ValueError("Invalid layer_type")

    if attn_pre_proj is not None:
        # Fusion path (VLLM_FL_ENABLE_MM_AR_RMSNORM=1): the attention block
        # returned the pre-projection activation.
        proj = (self.linear_attn.out_proj
                if self.layer_type == "linear_attention" else
                self.self_attn.o_proj)
        if (not self.layer_scale
                and attn_pre_proj.shape[0] > MM_AR_RMSNORM_MIN_TOKENS):
            # Fuse projection + TP all-reduce + residual add + RMSNorm.
            hidden_states, residual = fused_mm_allreduce_add_rmsnorm(
                attn_pre_proj,
                proj.weight,
                residual,
                1.0 + self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        # layer_scale or below-threshold M: project unfused.
        hidden_states, _ = proj(attn_pre_proj)
    else:
        hidden_states = self_attention_output

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype) + 1
            )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

    hidden_states = self.mlp(hidden_states)

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                f"shape must be the same {len(hidden_states.shape)}, "
                f"{len(self.ffn_layer_scale.shape)}"
            )
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype) + 1
            )

    return hidden_states, residual


def patch_mm_ar_rmsnorm() -> bool:
    """Patch upstream Qwen3Next classes with fusion-aware forwards."""
    Qwen3NextGatedDeltaNet.forward = _gdn_forward
    Qwen3NextAttention.forward = _attention_forward
    Qwen3NextDecoderLayer.forward = _decoder_layer_forward
    logger.info(
        "Patched upstream Qwen3Next GDN/Attention/DecoderLayer forwards "
        "for mm_allreduce_add_rmsnorm fusion (active only when "
        "VLLM_FL_ENABLE_MM_AR_RMSNORM=1)")
    return True

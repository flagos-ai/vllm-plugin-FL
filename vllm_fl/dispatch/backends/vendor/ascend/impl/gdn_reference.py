# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Tensor GDN operators with explicit weights, states and mutation contracts.

Convolution state uses [slots, channels, history]; recurrent state uses
[slots, value_heads, value_dim, key_dim]. Decode preserves padding slot zero.
Both operators update the supplied state/output buffers in place. They own
numerical computation, not model metadata, cache allocation or graph lifetime.
"""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_torch,
    causal_conv1d_update_torch,
)


def _prepare_qkv(x, num_key_heads, num_value_heads, head_k_dim, head_v_dim):
    q, k, v = torch.split(
        x, [num_key_heads * head_k_dim] * 2 + [num_value_heads * head_v_dim],
        dim=-1,
    )
    q = q.reshape(-1, num_key_heads, head_k_dim)
    k = k.reshape_as(q)
    v = v.reshape(-1, num_value_heads, head_v_dim)
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    repeats = num_value_heads // num_key_heads
    if repeats != 1:
        q = q.repeat_interleave(repeats, dim=1)
        k = k.repeat_interleave(repeats, dim=1)
    return q, k, v


def _gating(a, b, A_log, dt_bias, *, packed_beta):
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float(), threshold=20)
    beta = b.float().sigmoid()
    if packed_beta:
        beta = beta.to(b.dtype).float()
    return g, beta


def gdn_decode(
    x, a, b, indices, conv_state, ssm_state, output, *,
    conv_weights, conv_bias, activation, A_log, dt_bias,
    num_key_heads, num_value_heads, head_k_dim, head_v_dim, packed_beta,
):
    """Update one token per lane; indices <=0 are inactive padded lanes."""
    indices = indices.long()
    valid = indices > 0
    slots = indices.clamp_min(0)
    conv_old = conv_state.index_select(0, slots)
    conv_new = conv_old.clone()
    x = causal_conv1d_update_torch(
        x.unsqueeze(-1), conv_new, conv_weights, conv_bias, activation
    ).squeeze(-1)
    conv_state.index_copy_(
        0, slots, torch.where(valid[:, None, None], conv_new, conv_old)
    )
    q, k, v = _prepare_qkv(
        x, num_key_heads, num_value_heads, head_k_dim, head_v_dim
    )
    old = ssm_state.index_select(0, slots)
    state = torch.where(valid[:, None, None, None], old, 0).float()
    g, beta = _gating(a, b, A_log, dt_bias, packed_beta=packed_beta)
    state = state * g.exp()[..., None, None]
    delta = (v.float() - (state * k.unsqueeze(-2)).sum(-1)) * beta[..., None]
    state = state + delta.unsqueeze(-1) * k.unsqueeze(-2)
    result = (state * (q * head_k_dim**-0.5).unsqueeze(-2)).sum(-1)
    output.copy_(torch.where(valid[:, None, None], result, 0).to(output.dtype))
    ssm_state.index_copy_(
        0, slots, torch.where(valid[:, None, None, None], state.to(old.dtype), old)
    )


def gdn_prefill(
    x, a, b, cache_index, has_initial_state, conv_state, ssm_state, output, *,
    conv_weights, conv_bias, activation, A_log, dt_bias,
    num_key_heads, num_value_heads, head_k_dim, head_v_dim,
):
    """Update one prefill sequence and its supplied [H,V,K] recurrent row."""
    x = causal_conv1d_torch(
        x.T, conv_weights, conv_bias, conv_state,
        torch.tensor([0, x.shape[0]], device=x.device),
        cache_index, has_initial_state, activation,
    ).T
    q, k, v = _prepare_qkv(
        x, num_key_heads, num_value_heads, head_k_dim, head_v_dim
    )
    q, k = q.to(x.dtype), k.to(x.dtype)
    g, beta = _gating(a, b, A_log, dt_bias, packed_beta=False)
    initial = torch.where(has_initial_state[0], ssm_state, 0).float()
    initial = initial.transpose(-1, -2).unsqueeze(0)
    result, final = torch_chunk_gated_delta_rule(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        g.unsqueeze(0), beta.unsqueeze(0), initial_state=initial,
        output_final_state=True, use_qk_l2norm_in_kernel=False,
    )
    output.copy_(result.squeeze(0).to(output.dtype))
    ssm_state.copy_(final.squeeze(0).transpose(-1, -2).to(ssm_state.dtype))

# SPDX-License-Identifier: Apache-2.0
"""PyTorch KDA reference used when no accelerated implementation is selected."""

from __future__ import annotations

import torch


def safe_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    orig_shape = g.shape[:-1]
    g_2d = g.reshape(-1, g.shape[-1])
    heads = A_log.numel()
    if heads * head_k_dim != g_2d.shape[-1]:
        raise ValueError(
            "KDA gate hidden dimension does not match heads * head_k_dim: "
            f"{g_2d.shape[-1]} != {heads} * {head_k_dim}"
        )
    gate = g_2d.float().view(-1, heads, head_k_dim)
    if g_bias is not None:
        gate = gate + g_bias.float().reshape(heads, head_k_dim)
    gate = lower_bound * torch.sigmoid(
        torch.exp(A_log.float()).reshape(1, heads, 1) * gate
    )
    return gate.reshape(*orig_shape, heads, head_k_dim)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    return (x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True) + eps)).to(
        x.dtype
    )


def _recurrent_kda_sequence(q, k, v, gate, beta, state, scale):
    dtype = v.dtype
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    gate_f, beta_f = gate.float(), beta.float()
    state_f = state.float()
    output = torch.empty_like(v_f)
    for token in range(q.shape[0]):
        q_i = q_f[token] * scale
        k_i = k_f[token]
        v_i = v_f[token]
        gate_i = gate_f[token]
        beta_i = beta_f[token]
        state_f = state_f * torch.exp(gate_i)[..., None]
        residual = v_i - (k_i[..., None] * state_f).sum(dim=-2)
        state_f = state_f + torch.einsum(
            "hk,hv->hkv", beta_i[..., None] * k_i, residual
        )
        output[token] = torch.einsum("hk,hkv->hv", q_i, state_f)
    return output.to(dtype), state_f


def _sequence_ranges(batch, tokens, cu_seqlens):
    if cu_seqlens is None:
        return [(batch_id, 0, tokens) for batch_id in range(batch)]
    if batch != 1:
        raise ValueError("Variable-length KDA expects a flattened batch of 1")
    boundaries = cu_seqlens.detach().to("cpu", torch.int64).tolist()
    return [(0, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    inplace_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
):
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if scale is None:
        scale = key_dim**-0.5
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    ranges = _sequence_ranges(batch, tokens, cu_seqlens)
    if initial_state is None:
        states = torch.zeros(
            len(ranges),
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        states = initial_state if inplace_final_state else initial_state.clone()

    if ssm_state_indices is None:
        state_indices = list(range(len(ranges)))
    else:
        indices_cpu = ssm_state_indices.detach().to("cpu", torch.int64)
        state_indices = (
            indices_cpu.tolist()
            if indices_cpu.ndim == 1
            else indices_cpu[:, 0].tolist()
        )

    output = torch.empty_like(v)
    final_states = []
    for seq_id, (batch_id, start, end) in enumerate(ranges):
        state_id = int(state_indices[seq_id])
        seq_out, seq_state = _recurrent_kda_sequence(
            q[batch_id, start:end],
            k[batch_id, start:end],
            v[batch_id, start:end],
            gate[batch_id, start:end],
            beta[batch_id, start:end],
            states[state_id],
            float(scale),
        )
        output[batch_id, start:end] = seq_out
        states[state_id].copy_(seq_state)
        final_states.append(seq_state)

    if not output_final_state:
        return output, None
    if initial_state is not None and inplace_final_state:
        return output, initial_state
    if cu_seqlens is not None or batch == len(ranges):
        return output, torch.stack(final_states, dim=0)
    return output, states


def chunk_kda_with_safe_gate(
    q,
    k,
    v,
    raw_g,
    beta,
    A_log,
    g_bias,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    lower_bound=-5.0,
):
    gate = safe_kda_gate(
        raw_g.reshape(*raw_g.shape[:-2], -1),
        A_log,
        raw_g.shape[-1],
        g_bias=g_bias,
        lower_bound=lower_bound,
    )
    return recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["chunk_kda_with_safe_gate", "recurrent_kda", "safe_kda_gate"]

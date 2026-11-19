# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Tri Dao
#
# The normalization arithmetic is adapted from vLLM's FLA layernorm_guard,
# originally licensed under the MIT license, Copyright (c) 2024, Tri Dao.
"""Restricted PCP GDN output path; convolution and recurrence stay native."""

import torch
import triton
import triton.language as tl


@triton.jit
def _strided_gated_norm(
    X, Z, W, ROW_ORDER, Y, RSTD,
    M, N: tl.constexpr, HEADS: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr,
    ZS0: tl.constexpr, ZS1: tl.constexpr,
    EPS, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr,
    REORDER: tl.constexpr, ACTIVATION: tl.constexpr,
):
    # Preserve the native row tile, FP32 reduction, and gate arithmetic.
    rows = tl.program_id(0) * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    tokens, heads = rows // HEADS, rows % HEADS
    if REORDER:
        tokens = tl.load(ROW_ORDER + tokens, rows < M, other=0)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    valid = mask & (tokens[:, None] >= 0)
    x = tl.load(X + tokens[:, None] * XS0 + heads[:, None] * XS1
                + cols[None, :], valid, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + EPS)
    tl.store(RSTD + rows, rstd, rows < M)
    w = tl.load(W + cols, cols < N, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]
    z = tl.load(Z + tokens[:, None] * ZS0 + heads[:, None] * ZS1
                + cols[None, :], valid, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        y *= z * tl.sigmoid(z)
    elif ACTIVATION == "sigmoid":
        y *= tl.sigmoid(z)
    tl.store(Y + rows[:, None] * N + cols[None, :], y, mask)


def strided_gated_norm(norm, x, z, output_rows=None):
    """Read [token, head, dim] views; optionally emit a token permutation.

    The caller validates the native BF16 RMSNormGated configuration. Row order
    comes from ZigZagPlan and is a permutation plus optional -1 padding rows.
    Only addressing changes; use the native launch geometry and arithmetic.
    """
    from vllm.model_executor.layers.fla.ops.layernorm_guard import (
        calc_rows_per_block,
    )

    tokens, heads, width = x.shape
    tokens = tokens if output_rows is None else output_rows.numel()
    out = torch.empty((tokens, heads, width), dtype=x.dtype, device=x.device)
    rows = tokens * heads
    if rows:
        rows_per_block = calc_rows_per_block(rows, x.device)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(width)
        _strided_gated_norm[(triton.cdiv(rows, rows_per_block),)](
            x, z, norm.weight, output_rows, out, rstd,
            rows, width, heads, *x.stride()[:2], *z.stride()[:2], norm.eps,
            BLOCK_N=block, ROWS_PER_BLOCK=rows_per_block,
            REORDER=output_rows is not None, ACTIVATION=norm.activation,
            num_warps=min(max(block // 256, 1), 8),
        )
    return out


def _eligible(layer, hidden_states, output_rows):
    if (not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2 or hidden_states.shape[0] == 0):
        return False
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        GatedDeltaNetAttention,
    )

    if (type(layer) is not GatedDeltaNetAttention
            or getattr(layer._forward_method, "__func__", None)
            is not GatedDeltaNetAttention.forward_cuda
            or layer.gqa_interleaved_layout or hasattr(layer, "in_proj_qkv")
            or layer.head_v_dim != 128
            or hidden_states.shape[1] != layer.hidden_size
            or layer.out_proj.reduce_results):
        return False
    if any(type(proj.quant_method) is not UnquantizedLinearMethod
           for proj in (layer.in_proj_qkvz, layer.in_proj_ba, layer.out_proj)):
        return False
    norm = layer.norm
    if (type(norm) is not RMSNormGated
            or getattr(norm._forward_method, "__func__", None)
            is not RMSNormGated.forward_cuda
            or not norm.norm_before_gate or norm.group_size is not None
            or norm.bias is not None
            or norm.activation not in {"silu", "swish", "sigmoid"}
            or norm.weight.shape != (layer.head_v_dim,)
            or norm.weight.device != hidden_states.device
            or norm.weight.dtype not in {torch.bfloat16, torch.float32}
            or not norm.weight.is_contiguous()):
        return False
    return (output_rows is None or (
        output_rows.ndim == 1 and output_rows.numel() >= hidden_states.shape[0]
        and output_rows.device == hidden_states.device
        and output_rows.dtype == torch.long and output_rows.is_contiguous()
    ))


def try_prefill_gdn(layer, hidden_states, output_rows=None):
    """Return TP partials in the requested row order, or None before any work.

    Called only from the PCP prefill layer wrapper, with output reduction
    disabled. Keep input projection and the stateful custom op in original
    sequence order. Only the post-core, token-wise norm/projection is reordered.
    """
    if not _eligible(layer, hidden_states, output_rows):
        return None
    from vllm.utils.torch_utils import _encode_layer_name

    tokens = hidden_states.shape[0]
    mixed_qkvz, _ = layer.in_proj_qkvz(hidden_states)
    ba, _ = layer.in_proj_ba(hidden_states)
    qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    z_size = layer.value_dim // layer.tp_size
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.view(tokens, -1, layer.head_v_dim)
    b, a = ba.chunk(2, dim=-1)
    b, a = b.contiguous(), a.contiguous()
    # Native custom-op output is zero-initialized, including padded positions.
    # Preserve its convolution, recurrent state writes and output-copy contract.
    core = torch.zeros(
        (tokens, layer.num_v_heads // layer.tp_size, layer.head_v_dim),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    torch.ops.vllm.gdn_attention_core(
        mixed_qkv, b, a, core, _encode_layer_name(layer.prefix),
    )
    normalized = strided_gated_norm(layer.norm, core, z, output_rows)
    partial, _ = layer.out_proj(normalized.flatten(1))
    # Return the projection allocation directly; no full-output copy/packing.
    return partial

# SPDX-License-Identifier: Apache-2.0
"""Fused inverse-RoPE and INT8 activation quantization for DSV4 wo_a."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from vllm_fl.dispatch import resolve_op

DSV4_INV_ROPE_QUANT_INT8_OP = "deepseek_v4_inv_rope_quant_int8"
# The output projection runs inside the full-graph compiled model. Resolve the
# backend before tracing so Dynamo never enters OpManager's synchronization
# path; backend selection still goes through OpManager once at module import.
_dispatch_inv_rope_quant_int8 = resolve_op(DSV4_INV_ROPE_QUANT_INT8_OP)


@triton.jit
def _inv_rope_quant_int8_kernel(
    o,
    positions,
    cos_sin,
    o_q,
    o_scale,
    o_stride_t,
    o_stride_h,
    cs_stride_t,
    q_stride_t,
    q_stride_g,
    scale_stride_t,
    scale_stride_g,
    NUM_TOKENS,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, BLOCK_D)
    valid = offsets < GROUP_DIM
    head = offsets // HEAD_DIM
    dim = offsets % HEAD_DIM
    source_head = group * HEADS_PER_GROUP + head
    ptr = o + token * o_stride_t + source_head * o_stride_h + dim
    x = tl.load(ptr, mask=valid, other=0.0).to(tl.float32)

    rope_local = dim - NOPE_DIM
    is_rope = valid & (dim >= NOPE_DIM)
    partner = tl.load(
        ptr + tl.where((rope_local & 1) == 0, 1, -1),
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)
    pos = tl.load(positions + token, mask=token < NUM_TOKENS, other=0)
    cs = cos_sin + pos * cs_stride_t
    pair = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cs + pair, mask=is_rope, other=1.0)
    sin_v = tl.load(cs + HALF_ROPE + pair, mask=is_rope, other=0.0)
    inv = tl.where(
        (rope_local & 1) == 0,
        x * cos_v + partner * sin_v,
        x * cos_v - partner * sin_v,
    )
    x = tl.where(is_rope, inv, x).to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(
        tl.max(tl.where(valid, tl.abs(x), 0.0), axis=0), 1.0e-4
    )
    scale = absmax / 127.0
    rounded = x / scale + tl.where(x >= 0, 0.5, -0.5)
    quant = tl.clamp(rounded, -127.0, 127.0).to(tl.int8)
    tl.store(
        o_q + token * q_stride_t + group * q_stride_g + offsets,
        quant,
        mask=valid,
    )
    tl.store(
        o_scale + token * scale_stride_t + group * scale_stride_g,
        scale,
    )


def fused_inv_rope_quant_int8_triton(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return group-major symmetric INT8 activations and row scales."""
    tokens, heads, head_dim = o.shape
    assert heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    group_dim = heads_per_group * head_dim
    block_d = triton.next_power_of_2(group_dim)
    o_q = torch.empty(
        (n_groups, tokens, group_dim),
        dtype=torch.int8,
        device=o.device,
    )
    o_scale = torch.empty(
        (n_groups, tokens, 1),
        dtype=torch.float32,
        device=o.device,
    )
    _inv_rope_quant_int8_kernel[(tokens, n_groups)](
        o,
        positions,
        cos_sin_cache,
        o_q,
        o_scale,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        o_q.stride(1),
        o_q.stride(0),
        o_scale.stride(1),
        o_scale.stride(0),
        tokens,
        HEADS_PER_GROUP=heads_per_group,
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        HALF_ROPE=rope_dim // 2,
        GROUP_DIM=group_dim,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=3,
    )
    return o_q, o_scale


def fused_inv_rope_quant_int8(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch DSV4 inverse-RoPE INT8 quantization through OpManager."""
    return _dispatch_inv_rope_quant_int8(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
    )

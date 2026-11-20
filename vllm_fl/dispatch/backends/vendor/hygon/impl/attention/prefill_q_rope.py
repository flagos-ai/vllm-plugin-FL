# Copyright (c) 2026 BAAI. All rights reserved.

"""Large-M DV4 BF16 Indexer Q-RoPE, with the original rounding order.

Each CTA processes eight token/head rows instead of a single 128-D head.
Head and token strides remain explicit; the final feature dimension is dense.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _prefill_indexer_q_rope_bf16_kernel(
    positions, q, cos_sin, weights, output, weights_out,
    M: tl.constexpr, H: tl.constexpr,
    q_stride0, q_stride1, cos_stride, weights_stride,
    out_stride0, out_stride1, weights_out_stride,
    softmax_scale, head_scale,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    token, head = row // H, row % H
    valid = token < M
    position = tl.load(positions + token, valid, 0)
    q_base = q + token[:, None] * q_stride0 + head[:, None] * q_stride1
    out_base = output + token[:, None] * out_stride0 + head[:, None] * out_stride1

    nope_offsets = tl.arange(0, 64)
    nope = tl.load(q_base + nope_offsets[None, :], valid[:, None], 0)
    tl.store(out_base + nope_offsets[None, :], nope, valid[:, None])

    offsets = tl.arange(0, 32)
    cos = tl.load(
        cos_sin + position[:, None] * cos_stride + offsets[None, :],
        valid[:, None], 0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin + position[:, None] * cos_stride + offsets[None, :] + 32,
        valid[:, None], 0,
    ).to(tl.float32)
    even = tl.load(q_base + 64 + 2 * offsets[None, :], valid[:, None], 0).to(tl.float32)
    odd = tl.load(q_base + 65 + 2 * offsets[None, :], valid[:, None], 0).to(tl.float32)
    # Preserve the existing multiply/FMA and BF16 rounding boundaries.
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin
    tl.store(out_base + 64 + 2 * offsets[None, :], rotated_even.to(tl.bfloat16), valid[:, None])
    tl.store(out_base + 65 + 2 * offsets[None, :], rotated_odd.to(tl.bfloat16), valid[:, None])

    weight = tl.load(weights + token * weights_stride + head, valid, 0).to(tl.float32)
    weight *= softmax_scale
    weight *= head_scale
    tl.store(weights_out + token * weights_out_stride + head, weight, valid)


def prefill_indexer_q_rope_bf16_out(
    positions, q, cos_sin, weights, output, weights_out,
    softmax_scale, head_scale,
):
    """Caller validates BF16 [M,64,128], a 64-D RoPE table and dense last dims."""
    m, h, _ = q.shape
    _prefill_indexer_q_rope_bf16_kernel[(triton.cdiv(m * h, 8),)](
        positions, q, cos_sin, weights, output, weights_out, m, h,
        q.stride(0), q.stride(1), cos_sin.stride(0), weights.stride(0),
        output.stride(0), output.stride(1), weights_out.stride(0),
        softmax_scale, head_scale, 8, num_warps=2,
    )

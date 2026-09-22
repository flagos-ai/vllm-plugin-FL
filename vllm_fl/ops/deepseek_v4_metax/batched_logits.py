# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

"""Batched indexer logits kernel (Triton, MetaX).

Replaces the per-sequence Python loop in the graphsafe decode path of
vllm_fl/ops/sparse_attn_indexer.py: 64 x (_dequantize_fp8 + bf16_mqa_logits)
kernel launches become a single launch computing logits for the whole batch.

Semantics (must match bf16_mqa_logits per-row):
    logits[r, j] = sum_h w[r, h] * relu(q[r, h, :] . k_bf16[j, :])   for j < valid[r]
    logits[r, j] = -inf                                              for j >= valid[r]
    k_bf16[j, :] = fp8_dequant(k_fp8[j, :]) * scale[j]

Grid: (num_k_blocks, rows). Each program computes BC K-slot scores for one
sequence: tl.dot(q[HEADS, D], trans(K[BC, D])) -> [HEADS, BC], weighted relu-sum.

Capture-safety: shapes/grid depend only on (rows, NB); variable data flows via
GPU tensors (valid_lens). No .item(), no dynamic allocation inside.
"""

import triton
import triton.language as tl


@triton.jit
def _batched_mqa_logits_kernel(
    q_ptr,  # [rows, H, D] bf16
    k_fp8_ptr,  # [rows * NB * BC, D] fp8 (e4m3) gathered workspace
    k_scale_ptr,  # [rows * NB * BC] fp32
    w_ptr,  # [rows, H] fp32
    valid_ptr,  # [rows] int32
    out_ptr,  # [rows, NB * BC] fp32 logits
    stride_q_row,
    stride_out_row,
    NB: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(1)
    blk = tl.program_id(0)
    valid_len = tl.load(valid_ptr + row)

    h_offs = tl.arange(0, H)
    d_offs = tl.arange(0, D)
    # load q row: [H, D] bf16
    q = tl.load(q_ptr + row * stride_q_row + h_offs[:, None] * D + d_offs[None, :])

    k_offs = blk * BC + tl.arange(0, BC)
    k_mask = k_offs < valid_len
    # load fp8 K block: [BC, D] -> dequant to bf16
    kbase = (row * NB * BC + k_offs)[:, None] * D + d_offs[None, :]
    k_fp8 = tl.load(k_fp8_ptr + kbase, mask=k_mask[:, None], other=0.0)
    scale = tl.load(k_scale_ptr + row * NB * BC + k_offs, mask=k_mask, other=0.0)
    k_bf16 = k_fp8.to(tl.bfloat16) * scale[:, None].to(tl.bfloat16)

    # scores[H, BC] = q @ K^T
    scores = tl.dot(q, tl.trans(k_bf16))  # [H, BC] fp32
    w = tl.load(w_ptr + row * H + h_offs)  # [H] fp32
    scores = tl.maximum(scores, 0.0)
    row_scores = tl.sum(scores * w[:, None], axis=0)  # [BC] fp32

    out = tl.where(k_mask, row_scores, -float("inf"))
    tl.store(out_ptr + row * stride_out_row + k_offs, out)


def batched_mqa_logits(q, k_fp8, k_scale, weights, valid_lens, out, num_slots):
    """q [rows,H,D] bf16; k_fp8 [rows*num_slots,D] fp8; k_scale [rows*num_slots] fp32;
    weights [rows,H] fp32; valid_lens [rows] int; out [rows,num_slots] fp32."""
    rows, H, D = q.shape
    BC = 128
    NB = num_slots // BC
    assert num_slots % BC == 0
    _batched_mqa_logits_kernel[(NB, rows)](
        q,
        k_fp8,
        k_scale,
        weights,
        valid_lens,
        out,
        q.stride(0),
        out.stride(0),
        NB=NB,
        H=H,
        D=D,
        BC=BC,
        num_warps=4,
    )
    return out

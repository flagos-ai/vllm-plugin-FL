# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

"""
Triton implementation of flash_mla_with_kvcache for MLA attention.
Supports both sparse (FP8 KV cache + topk indices) and dense (paged attention) modes.
Only supports sm90 (Hopper) architecture.
"""

import dataclasses
import os
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


@dataclasses.dataclass
class FlashMLASchedMeta:
    """Stores tile scheduler metadata for FlashMLA."""

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]
        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False
    config: Optional[Config] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    """Returns an empty FlashMLASchedMeta instance."""
    return (FlashMLASchedMeta(), None)


@triton.autotune(
    configs=[
        triton.Config({"BK": 32, "BH": 16}, num_warps=4, num_stages=1),
        triton.Config({"BK": 16, "BH": 16}, num_warps=4, num_stages=1),
    ],
    key=["HQ", "DQK", "TOPK", "HAVE_ATTN_SINK", "HAVE_TOPK_LENGTH", "IS_FP8"],
)
@triton.jit
def _sparse_decode_kernel(
    q,
    kv,
    kv_scales,
    kv_rope,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kvn,
    stride_scales_n,
    stride_rope_n,
    stride_ib,
    stride_isq,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    IS_FP8: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """
    Sparse decode kernel with online softmax.
    Grid: (batch_size * seq_q * ceil(HQ / BH),)
    Each program handles BH heads for one (batch, seq_q) position.

    For FP8 mode:
      - kv: [num_tokens, 512] float8_e4m3fn (NoPE part)
      - kv_scales: [num_tokens, 4] float32 (per-128-element scales)
      - kv_rope: [num_tokens, 64] bfloat16 (RoPE part)
    For BF16 mode:
      - kv: [num_tokens, DQK] bfloat16 (full KV)
      - kv_scales, kv_rope: unused
    """
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_b = pid // (SQ * num_head_blocks)
    remainder = pid % (SQ * num_head_blocks)
    i_sq = remainder // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = remainder % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    kv_base = kv
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_b if HAVE_TOPK_LENGTH else 0
    o_base = output + i_b * stride_ob + i_sq * stride_osq + gbh_base * stride_oh
    l_base = lse + i_b * stride_lseb + gbh_base * stride_lseh + i_sq
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")
    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_ptr = BK * ck + offs_t
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        kv_ids = tl.where(mask_ids, kv_ids, 0)
        if IS_FP8:
            kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
            kv_fp8_0 = tl.load(kv_ptr, cache_modifier=".cg")
            kv_fp8_1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")
            scale0 = tl.load(kv_scales + kv_ids * stride_scales_n + 0)
            scale1 = tl.load(kv_scales + kv_ids * stride_scales_n + 1)
            scale2 = tl.load(kv_scales + kv_ids * stride_scales_n + 2)
            scale3 = tl.load(kv_scales + kv_ids * stride_scales_n + 3)
            mask_lo = offs_d[:, None] < 128
            kv_blk0 = tl.where(
                mask_lo,
                kv_fp8_0.to(tl.float32) * scale0[None, :],
                kv_fp8_0.to(tl.float32) * scale1[None, :],
            ).to(tl.bfloat16)
            kv_blk1 = tl.where(
                mask_lo,
                kv_fp8_1.to(tl.float32) * scale2[None, :],
                kv_fp8_1.to(tl.float32) * scale3[None, :],
            ).to(tl.bfloat16)
        else:
            kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
            kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            if IS_FP8:
                rope_ptr = kv_rope + offs_td[:, None] + kv_ids[None, :] * stride_rope_n
                tkv_blk = tl.load(rope_ptr, cache_modifier=".cg")
            else:
                tkv_ptr = kv_base + DP + offs_td[:, None] + kv_ids[None, :] * stride_kvn
                tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h * stride_lseh, lse_out)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp
    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


@triton.autotune(
    configs=[triton.Config({"BK": 16, "BH": 16}, num_warps=2, num_stages=1)],
    key=[
        "HQ",
        "TOPK",
        "EXTRA_TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
        "HAVE_EXTRA",
        "HAVE_EXTRA_TOPK_LENGTH",
    ],
)
@triton.jit
def _sparse_decode_model1_kernel(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    attn_sink,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    SQ,
    HQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS,
    EXTRA_NUM_BLOCKS,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    HAVE_EXTRA: tl.constexpr,
    HAVE_EXTRA_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_b = pid // (SQ * num_head_blocks)
    remainder = pid % (SQ * num_head_blocks)
    i_sq = remainder // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = remainder % num_head_blocks
    gbh_base = i_gbh * BH
    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_b if HAVE_TOPK_LENGTH else 0
    extra_topk_length_ptr = extra_topk_length + i_b if HAVE_EXTRA_TOPK_LENGTH else 0
    o_base = output + i_b * stride_ob + i_sq * stride_osq + gbh_base * stride_oh
    l_base = lse + i_b * stride_lseb + gbh_base * stride_lseh + i_sq
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BK)
    offs_rope = tl.arange(0, ROPE)
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < NOPE - BDP,
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + offs_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )
    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_offs = BK * ck + offs_t
        t_msk = t_offs < topk_len
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < NOPE - BDP,
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max
    if HAVE_EXTRA:
        extra_topk_len = (
            tl.load(extra_topk_length_ptr) if HAVE_EXTRA_TOPK_LENGTH else EXTRA_TOPK
        )
        ENK = tl.cdiv(extra_topk_len, BK)
        for ck in range(ENK):
            t_offs = BK * ck + offs_t
            t_msk = t_offs < extra_topk_len
            kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
            block_ids = kv_ids // EXTRA_PAGE_SIZE
            rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
            valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
            block_ids = tl.where(valid_ids, block_ids, 0)
            rel_ids = tl.where(valid_ids, rel_ids, 0)
            token_base = (
                extra_kv
                + block_ids.to(tl.int64) * stride_extra_kv_block
                + rel_ids * TOKEN_DATA_BYTES
            )
            scale_base = (
                extra_kv
                + block_ids.to(tl.int64) * stride_extra_kv_block
                + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
                + rel_ids * SCALE_BYTES
            )
            kv_fp8_0_u8 = tl.load(
                token_base[None, :] + offs_d[:, None],
                mask=valid_ids[None, :],
                other=0,
                cache_modifier=".cg",
            )
            kv_fp8_1_u8 = tl.load(
                token_base[None, :] + (BDP + offs_d[:, None]),
                mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
                other=0,
                cache_modifier=".cg",
            )
            scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
            scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
            scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
            scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
            scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
            scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
            scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
            scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_0 = tl.where(
                offs_d[:, None] < 64,
                scale0[None, :],
                tl.where(
                    offs_d[:, None] < 128,
                    scale1[None, :],
                    tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
                ),
            )
            kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
            kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_1 = tl.where(
                offs_d[:, None] < 64,
                scale4[None, :],
                tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
            )
            nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
            rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
            rope_blk = tl.load(
                rope_ptr[None, :] + offs_rope[:, None],
                mask=valid_ids[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            kv_blk1 = tl.where(
                offs_d[:, None] < NOPE - BDP,
                nope_tail,
                tl.load(
                    rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                    mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                    other=0.0,
                    cache_modifier=".cg",
                ),
            )
            qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
            qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
            qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
            qk *= sm_scale
            qk = tl.where(valid_ids[None, :], qk, float("-inf"))
            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp(qk - new_max[:, None])
            sum_qk = tl.sum(exp_qk, axis=1)
            alpha = tl.math.exp(max_log - new_max)
            sum_exp = sum_exp * alpha + sum_qk
            acc0 = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk0.trans(),
                acc0 * alpha[:, None],
                out_dtype=tl.float32,
            )
            acc1 = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk1.trans(),
                acc1 * alpha[:, None],
                out_dtype=tl.float32,
            )
            max_log = new_max
    valid_mask = max_log != float("-inf")
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h * stride_lseh, lse_out)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp
    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


@triton.jit
def _partial_model1_kernel(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    part_max,
    part_sum,
    part_acc,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    SQ: tl.constexpr,
    HQc: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    EXTRA_NUM_BLOCKS: tl.constexpr,
    SPLITS: tl.constexpr,
    BKc: tl.constexpr,
    BHc: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQc + BHc - 1) // BHc
    pid = tl.program_id(0)
    i_split = pid % SPLITS
    base_pid = pid // SPLITS
    i_b = base_pid // (SQ * num_head_blocks)
    rem = base_pid % (SQ * num_head_blocks)
    i_sq = (rem // num_head_blocks).to(tl.int64)
    i_gbh = rem % num_head_blocks
    gbh_base = i_gbh * BHc
    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8
    offs_h = tl.arange(0, BHc)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BKc)
    offs_rope = tl.arange(0, ROPE)
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < NOPE - BDP,
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + offs_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )
    max_log = tl.full([BHc], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BHc], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BHc, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BHc, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length + i_b)
    start = topk_len * i_split // SPLITS
    end = topk_len * (i_split + 1) // SPLITS
    seg_len = end - start
    NK = tl.cdiv(seg_len, BKc)
    for ck in range(NK):
        t_offs = start + BKc * ck + offs_t
        t_msk = t_offs < end
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < NOPE - BDP,
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max
    extra_len = tl.load(extra_topk_length + i_b)
    estart = extra_len * i_split // SPLITS
    eend = extra_len * (i_split + 1) // SPLITS
    eseg = eend - estart
    ENK = tl.cdiv(eseg, BKc)
    for ck in range(ENK):
        t_offs = estart + BKc * ck + offs_t
        t_msk = t_offs < eend
        kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // EXTRA_PAGE_SIZE
        rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < NOPE - BDP,
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max
    base = (i_b * SPLITS + i_split) * HQc + gbh_base
    tl.store(part_max + base + offs_h, max_log)
    tl.store(part_sum + base + offs_h, sum_exp)
    acc_base = ((i_b * SPLITS + i_split) * HQc + gbh_base) * 512
    tl.store(part_acc + acc_base + offs_h[:, None] * 512 + offs_d[None, :], acc0)
    tl.store(
        part_acc + acc_base + offs_h[:, None] * 512 + (BDP + offs_d[None, :]), acc1
    )


@triton.jit
def _partial_model1_kernel_halfd(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    part_max,
    part_sum,
    part_acc,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    SQ: tl.constexpr,
    HQc: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    EXTRA_NUM_BLOCKS: tl.constexpr,
    SPLITS: tl.constexpr,
    BKc: tl.constexpr,
    BHc: tl.constexpr,
    V_START: tl.constexpr,
    WRITE_STATS: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQc + BHc - 1) // BHc
    pid = tl.program_id(0)
    i_split = pid % SPLITS
    base_pid = pid // SPLITS
    i_b = base_pid // (SQ * num_head_blocks)
    rem = base_pid % (SQ * num_head_blocks)
    i_sq = (rem // num_head_blocks).to(tl.int64)
    i_gbh = rem % num_head_blocks
    gbh_base = i_gbh * BHc
    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8
    offs_h = tl.arange(0, BHc)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BKc)
    offs_rope = tl.arange(0, ROPE)
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < NOPE - BDP,
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + offs_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )
    max_log = tl.full([BHc], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BHc], 0.0, dtype=tl.float32)
    acc = tl.zeros([BHc, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length + i_b)
    start = topk_len * i_split // SPLITS
    end = topk_len * (i_split + 1) // SPLITS
    seg_len = end - start
    NK = tl.cdiv(seg_len, BKc)
    for ck in range(NK):
        t_offs = start + BKc * ck + offs_t
        t_msk = t_offs < end
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < NOPE - BDP,
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        if V_START == 0:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk0.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        else:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk1.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        max_log = new_max
    extra_len = tl.load(extra_topk_length + i_b)
    estart = extra_len * i_split // SPLITS
    eend = extra_len * (i_split + 1) // SPLITS
    eseg = eend - estart
    ENK = tl.cdiv(eseg, BKc)
    for ck in range(ENK):
        t_offs = estart + BKc * ck + offs_t
        t_msk = t_offs < eend
        kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // EXTRA_PAGE_SIZE
        rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < NOPE - BDP,
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        if V_START == 0:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk0.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        else:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk1.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        max_log = new_max
    if WRITE_STATS:
        base = (i_b * SPLITS + i_split) * HQc + gbh_base
        tl.store(part_max + base + offs_h, max_log)
        tl.store(part_sum + base + offs_h, sum_exp)
    acc_base = ((i_b * SPLITS + i_split) * HQc + gbh_base) * 512
    tl.store(
        part_acc + acc_base + offs_h[:, None] * 512 + (V_START + offs_d[None, :]), acc
    )


@triton.jit
def _partial_model1_kernel_halfd_qkfold_blockdiag(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    part_max,
    part_sum,
    part_acc,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    SQ: tl.constexpr,
    HQc: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    EXTRA_NUM_BLOCKS: tl.constexpr,
    SPLITS: tl.constexpr,
    BKc: tl.constexpr,
    BHc: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQc + BHc - 1) // BHc
    pid = tl.program_id(0)
    i_split = pid % SPLITS
    base_pid = pid // SPLITS
    i_b = base_pid // (SQ * num_head_blocks)
    rem = base_pid % (SQ * num_head_blocks)
    i_sq = (rem // num_head_blocks).to(tl.int64)
    i_gbh = rem % num_head_blocks
    gbh_base = i_gbh * BHc
    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8
    offs_h = tl.arange(0, BHc)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BKc)
    offs_rope = tl.arange(0, ROPE)
    real_h = offs_h % 8
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    q_ptr = q_base + real_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < NOPE - BDP,
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + real_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )
    max_log = tl.full([BHc], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BHc], 0.0, dtype=tl.float32)
    acc = tl.zeros([BHc, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length + i_b)
    start = topk_len * i_split // SPLITS
    end = topk_len * (i_split + 1) // SPLITS
    seg_len = end - start
    NK = tl.cdiv(seg_len, BKc)
    for ck in range(NK):
        t_offs = start + BKc * ck + offs_t
        t_msk = t_offs < end
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < NOPE - BDP,
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        prob = exp_qk.to(tl.bfloat16)
        zero_prob = tl.zeros([BHc, BKc], dtype=tl.bfloat16)
        prob_lo = tl.where(offs_h[:, None] < 8, prob, zero_prob)
        prob_hi = tl.where(offs_h[:, None] >= 8, prob, zero_prob)
        prob_pair = tl.reshape(tl.join(prob_lo, prob_hi), [BHc, BKc * 2])
        kv_pair = tl.reshape(tl.join(kv_blk0, kv_blk1), [BDP, BKc * 2])
        acc = tl.dot(
            prob_pair, kv_pair.trans(), acc * alpha[:, None], out_dtype=tl.float32
        )
        max_log = new_max
    extra_len = tl.load(extra_topk_length + i_b)
    estart = extra_len * i_split // SPLITS
    eend = extra_len * (i_split + 1) // SPLITS
    eseg = eend - estart
    ENK = tl.cdiv(eseg, BKc)
    for ck in range(ENK):
        t_offs = estart + BKc * ck + offs_t
        t_msk = t_offs < eend
        kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // EXTRA_PAGE_SIZE
        rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < NOPE - BDP),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < NOPE - BDP,
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= NOPE - BDP),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        prob = exp_qk.to(tl.bfloat16)
        zero_prob = tl.zeros([BHc, BKc], dtype=tl.bfloat16)
        prob_lo = tl.where(offs_h[:, None] < 8, prob, zero_prob)
        prob_hi = tl.where(offs_h[:, None] >= 8, prob, zero_prob)
        prob_pair = tl.reshape(tl.join(prob_lo, prob_hi), [BHc, BKc * 2])
        kv_pair = tl.reshape(tl.join(kv_blk0, kv_blk1), [BDP, BKc * 2])
        acc = tl.dot(
            prob_pair, kv_pair.trans(), acc * alpha[:, None], out_dtype=tl.float32
        )
        max_log = new_max
    base = (i_b * SPLITS + i_split) * HQc + gbh_base
    tl.store(part_max + base + offs_h, max_log)
    tl.store(part_sum + base + offs_h, sum_exp)
    acc_base = ((i_b * SPLITS + i_split) * HQc + gbh_base) * 512
    v_half = (offs_h >= 8).to(tl.int64) * BDP
    tl.store(
        part_acc + acc_base + real_h[:, None] * 512 + v_half[:, None] + offs_d[None, :],
        acc,
    )


@triton.jit
def _reduce_split_kernel(
    part_max,
    part_sum,
    part_acc,
    attn_sink,
    out,
    lse,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    Bsz: tl.constexpr,
    HQc: tl.constexpr,
    SPLITS: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    dblock = tl.program_id(1)
    h = pid % HQc
    b = pid // HQc
    offs_d = tl.arange(0, D_BLOCK)
    d0 = dblock * D_BLOCK + offs_d
    m = tl.full((), float("-inf"), dtype=tl.float32)
    for s in range(SPLITS):
        v = tl.load(part_max + (b * SPLITS + s) * HQc + h)
        m = tl.maximum(m, v)
    gsum = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([D_BLOCK], dtype=tl.float32)
    for s in range(SPLITS):
        pm = tl.load(part_max + (b * SPLITS + s) * HQc + h)
        ps = tl.load(part_sum + (b * SPLITS + s) * HQc + h)
        w = tl.math.exp(pm - m)
        gsum += ps * w
        pa = tl.load(part_acc + ((b * SPLITS + s) * HQc + h) * 512 + d0)
        acc += pa * w
    orig_lse = m + tl.math.log(gsum)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + h)
        factor = tl.math.exp(m) / (tl.math.exp(orig_lse) + tl.math.exp(sink))
    else:
        factor = 1.0 / gsum
    vals = acc * factor
    tl.store(
        out + b * stride_ob + 0 * stride_osq + h * stride_oh + d0, vals.to(tl.bfloat16)
    )
    if dblock == 0:
        tl.store(lse + b * stride_lseb + h * stride_lseh + 0, orig_lse)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 16}, num_warps=4, num_stages=1),
    ],
    key=["HQ", "DQK", "HAVE_CAUSAL"],
)
@triton.jit
def _dense_decode_kernel(
    Q_ptr,
    stride_q_b,
    stride_q_sq,
    stride_q_h,
    KV_cache,
    stride_kv_bs,
    Block_table,
    stride_bt_b,
    Seq_lens,
    Out,
    stride_o_b,
    stride_o_sq,
    stride_o_h,
    LSE,
    stride_lse_b,
    stride_lse_h,
    sm_scale,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAVE_CAUSAL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Dense decode kernel with paged attention and online softmax.
    Grid: (ceil(HQ / BLOCK_H), batch_size * seq_q)
    """
    pid_h_block = tl.program_id(0)
    pid_b_sq = tl.program_id(1)
    i_b = pid_b_sq // SQ
    i_sq = pid_b_sq % SQ
    cur_head = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_head = cur_head < HQ
    offs_d_nope = tl.arange(0, HEAD_DIM_V)
    offs_q_nope = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_nope[None, :]
    )
    q_nope = tl.load(Q_ptr + offs_q_nope, mask=mask_head[:, None], other=0.0)
    offs_d_pe = tl.arange(HEAD_DIM_V, DQK)
    offs_q_pe = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_pe[None, :]
    )
    q_pe = tl.load(Q_ptr + offs_q_pe, mask=mask_head[:, None], other=0.0)
    e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)
    cur_batch_seq_len = tl.load(Seq_lens + i_b)
    Block_table += i_b * stride_bt_b
    offs_n = tl.arange(0, BLOCK_N)
    loop_time = cur_batch_seq_len // BLOCK_N
    remainder = cur_batch_seq_len % BLOCK_N
    for i in range(0, loop_time):
        kv_page_number = tl.load(Block_table + offs_n // PAGE_SIZE)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c)
        k_c = tl.trans(v_c)
        qk = tl.dot(q_nope, k_c)
        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe)
        qk = tl.dot(q_pe, k_pe, acc=qk)
        qk *= sm_scale
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        offs_n += BLOCK_N
    if remainder:
        mask_kvsplit = offs_n < cur_batch_seq_len
        kv_page_number = tl.load(
            Block_table + offs_n // PAGE_SIZE, mask=mask_kvsplit, other=0
        )
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c, mask=mask_kvsplit[:, None], other=0.0)
        k_c = tl.trans(v_c)
        qk = tl.dot(q_nope, k_c)
        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)
        qk = tl.dot(q_pe, k_pe, acc=qk)
        qk *= sm_scale
        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    offs_o = (
        i_b * stride_o_b
        + i_sq * stride_o_sq
        + cur_head[:, None] * stride_o_h
        + offs_d_nope[None, :]
    )
    tl.store(
        Out + offs_o,
        (acc / e_sum[:, None]).to(Out.dtype.element_ty),
        mask=mask_head[:, None],
    )
    lse_val = e_max + tl.math.log(e_sum)
    lse_offset = i_b * stride_lse_b + cur_head * stride_lse_h + i_sq
    tl.store(LSE + lse_offset, lse_val, mask=mask_head)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of flash_mla_with_kvcache.
    Functionally equivalent to the CUDA implementation.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v)
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32
    """
    sched_meta = tile_scheduler_metadata
    assert isinstance(sched_meta, FlashMLASchedMeta)
    assert num_splits is None
    assert q.ndim == 4
    assert k_cache.ndim == 4
    topk = indices.shape[-1] if indices is not None else None
    extra_k_page_block_size = (
        extra_k_cache.shape[1] if extra_k_cache is not None else None
    )
    extra_topk_val = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if not sched_meta.have_initialized:
        if indices is not None:
            assert not causal, "causal must be False when sparse attention is enabled"
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],
            causal,
            is_fp8_kvcache,
            topk,
            extra_k_page_block_size,
            extra_topk_val,
        )
    else:
        helper_msg = " Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], (
            "sched_meta.config.b must be equal to batch_size." + helper_msg
        )
        assert sched_meta.config.s_q == q.shape[1], (
            "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        )
        assert sched_meta.config.h_q == q.shape[2], (
            "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        )
        assert sched_meta.config.page_block_size == k_cache.shape[1], (
            "sched_meta.config.page_block_size must be equal to page_block_size."
            + helper_msg
        )
        assert sched_meta.config.h_k == k_cache.shape[2], (
            "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        )
        assert sched_meta.config.causal == causal, (
            "sched_meta.config.causal must be equal to causal." + helper_msg
        )
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, (
            "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache."
            + helper_msg
        )
        assert sched_meta.config.topk == topk, (
            "sched_meta.config.topk must be equal to the last dim of indices."
            + helper_msg
        )
        assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, (
            "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache."
            + helper_msg
        )
        assert sched_meta.config.extra_topk == extra_topk_val, (
            "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache."
            + helper_msg
        )
    (batch_size, seq_q, num_heads_q, head_dim_k) = q.shape
    num_heads_k = k_cache.shape[2]
    if out is None:
        out = torch.empty(
            (batch_size, seq_q, num_heads_q, head_dim_v), dtype=q.dtype, device=q.device
        )
    else:
        assert out.shape == (batch_size, seq_q, num_heads_q, head_dim_v)
        assert out.dtype == q.dtype
        assert out.device == q.device
        assert out.stride(-1) == 1
    lse = torch.empty(
        (batch_size, num_heads_q, seq_q), dtype=torch.float32, device=q.device
    )
    if indices is not None:
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True for sparse attention"
        assert num_heads_k == 1, (
            "Currently only MQA (h_kv == 1) is supported for sparse decoding"
        )
        assert head_dim_v == 512, "Only head_size_v == 512 is supported"
        assert num_heads_q in (16, 64, 128), "Only h_q == 16, 64 or 128 is supported"
        assert head_dim_k in (512, 576), (
            "Only head_size_k == 512 or 576 is supported for sparse decoding"
        )
        assert q.dtype == torch.bfloat16
        assert k_cache.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)
        assert topk is not None and topk > 0
        assert topk % 64 == 0, "topk must be divisible by 64"
        assert indices.ndim == 3 and indices.shape[:2] == (batch_size, seq_q)
        assert indices.dtype == torch.int32
        assert indices.stride(-1) == 1
        if topk_length is not None:
            assert topk_length.shape == (batch_size,)
            assert topk_length.dtype == torch.int32
            assert topk_length.is_contiguous()
        if attn_sink is not None:
            assert attn_sink.shape == (num_heads_q,)
            assert attn_sink.dtype == torch.float32
        if extra_k_cache is not None:
            assert extra_indices_in_kvcache is not None, (
                "extra_indices_in_kvcache must be provided when extra_k_cache is provided"
            )
            assert extra_k_cache.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)
        else:
            assert extra_indices_in_kvcache is None, (
                "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided"
            )
            assert extra_topk_length is None, (
                "extra_topk_length must not be provided when extra_k_cache is not provided"
            )
        if extra_indices_in_kvcache is not None:
            assert extra_indices_in_kvcache.ndim == 3
            assert extra_indices_in_kvcache.shape[:2] == (batch_size, seq_q)
            assert extra_indices_in_kvcache.dtype == torch.int32
            assert extra_indices_in_kvcache.stride(-1) == 1
            assert extra_indices_in_kvcache.shape[-1] % 64 == 0
        if extra_topk_length is not None:
            assert extra_topk_length.shape == (batch_size,)
            assert extra_topk_length.dtype == torch.int32
            assert extra_topk_length.is_contiguous()
        if head_dim_k == 576:
            assert k_cache.shape[-1] == 656, (
                "V32 sparse FP8 cache must use 656 bytes per token"
            )
            assert k_cache.stride(1) == 656, (
                "The whole block must be contiguous for V32 KV cache"
            )
            assert topk_length is None, "V3.2/V32 does not support dynamic topk length"
            assert extra_k_cache is None, "V3.2/V32 does not support extra KV cache"
            assert extra_indices_in_kvcache is None, (
                "V3.2/V32 does not support extra indices"
            )
            assert extra_topk_length is None, (
                "V3.2/V32 does not support extra topk length"
            )
        else:
            assert k_cache.shape[-1] == 584, (
                "MODEL1 sparse FP8 cache must use 584 bytes per token"
            )
            assert k_cache.stride(1) == 584, (
                "The whole block must be contiguous for MODEL1 KV cache"
            )
            if extra_k_cache is not None:
                assert extra_k_cache.ndim == 4
                assert extra_k_cache.shape[2] == 1
                assert extra_k_cache.shape[-1] == 584
                assert extra_k_cache.stride(1) == 584
        _sparse_decode_dispatch(
            q,
            k_cache,
            indices,
            out,
            lse,
            attn_sink,
            topk_length,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            batch_size,
            seq_q,
            num_heads_q,
            head_dim_k,
            head_dim_v,
            topk,
            k_cache.shape[1],
            softmax_scale,
            is_fp8_kvcache,
        )
    else:
        assert (
            attn_sink is None
            and extra_k_cache is None
            and (extra_indices_in_kvcache is None)
            and (topk_length is None)
            and (extra_topk_length is None)
        ), (
            "indices, attn_sink, extra_k_cache, extra_indices_in_kvcache, topk_length and extra_topk_length must be None when dense attention is used."
        )
        assert block_table is not None and cache_seqlens is not None, (
            "block_table and cache_seqlens must be provided when dense attention is used."
        )
        assert num_heads_k == 1, "Only num_heads_k == 1 is supported for dense MLA"
        if seq_q > 1 and causal:
            raise NotImplementedError(
                "causal dense attention with seq_q > 1 is not implemented"
            )
        _dense_decode_dispatch(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            out,
            lse,
            batch_size,
            seq_q,
            num_heads_q,
            head_dim_k,
            head_dim_v,
            k_cache.shape[1],
            softmax_scale,
            causal,
        )
    return (out, lse)


_SPLIT_TOPK_BUF_CACHE = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _split_topk_enabled() -> bool:
    return os.environ.get("VLLM_FL_METAX_DECODE_SPLIT_TOPK", "0") == "1"


def _get_split_topk_buffers(
    device: torch.device, splits: int, max_b: int, hq: int, dv: int
):
    dev_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    key = (dev_index, splits, max_b, hq, dv)
    bufs = _SPLIT_TOPK_BUF_CACHE.get(key)
    if bufs is None:
        part_max = torch.empty((max_b, splits, hq), device=device, dtype=torch.float32)
        part_sum = torch.empty((max_b, splits, hq), device=device, dtype=torch.float32)
        part_acc = torch.empty(
            (max_b, splits, hq, dv), device=device, dtype=torch.float32
        )
        bufs = (part_max, part_sum, part_acc)
        _SPLIT_TOPK_BUF_CACHE[key] = bufs
    return bufs


def _try_sparse_decode_model1_split(
    q,
    kv,
    indices,
    out,
    lse,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    topk,
    page_block_size,
    softmax_scale,
):
    """Launch graph-safe split-topk prototype for MODEL1 decode when explicitly enabled.

    Returns True if the split path launched. Unsupported shapes use the original
    semantic kernel path; this is not a reference/no-op fallback.
    """
    if not _split_topk_enabled():
        return False
    if head_dim_k != 512 or head_dim_v != 512:
        return False
    if seq_q != 1 or num_heads_q != 16:
        return False
    if topk_length is None or extra_topk_length is None:
        return False
    if extra_kv is None or extra_indices is None:
        return False
    if attn_sink is None:
        return False
    splits = _env_int("VLLM_FL_METAX_DECODE_SPLIT_TOPK_S", 4)
    bk = _env_int("VLLM_FL_METAX_DECODE_SPLIT_TOPK_BK", 16)
    bh = _env_int("VLLM_FL_METAX_DECODE_SPLIT_TOPK_BH", 16)
    warps = _env_int("VLLM_FL_METAX_DECODE_SPLIT_TOPK_WARPS", 2)
    stages = _env_int("VLLM_FL_METAX_DECODE_SPLIT_TOPK_STAGES", 1)
    max_b = _env_int("VLLM_FL_METAX_DECODE_SPLIT_TOPK_MAX_B", 64)
    if splits not in (2, 4, 8, 16) or bk not in (8, 16, 32) or bh != 16:
        return False
    if batch_size > max_b:
        return False
    (part_max, part_sum, part_acc) = _get_split_topk_buffers(
        q.device, splits, max_b, num_heads_q, head_dim_v
    )
    pm = part_max[:batch_size]
    ps = part_sum[:batch_size]
    pa = part_acc[:batch_size]
    num_head_blocks = (num_heads_q + bh - 1) // bh
    if os.environ.get("VLLM_FL_METAX_DECODE_MODEL1_HALFD", "0") == "1":
        qkfold = (
            os.environ.get("VLLM_FL_METAX_DECODE_HALFD_QKFOLD", "0") == "1"
            and num_heads_q == 16
            and (bh == 16)
            and (head_dim_k == 512)
            and (head_dim_v == 512)
            and (num_head_blocks == 1)
        )
        if qkfold:
            _partial_model1_kernel_halfd_qkfold_blockdiag[
                batch_size * seq_q * num_head_blocks * splits,
            ](
                q,
                kv,
                indices,
                extra_kv,
                extra_indices,
                topk_length,
                extra_topk_length,
                softmax_scale,
                pm,
                ps,
                pa,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                kv.stride(0),
                indices.stride(0),
                indices.stride(1),
                extra_kv.stride(0),
                extra_indices.stride(0),
                extra_indices.stride(1),
                seq_q,
                num_heads_q,
                page_block_size,
                extra_kv.shape[1],
                kv.shape[0],
                extra_kv.shape[0],
                splits,
                bk,
                bh,
                num_warps=warps,
                num_stages=stages,
            )
        else:
            for _v_start, _write_stats in ((0, True), (256, False)):
                _partial_model1_kernel_halfd[
                    batch_size * seq_q * num_head_blocks * splits,
                ](
                    q,
                    kv,
                    indices,
                    extra_kv,
                    extra_indices,
                    topk_length,
                    extra_topk_length,
                    softmax_scale,
                    pm,
                    ps,
                    pa,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    kv.stride(0),
                    indices.stride(0),
                    indices.stride(1),
                    extra_kv.stride(0),
                    extra_indices.stride(0),
                    extra_indices.stride(1),
                    seq_q,
                    num_heads_q,
                    page_block_size,
                    extra_kv.shape[1],
                    kv.shape[0],
                    extra_kv.shape[0],
                    splits,
                    bk,
                    bh,
                    V_START=_v_start,
                    WRITE_STATS=_write_stats,
                    num_warps=warps,
                    num_stages=stages,
                )
    else:
        _partial_model1_kernel[batch_size * seq_q * num_head_blocks * splits,](
            q,
            kv,
            indices,
            extra_kv,
            extra_indices,
            topk_length,
            extra_topk_length,
            softmax_scale,
            pm,
            ps,
            pa,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            indices.stride(0),
            indices.stride(1),
            extra_kv.stride(0),
            extra_indices.stride(0),
            extra_indices.stride(1),
            seq_q,
            num_heads_q,
            page_block_size,
            extra_kv.shape[1],
            kv.shape[0],
            extra_kv.shape[0],
            splits,
            bk,
            bh,
            num_warps=warps,
            num_stages=stages,
        )
    _reduce_split_kernel[batch_size * num_heads_q, 2](
        pm,
        ps,
        pa,
        attn_sink,
        out,
        lse,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        batch_size,
        num_heads_q,
        splits,
        True,
        256,
        num_warps=4,
        num_stages=1,
    )
    return True


def _sparse_decode_dispatch(
    q,
    kv,
    indices,
    out,
    lse,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    topk,
    page_block_size,
    softmax_scale,
    is_fp8_kvcache,
):
    """Launch sparse decode kernel."""
    BH = 16
    num_head_blocks = (num_heads_q + BH - 1) // BH
    grid = (batch_size * seq_q * num_head_blocks,)
    skv = kv.shape[0] * page_block_size
    if head_dim_k == 512:
        if _try_sparse_decode_model1_split(
            q,
            kv,
            indices,
            out,
            lse,
            attn_sink,
            topk_length,
            extra_kv,
            extra_indices,
            extra_topk_length,
            batch_size,
            seq_q,
            num_heads_q,
            head_dim_k,
            head_dim_v,
            topk,
            page_block_size,
            softmax_scale,
        ):
            return
        _sparse_decode_model1_kernel[grid](
            q,
            kv,
            indices,
            extra_kv if extra_kv is not None else kv,
            extra_indices if extra_indices is not None else indices,
            attn_sink if attn_sink is not None else None,
            topk_length if topk_length is not None else None,
            extra_topk_length if extra_topk_length is not None else None,
            softmax_scale,
            out,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            indices.stride(0),
            indices.stride(1),
            extra_kv.stride(0) if extra_kv is not None else kv.stride(0),
            extra_indices.stride(0) if extra_indices is not None else indices.stride(0),
            extra_indices.stride(1) if extra_indices is not None else indices.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            seq_q,
            num_heads_q,
            page_block_size,
            extra_kv.shape[1] if extra_kv is not None else 1,
            kv.shape[0],
            extra_kv.shape[0] if extra_kv is not None else 0,
            topk,
            extra_indices.shape[-1] if extra_indices is not None else 0,
            attn_sink is not None,
            topk_length is not None,
            extra_kv is not None,
            extra_topk_length is not None,
        )
        return
    if is_fp8_kvcache:
        kv_bytes = kv.reshape(-1, 656).contiguous()
        kv_nope = kv_bytes[:, :512].contiguous().view(torch.float8_e4m3fn)
        stride_kvn = kv_nope.stride(0)
        kv_scales = kv_bytes[:, 512:528].contiguous().view(torch.float32)
        stride_scales_n = kv_scales.stride(0)
        kv_rope = kv_bytes[:, 528:656].contiguous().view(torch.bfloat16)
        stride_rope_n = kv_rope.stride(0)
    else:
        kv_nope = kv.reshape(-1, kv.shape[-1]).contiguous()
        stride_kvn = kv_nope.stride(0)
        kv_scales = kv_nope
        stride_scales_n = 0
        kv_rope = kv_nope
        stride_rope_n = 0
    _sparse_decode_kernel[grid](
        q,
        kv_nope,
        kv_scales,
        kv_rope,
        indices,
        attn_sink if attn_sink is not None else None,
        topk_length if topk_length is not None else None,
        softmax_scale,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        stride_kvn,
        stride_scales_n,
        stride_rope_n,
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        seq_q,
        num_heads_q,
        head_dim_k,
        skv,
        topk,
        attn_sink is not None,
        topk_length is not None,
        is_fp8_kvcache,
    )


def _dense_decode_dispatch(
    q,
    kv_cache,
    block_table,
    cache_seqlens,
    out,
    lse,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    page_block_size,
    softmax_scale,
    causal,
):
    """Launch dense decode kernel."""
    BLOCK_H = 16
    num_head_blocks = (num_heads_q + BLOCK_H - 1) // BLOCK_H
    kv_flat = kv_cache.view(-1, head_dim_k).contiguous()
    block_table = block_table.contiguous()
    grid = (num_head_blocks, batch_size * seq_q)
    _dense_decode_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_flat,
        kv_flat.stride(0),
        block_table,
        block_table.stride(0),
        cache_seqlens,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse,
        lse.stride(0),
        lse.stride(1),
        softmax_scale,
        seq_q,
        num_heads_q,
        head_dim_k,
        head_dim_v,
        page_block_size,
        causal,
    )

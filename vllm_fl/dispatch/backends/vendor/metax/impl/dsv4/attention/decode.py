# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""DSV4 TP8 decode: padding-aware blockdiag and split-topk orchestration."""

import importlib
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from . import tuning as _tuning
from .launch import local_launch, local_autotune

_native = importlib.import_module("flaggems_vllm.ops.flash_mla_with_kvcache")
FlashMLASchedMeta = _native.FlashMLASchedMeta
_can_use_tle_dense_decode = _native._can_use_tle_dense_decode
_tle_dense_decode_launch = getattr(_native, "_tle_dense_decode_launch", None)
_SPLIT_TOPK_BUF_CACHE = {}
_sparse_decode_kernel = local_autotune(
    _native._sparse_decode_kernel,
    configs=[
        triton.Config({"BK": 32, "BH": 16}, num_warps=4, num_stages=1),
        triton.Config({"BK": 16, "BH": 16}, num_warps=4, num_stages=1),
    ],
    key=["HQ", "DQK", "TOPK", "HAVE_ATTN_SINK", "HAVE_TOPK_LENGTH", "IS_FP8"],
)
_sparse_decode_model1_kernel = local_launch(
    _native._sparse_decode_model1_kernel,
    {"BK": 16, "BH": 16, "num_warps": 2, "num_stages": 1},
)
_dense_decode_kernel = local_autotune(
    _native._dense_decode_kernel,
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 16}, num_warps=4, num_stages=1),
    ],
    key=["HQ", "DQK", "HAVE_CAUSAL"],
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
        mask=offs_d[None, :] < (NOPE - BDP),
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
    start = (topk_len * i_split) // SPLITS
    end = (topk_len * (i_split + 1)) // SPLITS
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
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
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
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
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
    estart = (extra_len * i_split) // SPLITS
    eend = (extra_len * (i_split + 1)) // SPLITS
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
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
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
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
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
        pa = tl.load(part_acc + (((b * SPLITS + s) * HQc + h) * 512) + d0)
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
    real_num_heads: Optional[int] = None,
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
        helper_msg = (
            " Your input arguments are inconsistent with sched_meta. Please make "
            "sure the input arguments are consistent across different invocations "
            "of flash_mla_with_kvcache on the same sched_meta."
        )
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
            "sched_meta.config.extra_page_block_size must be equal to the "
            "page_block_size of extra_k_cache." + helper_msg
        )
        assert sched_meta.config.extra_topk == extra_topk_val, (
            "sched_meta.config.extra_topk must be equal to the last dim of "
            "extra_indices_in_kvcache." + helper_msg
        )

    batch_size, seq_q, num_heads_q, head_dim_k = q.shape
    num_heads_k = k_cache.shape[2]

    if out is None:
        out = torch.empty(
            (batch_size, seq_q, num_heads_q, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
    else:
        assert out.shape == (batch_size, seq_q, num_heads_q, head_dim_v)
        assert out.dtype == q.dtype
        assert out.device == q.device
        assert out.stride(-1) == 1
    lse = torch.empty(
        (batch_size, num_heads_q, seq_q),
        dtype=torch.float32,
        device=q.device,
    )

    if indices is not None:
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True for sparse attention"
        assert num_heads_k == 1, (
            "Currently only MQA (h_kv == 1) is supported for sparse decoding"
        )
        assert head_dim_v == 512, "Only head_size_v == 512 is supported"
        assert num_heads_q in (16, 64, 128), "Only h_q == 16, 64 or 128 is supported"
        assert head_dim_k in (
            512,
            576,
        ), "Only head_size_k == 512 or 576 is supported for sparse decoding"
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
                "extra_indices_in_kvcache must be provided when extra_k_cache "
                "is provided"
            )
            assert extra_k_cache.dtype in (
                torch.float8_e4m3fn,
                torch.int8,
                torch.uint8,
            )
        else:
            assert extra_indices_in_kvcache is None, (
                "extra_indices_in_kvcache must not be provided when extra_k_cache "
                "is not provided"
            )
            assert extra_topk_length is None, (
                "extra_topk_length must not be provided when extra_k_cache is "
                "not provided"
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
            real_num_heads=real_num_heads,
        )
    else:
        assert (
            attn_sink is None
            and extra_k_cache is None
            and extra_indices_in_kvcache is None
            and topk_length is None
            and extra_topk_length is None
        ), (
            "indices, attn_sink, extra_k_cache, extra_indices_in_kvcache, "
            "topk_length and extra_topk_length must be None when dense "
            "attention is used."
        )
        assert block_table is not None and cache_seqlens is not None, (
            "block_table and cache_seqlens must be provided when dense attention "
            "is used."
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

    return out, lse


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
    real_num_heads: Optional[int] = None,
):
    """Launch the accepted TP8 blockdiag split/reduce path.

    Returns True if the split path launched. Unsupported shapes use the original
    semantic kernel path; this is not a reference/no-op fallback.
    """
    if head_dim_k != 512 or head_dim_v != 512:
        return False
    if seq_q != 1 or num_heads_q != 16 or real_num_heads != 8:
        return False
    if topk_length is None or extra_topk_length is None:
        return False
    if extra_kv is None or extra_indices is None:
        return False
    if attn_sink is None:
        return False
    splits = _tuning.current.decode_splits
    bk = _tuning.current.decode_block_k
    bh = _tuning.current.decode_block_h
    warps = _tuning.current.decode_warps
    stages = _tuning.current.decode_stages
    max_b = _tuning.current.decode_max_batch
    if splits not in (2, 4, 8, 16) or bk not in (8, 16, 32) or bh != 16:
        return False
    if batch_size > max_b:
        return False
    part_max, part_sum, part_acc = _get_split_topk_buffers(
        q.device, splits, max_b, num_heads_q, head_dim_v
    )
    pm = part_max[:batch_size]
    ps = part_sum[:batch_size]
    pa = part_acc[:batch_size]
    num_head_blocks = 1  # TP8: eight real heads in one padded BH16 block.
    _partial_model1_kernel_halfd_qkfold_blockdiag[
        (batch_size * seq_q * num_head_blocks * splits,)
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
    _reduce_split_kernel[(batch_size * num_heads_q, 2)](
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
    real_num_heads: Optional[int] = None,
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
            real_num_heads=real_num_heads,
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
            # Q strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            # KV and indices strides
            kv.stride(0),
            indices.stride(0),
            indices.stride(1),
            extra_kv.stride(0) if extra_kv is not None else kv.stride(0),
            extra_indices.stride(0) if extra_indices is not None else indices.stride(0),
            extra_indices.stride(1) if extra_indices is not None else indices.stride(1),
            # Output strides
            out.stride(0),
            out.stride(1),
            out.stride(2),
            # LSE strides
            lse.stride(0),
            lse.stride(1),
            # Scalar args
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
        # FP8 mode: kv has shape [num_blocks, page_block_size, 1, 656]
        # Layout per token (656 bytes):
        #   [0:512]   - 512 float8_e4m3fn values (NoPE)
        #   [512:528] - 4 float32 scales (16 bytes)
        #   [528:656] - 64 bfloat16 values (RoPE, 128 bytes)
        kv_bytes = kv.reshape(-1, 656).contiguous()  # [num_tokens, 656] uint8

        # NoPE FP8 part: first 512 bytes as float8_e4m3fn
        kv_nope = (
            kv_bytes[:, :512].contiguous().view(torch.float8_e4m3fn)
        )  # [num_tokens, 512]
        stride_kvn = kv_nope.stride(0)

        # Scales: bytes [512:528] as 4 float32 values
        kv_scales = (
            kv_bytes[:, 512:528].contiguous().view(torch.float32)
        )  # [num_tokens, 4]
        stride_scales_n = kv_scales.stride(0)

        # RoPE BF16 part: bytes [528:656] as 64 bfloat16 values
        kv_rope = (
            kv_bytes[:, 528:656].contiguous().view(torch.bfloat16)
        )  # [num_tokens, 64]
        stride_rope_n = kv_rope.stride(0)
    else:
        # BF16 mode: kv has shape [num_blocks, page_block_size, 1, head_dim_k]
        kv_nope = kv.reshape(-1, kv.shape[-1]).contiguous()
        stride_kvn = kv_nope.stride(0)
        kv_scales = kv_nope  # unused, pass same tensor
        stride_scales_n = 0
        kv_rope = kv_nope  # unused, pass same tensor
        stride_rope_n = 0

    # # TLE warp specialization path TODO
    # if _can_use_tle_sparse_decode(q, indices, head_dim_v, head_dim_k, is_fp8_kvcache):
    #     _tle_sparse_decode_launch(
    #         q, kv_nope, kv_scales, kv_rope, indices, out, lse,
    #         attn_sink, topk_length,
    #         batch_size, seq_q, num_heads_q,
    #         head_dim_k, head_dim_v, topk, skv,
    #         softmax_scale, is_fp8_kvcache,
    #         stride_kvn, stride_scales_n, stride_rope_n,
    #     )
    #     return

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
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # KV strides
        stride_kvn,
        stride_scales_n,
        stride_rope_n,
        # Indices strides
        indices.stride(0),
        indices.stride(1),
        # Output strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        # LSE strides
        lse.stride(0),
        lse.stride(1),
        # Scalar args
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

    # KV cache: [num_blocks, page_block_size, num_heads_k, head_dim_k]
    # Flatten to [num_tokens_total, head_dim_k] for paged access
    kv_flat = kv_cache.view(-1, head_dim_k).contiguous()
    block_table = block_table.contiguous()

    # TLE warp specialization path
    if _can_use_tle_dense_decode(q, kv_cache, block_table, head_dim_v, page_block_size):
        _tle_dense_decode_launch(
            q,
            kv_flat,
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
        )
        return

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

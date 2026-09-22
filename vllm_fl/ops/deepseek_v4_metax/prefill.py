# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

import os
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BK": 16, "BH": 16}, num_warps=2, num_stages=2)],
    key=["SQ", "HQ", "DQK", "SKV", "TOPK", "HAVE_ATTN_SINK", "HAVE_TOPK_LENGTH"],
)
@triton.jit
def triton_flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256
    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base
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
        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
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
    tl.store(max_log_base + offs_h, max_log)
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)
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
def triton_flash_mla_sparse_fwd_halfd_serial(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    V_START: tl.constexpr,
    WRITE_STATS: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256
    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base
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
    acc = tl.zeros([BH, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_idx = BK * ck + offs_t
        t_msk = t_idx < topk_len
        kv_ids = tl.load(t_base + t_idx, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        kv_ids = tl.where(mask_ids, kv_ids, 0)
        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
            tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        val_blk = kv_blk0 if V_START == 0 else kv_blk1
        acc = tl.dot(
            exp_qk.to(tl.bfloat16),
            val_blk.trans(),
            acc * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    if WRITE_STATS:
        tl.store(max_log_base + offs_h, max_log)
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    if WRITE_STATS:
        tl.store(l_base + offs_h, lse_out)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp
    out_vals = tl.where(valid_mask[:, None], acc * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + (V_START + offs_d)[None, :]
    tl.store(o_ptr, out_vals.to(tl.bfloat16))


@triton.jit
def triton_flash_mla_sparse_fwd_halfd_qkfold_blockdiag(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256
    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base
    offs_h = tl.arange(0, BH)
    real_h = offs_h % 8
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)
    q_ptr = q_base + real_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + real_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")
    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc = tl.zeros([BH, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_idx = BK * ck + offs_t
        t_msk = t_idx < topk_len
        kv_ids = tl.load(t_base + t_idx, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        kv_ids = tl.where(mask_ids, kv_ids, 0)
        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
            tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        prob = exp_qk.to(tl.bfloat16)
        zero_prob = tl.zeros([BH, BK], dtype=tl.bfloat16)
        prob_lo = tl.where(offs_h[:, None] < 8, prob, zero_prob)
        prob_hi = tl.where(offs_h[:, None] >= 8, prob, zero_prob)
        prob_pair = tl.reshape(tl.join(prob_lo, prob_hi), [BH, BK * 2])
        kv_pair = tl.reshape(tl.join(kv_blk0, kv_blk1), [BDP, BK * 2])
        acc = tl.dot(
            prob_pair, kv_pair.trans(), acc * alpha[:, None], out_dtype=tl.float32
        )
        max_log = new_max
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    tl.store(max_log_base + offs_h, max_log)
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + real_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp
    out_vals = tl.where(valid_mask[:, None], acc * factor[:, None], 0.0)
    v_half = (offs_h >= 8).to(tl.int64) * BDP
    o_ptr = o_base + real_h[:, None] * stride_oh + v_half[:, None] + offs_d[None, :]
    tl.store(o_ptr, out_vals.to(tl.bfloat16))


def _flash_mla_sparse_halfd_serial_enabled() -> bool:
    return os.environ.get("VLLM_FL_METAX_PREFILL_HALFD_SERIAL", "0") == "1"


def _flash_mla_sparse_qkfold_blockdiag_enabled() -> bool:
    return os.environ.get("VLLM_FL_METAX_PREFILL_QKFOLD_BLOCKDIAG", "0") == "1"


_HALFD_STATS = {"hit": 0, "miss": 0, "fail_match": 0}


_HEADPACK_STATS = {"hit": 0, "miss": 0}
_STAGED_ENABLED = os.environ.get("VLLM_FL_METAX_PREFILL_ATTN_STAGED", "0") == "1"
_STAGED_VERIFY = int(os.environ.get("VLLM_FL_METAX_PREFILL_ATTN_STAGED_VERIFY", "0"))
_STAGED_STATS = {"hit": 0, "fallback": 0}


def _head_packing_enabled():
    return os.environ.get("VLLM_FL_METAX_PREFILL_HEAD_PACKING", "0") == "1"


def _flash_mla_sparse_fwd_head_packing(
    q, kv, indices, sm_scale, DV, attn_sink, topk_length, output, max_logits, lse
):
    if not _head_packing_enabled():
        return False
    (SQ, HQ, DQK) = q.shape
    (_, HKV, _) = kv.shape
    TOPK = indices.shape[-1]
    if not (
        SQ >= 8
        and HQ == 16
        and (HKV == 1)
        and (DQK in (512, 576))
        and (DV == 512)
        and (TOPK > 0)
        and q.is_contiguous()
        and indices.is_contiguous()
    ):
        return False
    try:
        import torch.distributed as dist
    except Exception:
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if torch.compiler.is_compiling():
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        return False
    world = dist.get_world_size()
    if world < 2 or world % 2 != 0:
        return False
    SKV = kv.shape[0]
    rank = dist.get_rank()
    peer = rank ^ 1
    T2 = SQ // 2
    if rank % 2 == 0:
        (lo, hi, plo, phi) = (0, T2, T2, SQ)
    else:
        (lo, hi, plo, phi) = (T2, SQ, 0, T2)
    T_loc = hi - lo
    T_peer = phi - plo
    dev = q.device
    REAL = HQ // 2
    q_send = q[plo:phi, :REAL].contiguous()
    q_recv = torch.empty((T_loc, REAL, DQK), dtype=q.dtype, device=dev)
    plan = [dist.P2POp(dist.isend, q_send, peer), dist.P2POp(dist.irecv, q_recv, peer)]
    if attn_sink is not None:
        sink_send = attn_sink[:REAL].contiguous()
        sink_recv = torch.empty(REAL, dtype=attn_sink.dtype, device=dev)
        plan.append(dist.P2POp(dist.isend, sink_send, peer))
        plan.append(dist.P2POp(dist.irecv, sink_recv, peer))
    for req in dist.batch_isend_irecv(plan):
        req.wait()
    q16 = torch.empty((T_loc, HQ, DQK), dtype=q.dtype, device=dev)
    q16[:, :REAL] = q[lo:hi, :REAL]
    q16[:, REAL:] = q_recv
    if attn_sink is not None:
        sink16 = torch.cat((attn_sink[:REAL], sink_recv)).contiguous()
    else:
        sink16 = None
    out16 = torch.empty((T_loc, HQ, DV), dtype=output.dtype, device=dev)
    ml16 = torch.empty((T_loc, HQ), dtype=torch.float32, device=dev)
    lse16 = torch.empty((T_loc, HQ), dtype=torch.float32, device=dev)
    idx_loc = indices[lo:hi].contiguous()
    topk_loc = topk_length[lo:hi].contiguous() if topk_length is not None else None
    grid = (T_loc,)

    def launch_original(target_out, target_max, target_lse):
        for _v_start, _write_stats in ((0, True), (256, False)):
            triton_flash_mla_sparse_fwd_halfd_serial[grid](
                q16,
                kv,
                idx_loc,
                sink16,
                topk_loc,
                sm_scale,
                target_out,
                target_max,
                target_lse,
                q16.stride(1),
                q16.stride(0),
                kv.stride(1),
                kv.stride(0),
                idx_loc.stride(1),
                idx_loc.stride(0),
                target_out.stride(1),
                target_out.stride(0),
                target_max.stride(0),
                target_lse.stride(0),
                T_loc,
                HQ,
                DQK,
                SKV,
                TOPK,
                sink16 is not None,
                topk_loc is not None,
                BK=16,
                BH=16,
                V_START=_v_start,
                WRITE_STATS=_write_stats,
                num_warps=2,
                num_stages=1,
            )

    if (
        _STAGED_ENABLED
        and DQK == 512
        and (q16.dtype == torch.bfloat16)
        and (kv.dtype == torch.bfloat16)
        and kv.is_contiguous()
        and (idx_loc.dtype == torch.int32)
    ):
        from .attention_staged import buffers as staged_buffers, run as staged_run

        work = staged_buffers(q16, TOPK)
        staged_run(
            q16,
            kv,
            idx_loc,
            sink16,
            topk_loc,
            out16,
            ml16,
            lse16,
            work,
            qw=2,
            pw=2,
            sm_scale=sm_scale,
        )
        _STAGED_STATS["hit"] += 1
        hit = _STAGED_STATS["hit"]
        if hit <= 3 or hit % 256 == 0:
            sum((x.numel() * x.element_size() for x in work))
            pass
    else:
        _STAGED_STATS["fallback"] += 1
        launch_original(out16, ml16, lse16)
    output[lo:hi, :REAL] = out16[:, :REAL]
    o_send = out16[:, REAL:].contiguous()
    o_recv = torch.empty((T_peer, REAL, DV), dtype=output.dtype, device=dev)
    for req in dist.batch_isend_irecv(
        [dist.P2POp(dist.isend, o_send, peer), dist.P2POp(dist.irecv, o_recv, peer)]
    ):
        req.wait()
    output[plo:phi, :REAL] = o_recv
    _HEADPACK_STATS["hit"] += 1
    return True


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    max_logits_out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and (indices.dtype == torch.int32)
    )
    (SQ, HQ, DQK) = q.shape
    (SKV, HKV, _) = kv.shape
    assert d_v == 512, "Unsupported d_v"
    DV = d_v
    assert kv.shape[-1] == DQK
    (_, _, TOPK) = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 16 or HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"
    if out is None:
        output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    else:
        assert out.device == q.device and out.dtype == q.dtype
        assert out.shape == (SQ, HQ, DV), (
            f"out error shape {tuple(out.shape)} != {(SQ, HQ, DV)}"
        )
        output = out
    if max_logits_out is None:
        max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    else:
        assert (
            max_logits_out.device == q.device and max_logits_out.dtype == torch.float32
        )
        assert max_logits_out.shape == (SQ, HQ), "max_logits_out error shape"
        max_logits = max_logits_out
    if lse_out is None:
        lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    else:
        assert lse_out.device == q.device and lse_out.dtype == torch.float32
        assert lse_out.shape == (SQ, HQ), "lse_out error shape"
        lse = lse_out

    def triton_grid(META):
        return (triton.cdiv(HQ, META["BH"]) * SQ,)

    if _flash_mla_sparse_fwd_head_packing(
        q, kv, indices, sm_scale, DV, attn_sink, topk_length, output, max_logits, lse
    ):
        return (output, max_logits, lse)
    if _flash_mla_sparse_halfd_serial_enabled():
        _ehit = _HALFD_STATS["hit"] + 1
        _HALFD_STATS["hit"] = _ehit
    else:
        _emiss = _HALFD_STATS["miss"] + 1
        _HALFD_STATS["miss"] = _emiss
    if _flash_mla_sparse_halfd_serial_enabled() and (
        SQ > 1
        and HQ == 16
        and (HKV == 1)
        and (DQK in (512, 576))
        and (DV == 512)
        and (TOPK > 0)
        and q.is_contiguous()
        and kv.is_contiguous()
        and indices.is_contiguous()
        and (attn_sink is None or attn_sink.is_contiguous())
        and (topk_length is None or topk_length.is_contiguous())
    ):
        if _flash_mla_sparse_qkfold_blockdiag_enabled():
            triton_flash_mla_sparse_fwd_halfd_qkfold_blockdiag[triton_grid](
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                sm_scale,
                output,
                max_logits,
                lse,
                q.stride(1),
                q.stride(0),
                kv.stride(1),
                kv.stride(0),
                indices.stride(1),
                indices.stride(0),
                output.stride(1),
                output.stride(0),
                max_logits.stride(0),
                lse.stride(0),
                SQ,
                HQ,
                DQK,
                SKV,
                TOPK,
                attn_sink is not None,
                topk_length is not None,
                BK=16,
                BH=16,
                num_warps=4,
                num_stages=1,
            )
        else:
            for _v_start, _write_stats in ((0, True), (256, False)):
                triton_flash_mla_sparse_fwd_halfd_serial[triton_grid](
                    q,
                    kv,
                    indices,
                    attn_sink,
                    topk_length,
                    sm_scale,
                    output,
                    max_logits,
                    lse,
                    q.stride(1),
                    q.stride(0),
                    kv.stride(1),
                    kv.stride(0),
                    indices.stride(1),
                    indices.stride(0),
                    output.stride(1),
                    output.stride(0),
                    max_logits.stride(0),
                    lse.stride(0),
                    SQ,
                    HQ,
                    DQK,
                    SKV,
                    TOPK,
                    attn_sink is not None,
                    topk_length is not None,
                    BK=16,
                    BH=16,
                    V_START=_v_start,
                    WRITE_STATS=_write_stats,
                    num_warps=2,
                    num_stages=1,
                )
        return (output, max_logits, lse)
    if _flash_mla_sparse_halfd_serial_enabled() and _HALFD_STATS["fail_match"] < 8:
        if not (
            SQ > 1 and HQ == 16 and (HKV == 1) and (DQK in (512, 576)) and (DV == 512)
        ):
            _HALFD_STATS["fail_match"] += 1
            pass
    triton_flash_mla_sparse_fwd[triton_grid](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits,
        lse,
        q.stride(1),
        q.stride(0),
        kv.stride(1),
        kv.stride(0),
        indices.stride(1),
        indices.stride(0),
        output.stride(1),
        output.stride(0),
        max_logits.stride(0),
        lse.stride(0),
        SQ,
        HQ,
        DQK,
        SKV,
        TOPK,
        attn_sink is not None,
        topk_length is not None,
    )
    return (output, max_logits, lse)

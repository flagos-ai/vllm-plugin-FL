# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from the FlagGems/vLLM DSV4 implementation.

import torch
import triton
import triton.language as tl


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.jit
def _paged_indexer_mqa_logits_kernel(
    Q,  # [B,H,D] bf16
    KV_VALUES,  # [num_blocks, block_size*D] uint8/fp8 bytes region
    KV_SCALES,  # [num_blocks, block_size*num_qblocks] fp32 scale region
    WEIGHTS,  # [B,H] fp32
    VALID_LENS,  # [B] int32
    BLOCK_TABLE,  # [B,max_blocks] int32
    OUT,  # [B,N] fp32
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kv_block: tl.constexpr,
    stride_ks_block: tl.constexpr,
    stride_wb: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_bts: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_on: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_QBLOCKS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    offs_kv = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    valid_len = tl.load(VALID_LENS + pid_b)
    valid = offs_kv < valid_len
    out_mask = offs_kv < N

    phys_block_idx = offs_kv // BLOCK_SIZE
    intra = offs_kv - phys_block_idx * BLOCK_SIZE
    block_ids = tl.load(
        BLOCK_TABLE + pid_b * stride_btb + phys_block_idx * stride_bts,
        mask=out_mask,
        other=0,
    )
    block_ids = tl.where(valid, block_ids, 0)
    intra = tl.where(valid, intra, 0)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    # Indexer KV cache layout per page: all token values first, then all token scales.
    kv_ptrs = (
        KV_VALUES
        + block_ids[:, None].to(tl.int64) * stride_kv_block
        + intra[:, None] * D
        + offs_d[None, :]
    )
    kv_u8 = tl.load(kv_ptrs, mask=valid[:, None] & d_mask[None, :], other=0)
    kv_fp8 = kv_u8.to(tl.float8e4nv, bitcast=True)
    kv_f32 = kv_fp8.to(tl.float32)
    # D=128 in DSV4 indexer, NUM_QBLOCKS=1. Keep NUM_QBLOCKS arg for documentation/guard.
    scale = tl.load(
        KV_SCALES + block_ids.to(tl.int64) * stride_ks_block + intra * NUM_QBLOCKS,
        mask=valid,
        other=0.0,
    )

    acc = tl.zeros([BLOCK_KV], dtype=tl.float32)
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        q_ptrs = (
            Q
            + pid_b * stride_qb
            + offs_h[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q_vals = tl.load(q_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0).to(
            tl.float32
        )
        # [BK,D] dot [D,BH] -> [BK,BH]
        dots = tl.dot(kv_f32, tl.trans(q_vals), out_dtype=tl.float32) * scale[:, None]
        dots = tl.maximum(dots, 0.0)
        w = tl.load(
            WEIGHTS + pid_b * stride_wb + offs_h * stride_wh, mask=h_mask, other=0.0
        ).to(tl.float32)
        acc += tl.sum(dots * w[None, :], axis=1)

    vals = tl.where(valid, acc, float("-inf"))
    tl.store(OUT + pid_b * stride_ob + offs_kv * stride_on, vals, mask=out_mask)


def paged_indexer_mqa_logits(
    q,
    kv_cache,
    weights,
    valid_lens,
    block_table,
    out=None,
    num_slots=None,
    block_kv=64,
    block_h=16,
):
    """[early-exit 20260830: skip fully-masked blocks, semantics-identical]
    Direct paged FP8 MQA logits for MetaX DSV4 indexer cache.

        Semantics match current graph-safe decode path after cp_gather_indexer_k_quant_cache:
          logits[b,j] = sum_h weights[b,h] * relu(dot(q[b,h,:], fp8(k[b,j,:])*scale[b,j]))
          logits[b,j>=valid_lens[b]] = -inf
        Supports D=128 indexer cache with one fp32 scale per token.
    """
    assert (
        q.is_cuda
        and kv_cache.is_cuda
        and weights.is_cuda
        and valid_lens.is_cuda
        and block_table.is_cuda
    )
    B, H, D = q.shape
    assert D == 128, f"prototype only supports D=128, got {D}"
    block_size = kv_cache.shape[1]
    if num_slots is None:
        num_slots = block_table.shape[1] * block_size
    if out is None:
        out = torch.empty((B, num_slots), device=q.device, dtype=torch.float32)
    kv_flat = kv_cache.view(kv_cache.shape[0], -1)
    kv_values = kv_flat[:, : block_size * D]
    kv_scales = kv_flat[:, block_size * D :].view(torch.float32)
    num_qblocks = (
        D * 4 // 4
    )  # not used to infer; D=128 => 1 fp32 per token below overwritten
    num_qblocks = 1
    grid = (B, cdiv(num_slots, block_kv))
    _paged_indexer_mqa_logits_kernel[grid](
        q.contiguous(),
        kv_values,
        kv_scales,
        weights.contiguous(),
        valid_lens.contiguous(),
        block_table.contiguous(),
        out,
        q.contiguous().stride(0),
        q.contiguous().stride(1),
        q.contiguous().stride(2),
        kv_values.stride(0),
        kv_scales.stride(0),
        weights.contiguous().stride(0),
        weights.contiguous().stride(1),
        block_table.contiguous().stride(0),
        block_table.contiguous().stride(1),
        out.stride(0),
        out.stride(1),
        num_slots,
        H,
        D,
        block_size,
        num_qblocks,
        block_kv,
        block_h,
        triton.next_power_of_2(D),
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _paged_indexer_mqa_logits_match_kernel(
    Q,
    KV_VALUES,
    KV_SCALES,
    WEIGHTS,
    VALID_LENS,
    BLOCK_TABLE,
    OUT,
    stride_q_row: tl.constexpr,
    stride_out_row: tl.constexpr,
    stride_kv_block: tl.constexpr,
    stride_ks_block: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_bts: tl.constexpr,
    NB: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_QBLOCKS: tl.constexpr,
):
    row = tl.program_id(1)
    blk = tl.program_id(0)
    valid_len = tl.load(VALID_LENS + row)
    k_offs = blk * BC + tl.arange(0, BC)
    # Early-exit (ml-static-waste fix): blocks entirely beyond valid_len produce
    # all -inf in the original semantics (k_mask all false -> out = -inf).
    # Skip the q load, KV loads and tl.dot; only store -inf. This removes the
    # per-step compute/scratch that scaled statically with max_model_len instead
    # of the actual context length. Grid stays static -> graph-replay safe.
    if blk * BC >= valid_len:
        tl.store(
            OUT + row * stride_out_row + k_offs,
            tl.full((BC,), -float("inf"), dtype=tl.float32),
        )
        return
    h_offs = tl.arange(0, H)
    d_offs = tl.arange(0, D)
    q = tl.load(Q + row * stride_q_row + h_offs[:, None] * D + d_offs[None, :])

    k_mask = k_offs < valid_len
    phys_block_idx = k_offs // BLOCK_SIZE
    intra = k_offs - phys_block_idx * BLOCK_SIZE
    block_ids = tl.load(
        BLOCK_TABLE + row * stride_btb + phys_block_idx * stride_bts,
        mask=k_mask,
        other=0,
    )
    block_ids = tl.where(k_mask, block_ids, 0)
    intra = tl.where(k_mask, intra, 0)

    k_ptrs = (
        KV_VALUES
        + block_ids[:, None].to(tl.int64) * stride_kv_block
        + intra[:, None] * D
        + d_offs[None, :]
    )
    k_u8 = tl.load(k_ptrs, mask=k_mask[:, None], other=0)
    k_fp8 = k_u8.to(tl.float8e4nv, bitcast=True)
    scale = tl.load(
        KV_SCALES + block_ids.to(tl.int64) * stride_ks_block + intra * NUM_QBLOCKS,
        mask=k_mask,
        other=0.0,
    )
    k_bf16 = k_fp8.to(tl.bfloat16) * scale[:, None].to(tl.bfloat16)

    scores = tl.dot(q, tl.trans(k_bf16))
    w = tl.load(WEIGHTS + row * H + h_offs)
    scores = tl.maximum(scores, 0.0)
    row_scores = tl.sum(scores * w[:, None], axis=0)
    out = tl.where(k_mask, row_scores, -float("inf"))
    tl.store(OUT + row * stride_out_row + k_offs, out)


def paged_indexer_mqa_logits_match(
    q, kv_cache, weights, valid_lens, block_table, out=None, num_slots=None
):
    assert q.is_cuda and kv_cache.is_cuda
    B, H, D = q.shape
    assert D == 128 and H == 64, (H, D)
    block_size = kv_cache.shape[1]
    if num_slots is None:
        num_slots = block_table.shape[1] * block_size
    if out is None:
        out = torch.empty((B, num_slots), device=q.device, dtype=torch.float32)
    kv_flat = kv_cache.view(kv_cache.shape[0], -1)
    kv_values = kv_flat[:, : block_size * D]
    kv_scales = kv_flat[:, block_size * D :].view(torch.float32)
    BC = 128
    assert num_slots % BC == 0
    NB = num_slots // BC
    qc = q.contiguous()
    wc = weights.contiguous()
    btc = block_table.contiguous()
    vlc = valid_lens.contiguous()
    _paged_indexer_mqa_logits_match_kernel[(NB, B)](
        qc,
        kv_values,
        kv_scales,
        wc,
        vlc,
        btc,
        out,
        qc.stride(0),
        out.stride(0),
        kv_values.stride(0),
        kv_scales.stride(0),
        btc.stride(0),
        btc.stride(1),
        NB=NB,
        H=H,
        D=D,
        BC=BC,
        BLOCK_SIZE=block_size,
        NUM_QBLOCKS=1,
        num_warps=4,
    )
    return out

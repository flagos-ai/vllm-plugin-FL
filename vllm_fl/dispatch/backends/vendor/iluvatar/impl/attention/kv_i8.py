# Copyright (c) 2026 BAAI. All rights reserved.
"""Self-developed Asymmetric Split-Bitplane Group-Quantized Paged KV Cache ("K8V5"):
  * Keys (K, I8G16, 144 bytes/row): 8 groups of 16 int8 values + 8 bf16 scales, isolating
    high/low-frequency NeoX RoPE phase bands 1-to-1 with 16x16x16 WMMA K=16.
  * Values (V, I5G32, 88 bytes/row): 4 groups of 32 symmetric 5-bit values [-15..+15]
    (stored as unsigned u = q + 16 in [1..31] split into [64B low-4-bit][16B high-1-bit]
    + [8B 4xbf16 scales]), exploiting non-RoPE value smoothness and 1/sqrt(L) convex
    attention averaging while keeping every lane access naturally aligned.

Block layout inside kv_cache [num_blocks, 2, block_size, Hkv, 58] bf16
(= block_size * Hkv * 232 bytes per block, 7,424 bytes at BS=16, Hkv=2):
  * [0 .. BS*Hkv*144):                  K[BS, Hkv, 144B]  (16B aligned)
  * [BS*Hkv*144 .. BS*Hkv*(144+88)):    V[BS, Hkv, 88B]   (8B aligned)
"""

import torch
import triton
import triton.language as tl

K_ROW_BYTES = 144     # 128 int8 + 8 bf16 scales (I8G16)
V_ROW_BYTES = 88      # 64B lo4 + 16B hi1 + 8B 4xbf16 scales (I5G32)
PAIR_BYTES = K_ROW_BYTES + V_ROW_BYTES   # 232 bytes per (token, kv_head)
ROW_SLOTS = PAIR_BYTES // 4              # 58 bf16 slots per {0, 1} half of kv_cache
ROW_BYTES = K_ROW_BYTES                  # kept for backward-compat constants
_MIN_SCALE = 2.0 ** -24


@triton.jit
def _quantize_k_i8g16(x):
    """x: [128] fp32 -> (words [32] int32 = 4xint8 each, scales [8] bf16)."""
    x2 = tl.reshape(x, [8, 16])
    amax = tl.max(tl.abs(x2), axis=1)
    scale16 = tl.maximum(amax / 127.0, 5.9604645e-08).to(tl.bfloat16)
    inv = 1.0 / scale16.to(tl.float32)
    q = x2 * inv[:, None]
    q = q + tl.where(q >= 0, 0.5, -0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    q4 = tl.reshape(q, [32, 4])
    shift = (tl.arange(0, 4) * 8)[None, :]
    words = tl.sum((q4 & 0xFF) << shift, axis=1)
    return words, scale16


@triton.jit
def _quantize_v_i5g32(x):
    """x: [128] fp32 -> (w_lo [16] int32, w_hi [4] int32, scales [4] bf16)."""
    x4 = tl.reshape(x, [4, 32])
    amax = tl.max(tl.abs(x4), axis=1)
    scale32 = tl.maximum(amax / 15.0, 5.9604645e-08).to(tl.bfloat16)
    inv = 1.0 / scale32.to(tl.float32)
    qf = x4 * inv[:, None]
    qf = qf + tl.where(qf >= 0, 0.5, -0.5)
    u = tl.minimum(tl.maximum(qf, -15.0), 15.0).to(tl.int32) + 16  # [4, 32] in [1..31]
    lo = tl.reshape(u & 0xF, [16, 8])
    shift4 = (tl.arange(0, 8) * 4)[None, :]
    w_lo = tl.sum(lo << shift4, axis=1)        # [16] int32 = 64 bytes
    hi = (u >> 4) & 1                          # [4, 32]
    shift1 = tl.arange(0, 32)[None, :]
    w_hi = tl.sum(hi << shift1, axis=1)        # [4] int32 = 16 bytes
    return w_lo, w_hi, scale32


@triton.jit
def _reshape_and_cache_k8v5_kernel(
    key_ptr, value_ptr, kv_u8_ptr, slot_mapping_ptr,
    stride_kt, stride_kh, stride_vt, stride_vh,
    cache_block_stride,      # in BYTES (kv_u8.stride(0))
    v_block_off: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tok = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    pos = slot % BLOCK_SIZE
    offs_d = tl.arange(0, 128)
    offs_w32 = tl.arange(0, 32)
    offs_g8 = tl.arange(0, 8)
    offs_w16 = tl.arange(0, 16)
    offs_g4 = tl.arange(0, 4)
    blk_base = kv_u8_ptr + blk * cache_block_stride
    for h in tl.static_range(NUM_KV_HEADS):
        idx = pos * NUM_KV_HEADS + h
        k_row = blk_base + idx * 144
        k = tl.load(key_ptr + tok.to(tl.int64) * stride_kt + h * stride_kh + offs_d).to(tl.float32)
        kw, ks = _quantize_k_i8g16(k)
        tl.store(k_row.to(tl.pointer_type(tl.int32)) + offs_w32, kw)
        tl.store((k_row + 128).to(tl.pointer_type(tl.bfloat16)) + offs_g8, ks)

        v_row = blk_base + v_block_off + idx * 88
        v = tl.load(value_ptr + tok.to(tl.int64) * stride_vt + h * stride_vh + offs_d).to(tl.float32)
        v_lo, v_hi, vs = _quantize_v_i5g32(v)
        tl.store(v_row.to(tl.pointer_type(tl.int32)) + offs_w16, v_lo)
        tl.store((v_row + 64).to(tl.pointer_type(tl.int32)) + offs_g4, v_hi)
        tl.store((v_row + 80).to(tl.pointer_type(tl.bfloat16)) + offs_g4, vs)


def reshape_and_cache_i8(key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """key/value: [T, Hkv, 128] bf16; kv_cache: [num_blocks, 2, block_size, Hkv, 58] bf16."""
    T, Hkv, D = key.shape
    assert D == 128 and kv_cache.shape[-1] == ROW_SLOTS
    if T == 0:
        return
    block_size = kv_cache.shape[2]
    kv_u8 = kv_cache.view(torch.uint8)
    _reshape_and_cache_k8v5_kernel[(T,)](
        key, value, kv_u8, slot_mapping,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        kv_u8.stride(0),
        v_block_off=block_size * Hkv * K_ROW_BYTES,
        NUM_KV_HEADS=Hkv, BLOCK_SIZE=block_size, num_warps=1,
    )


@triton.jit
def _dequant_kv_pages_k8v5_kernel(
    kv_u8_ptr, k_dst_ptr, v_dst_ptr,
    bt_ptr, sp_ptr,
    stride_bt,
    src_block_stride,        # BYTES
    dst_block_stride, dst_token_stride, dst_head_stride,   # ELEMENTS (bf16)
    v_block_off: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_head = tl.program_id(1)
    is_v = tl.program_id(2)
    seq_idx = tl.load(sp_ptr + 2 * pid).to(tl.int64)
    page_idx = tl.load(sp_ptr + 2 * pid + 1).to(tl.int64)
    src_blk = tl.load(bt_ptr + seq_idx * stride_bt + page_idx).to(tl.int64)
    dst_blk = pid.to(tl.int64)
    offs_t = tl.arange(0, BLOCK_SIZE)
    blk_base = kv_u8_ptr + src_blk * src_block_stride
    idx = offs_t.to(tl.int64) * NUM_KV_HEADS + kv_head.to(tl.int64)
    if is_v == 0:
        offs_w = tl.arange(0, 32)
        k_row = blk_base + idx * 144
        words = tl.load((k_row[:, None]).to(tl.pointer_type(tl.int32)) + offs_w[None, :])
        sc = tl.load((k_row[:, None] + 128).to(tl.pointer_type(tl.bfloat16)) + (offs_w // 4)[None, :]).to(tl.float32)
        drow = k_dst_ptr + dst_blk * dst_block_stride + offs_t * dst_token_stride + kv_head * dst_head_stride
        for i in tl.static_range(4):
            vals = ((words << (24 - 8 * i)) >> 24).to(tl.float32) * sc
            tl.store(drow[:, None] + 4 * offs_w[None, :] + i, vals.to(k_dst_ptr.dtype.element_ty))
    else:
        offs_c = tl.arange(0, 16)
        v_row = blk_base + v_block_off + idx * 88
        v_lo = tl.load((v_row[:, None]).to(tl.pointer_type(tl.int32)) + offs_c[None, :])
        v_hi = tl.load(v_row[:, None] + 64 + offs_c[None, :]).to(tl.int32)
        vs = tl.load((v_row[:, None] + 80).to(tl.pointer_type(tl.bfloat16)) + (offs_c // 4)[None, :]).to(tl.float32)
        drow = v_dst_ptr + dst_blk * dst_block_stride + offs_t * dst_token_stride + kv_head * dst_head_stride
        for k in tl.static_range(8):
            u5 = ((v_lo >> (4 * k)) & 0xF) | (((v_hi >> k) & 1) << 4)
            vals = (u5 - 16).to(tl.float32) * vs
            tl.store(drow[:, None] + 8 * offs_c[None, :] + k, vals.to(v_dst_ptr.dtype.element_ty))


def dequant_kv_pages_i8(kv_cache: torch.Tensor, k_dst_bf16, v_dst_bf16, prefill_bt_gpu, sp_gpu):
    """Fused single-launch K8V5 page dequantization indexing block_table directly on GPU."""
    P = sp_gpu.shape[0]
    if P == 0:
        return
    block_size = kv_cache.shape[2]
    Hkv = kv_cache.shape[3]
    kv_u8 = kv_cache.view(torch.uint8)
    _dequant_kv_pages_k8v5_kernel[(P, Hkv, 2)](
        kv_u8, k_dst_bf16, v_dst_bf16,
        prefill_bt_gpu, sp_gpu,
        prefill_bt_gpu.stride(0),
        kv_u8.stride(0),
        k_dst_bf16.stride(0), k_dst_bf16.stride(1), k_dst_bf16.stride(2),
        v_block_off=block_size * Hkv * K_ROW_BYTES,
        NUM_KV_HEADS=Hkv, BLOCK_SIZE=block_size, num_warps=2,
    )


@triton.jit
def _dequant_pack_varlen_k8v5_kernel(
    kv_u8_ptr, k_dst_ptr, v_dst_ptr,
    bt_ptr, kv_starts_ptr,
    stride_bt,
    src_block_stride,                         # BYTES
    dst_token_stride, dst_head_stride,        # ELEMENTS (bf16)
    v_block_off: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    seq_idx = tl.program_id(1)
    kv_head = tl.program_id(2)
    kv_start = tl.load(kv_starts_ptr + seq_idx).to(tl.int64)
    kv_len = tl.load(kv_starts_ptr + seq_idx + 1).to(tl.int64) - kv_start
    pos0 = tile_id.to(tl.int64) * BLOCK_SIZE
    if pos0 >= kv_len:
        return
    src_blk = tl.load(bt_ptr + seq_idx.to(tl.int64) * stride_bt + tile_id).to(tl.int64)
    offs_t = tl.arange(0, BLOCK_SIZE)
    live = (pos0 + offs_t) < kv_len
    blk_base = kv_u8_ptr + src_blk * src_block_stride
    idx = offs_t.to(tl.int64) * NUM_KV_HEADS + kv_head.to(tl.int64)
    dst_off = (kv_start + pos0 + offs_t) * dst_token_stride + kv_head.to(tl.int64) * dst_head_stride

    # Dequantize K (I8G16: 144B)
    offs_w = tl.arange(0, 32)
    k_row = blk_base + idx * 144
    kw = tl.load(
        (k_row[:, None]).to(tl.pointer_type(tl.int32)) + offs_w[None, :],
        mask=live[:, None],
        other=0,
    )
    ks = tl.load(
        (k_row[:, None] + 128).to(tl.pointer_type(tl.bfloat16)) + (offs_w // 4)[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    for i in tl.static_range(4):
        k_vals = ((kw << (24 - 8 * i)) >> 24).to(tl.float32) * ks
        tl.store(
            k_dst_ptr + dst_off[:, None] + 4 * offs_w[None, :] + i,
            k_vals.to(k_dst_ptr.dtype.element_ty),
            mask=live[:, None],
        )

    # Dequantize V (I5G32: 88B split bit-plane)
    offs_c = tl.arange(0, 16)
    v_row = blk_base + v_block_off + idx * 88
    v_lo = tl.load(
        (v_row[:, None]).to(tl.pointer_type(tl.int32)) + offs_c[None, :],
        mask=live[:, None],
        other=0,
    )
    v_hi = tl.load(
        v_row[:, None] + 64 + offs_c[None, :],
        mask=live[:, None],
        other=0,
    ).to(tl.int32)
    vs = tl.load(
        (v_row[:, None] + 80).to(tl.pointer_type(tl.bfloat16)) + (offs_c // 4)[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    for k in tl.static_range(8):
        u5 = ((v_lo >> (4 * k)) & 0xF) | (((v_hi >> k) & 1) << 4)
        v_vals = (u5 - 16).to(tl.float32) * vs
        tl.store(
            v_dst_ptr + dst_off[:, None] + 8 * offs_c[None, :] + k,
            v_vals.to(v_dst_ptr.dtype.element_ty),
            mask=live[:, None],
        )


def dequant_pack_varlen_i8(
    kv_cache: torch.Tensor,
    k_dst_bf16: torch.Tensor,
    v_dst_bf16: torch.Tensor,
    prefill_bt_gpu: torch.Tensor,
    kv_starts_gpu: torch.Tensor,
    max_kv_len: int,
):
    """Fused K8V5 paged KV dequantization + contiguous varlen packing into [total_kv, Hkv, 128]."""
    block_size = kv_cache.shape[2]
    Hkv = kv_cache.shape[3]
    num_tiles = (max_kv_len + block_size - 1) // block_size
    num_seqs = prefill_bt_gpu.shape[0]
    if num_tiles <= 0 or num_seqs <= 0:
        return
    kv_u8 = kv_cache.view(torch.uint8)
    _dequant_pack_varlen_k8v5_kernel[(num_tiles, num_seqs, Hkv)](
        kv_u8,
        k_dst_bf16,
        v_dst_bf16,
        prefill_bt_gpu,
        kv_starts_gpu,
        prefill_bt_gpu.stride(0),
        kv_u8.stride(0),
        k_dst_bf16.stride(0),
        k_dst_bf16.stride(1),
        v_block_off=block_size * Hkv * K_ROW_BYTES,
        NUM_KV_HEADS=Hkv,
        BLOCK_SIZE=block_size,
        num_warps=2,
    )

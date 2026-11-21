# Copyright (c) 2026 BAAI. All rights reserved.
"""GQA-packed paged prefill attention kernel (Triton).

Why a dedicated prefill kernel
------------------------------
vLLM's Triton "unified attention" kernel serves prefill and decode with one
code path.  For grouped-query models it packs ``num_queries_per_kv`` query
heads of one KV head into the rows of a tile, but it fixes the tile height at
16 rows.  For a GQA ratio of 8 that is only **2 query tokens per program**:
every K/V tile is re-read from HBM/L2 for every two tokens of a prefill
chunk, so the kernel is memory-bound even though prefill is compute-bound by
nature (arithmetic intensity ~ 2 tokens x 8 heads = 16 rows per K/V byte).

This kernel keeps the GQA packing idea but makes the number of query tokens
per program a tuning parameter (``BLOCK_Q``, default 16 -> 128 rows per
program), so each K/V tile loaded from the paged cache feeds 8x more MMA work.
It also splits the KV loop into an unmasked interior part and a masked
diagonal part so the causal mask is only evaluated where it can be false.

Supported configuration (everything else must use the generic kernel):
  * causal decoder attention, no sliding window / ALiBi / softcap / sinks
  * bf16 / fp16 Q, K, V; fp32 accumulation; unquantized KV cache
  * KV cache layout [num_blocks, block_size, num_kv_heads, head_size]
    (the layout used by vLLM's Triton backend after ``kv_cache.unbind(1)``)
  * variable-length batches (cu_seqlens_q), chunked prefill (context prefix
    already in the cache), head_size a power of two <= 256
Decode (query_len == 1 for all sequences) is handled better by the split-K
decode kernel and is not routed here.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _find_seq_idx(cu_seqlens_q_ptr, q_block_global_idx, num_seqs, origin, BLOCK_Q: tl.constexpr):
    # Binary search for the sequence whose q-block range contains the program id.
    # cu_seqlens_q may start at a nonzero row offset (``origin``, e.g. when the
    # decode rows of a mixed batch precede the prefill rows); block ranges are
    # [(cu[s]-origin) // BLOCK_Q + s, (cu[s+1]-origin) // BLOCK_Q + s + 1).
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = (tl.load(cu_seqlens_q_ptr + mid) - origin) // BLOCK_Q + mid
        if val <= q_block_global_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _gqa_prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_q_ptr, seq_lens_ptr, block_table_ptr,
    scale_log2e,
    num_seqs,
    stride_qt, stride_qh,
    stride_ot, stride_oh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    NUM_KV_HEADS: tl.constexpr,
    G: tl.constexpr,            # query heads per kv head
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # kv cache block (page) size
    BLOCK_Q: tl.constexpr,      # query tokens per program
    BLOCK_M: tl.constexpr,      # = BLOCK_Q * G rows
    TILE_N: tl.constexpr,       # keys per inner iteration
):
    q_block_global = tl.program_id(0)
    kv_head = tl.program_id(1)

    q_origin = tl.load(cu_seqlens_q_ptr)
    seq_idx = _find_seq_idx(cu_seqlens_q_ptr, q_block_global, num_seqs, q_origin, BLOCK_Q)
    q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    q_len = q_end - q_start
    q_block_local = q_block_global - ((q_start - q_origin) // BLOCK_Q + seq_idx)
    if q_block_local * BLOCK_Q >= q_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - q_len

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, TILE_N)
    qpos = q_block_local * BLOCK_Q + offs_m // G          # query position within the chunk
    head = kv_head * G + offs_m % G
    q_valid = qpos < q_len
    qpos_c = tl.minimum(qpos, q_len - 1)  # padded rows re-read a valid row (never out of bounds)
    q_rows = (q_start + qpos_c).to(tl.int64) * stride_qt + head.to(tl.int64) * stride_qh
    q = tl.load(q_ptr + q_rows[:, None] + offs_d[None, :], mask=q_valid[:, None], other=0.0)

    # absolute key position attended by each row (inclusive bound)
    q_abs = context_len + qpos
    # keys [0, kv_hi) where kv_hi = context_len + min(q_len, (q_block_local+1)*BLOCK_Q)
    kv_hi = context_len + tl.minimum(q_len, (q_block_local + 1) * BLOCK_Q)
    # keys below kv_lo_mask never need a causal mask (all rows see them)
    kv_lo_mask = context_len + q_block_local * BLOCK_Q
    n_full = kv_lo_mask // TILE_N              # number of tiles that need no mask at all
    n_tiles = (kv_hi + TILE_N - 1) // TILE_N

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    bt_base = block_table_ptr + seq_idx.to(tl.int64) * stride_bt
    k_head_off = kv_head * stride_kh
    v_head_off = kv_head * stride_vh

    # ---- interior tiles: no masking needed ------------------------------
    for j in range(0, n_full):
        key = j * TILE_N + offs_n
        blk = tl.load(bt_base + key // BLOCK_SIZE).to(tl.int64)
        k_off = blk * stride_kb + (key % BLOCK_SIZE) * stride_ks + k_head_off
        v_off = blk * stride_vb + (key % BLOCK_SIZE) * stride_vs + v_head_off
        k = tl.load(k_ptr + k_off[None, :] + offs_d[:, None])          # [D, TILE_N]
        v = tl.load(v_ptr + v_off[:, None] + offs_d[None, :])          # [TILE_N, D]
        s = tl.dot(q, k) * scale_log2e                                  # log2 domain
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # ---- diagonal tiles: causal + sequence-length masking ---------------
    for j in range(n_full, n_tiles):
        key = j * TILE_N + offs_n
        kmask = key < kv_hi
        blk = tl.load(bt_base + key // BLOCK_SIZE, mask=kmask, other=0).to(tl.int64)
        k_off = blk * stride_kb + (key % BLOCK_SIZE) * stride_ks + k_head_off
        v_off = blk * stride_vb + (key % BLOCK_SIZE) * stride_vs + v_head_off
        k = tl.load(k_ptr + k_off[None, :] + offs_d[:, None], mask=kmask[None, :], other=0.0)
        v = tl.load(v_ptr + v_off[:, None] + offs_d[None, :], mask=kmask[:, None], other=0.0)
        s = tl.dot(q, k) * scale_log2e
        causal = key[None, :] <= q_abs[:, None]
        s = tl.where(causal & kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # rows with no valid key yet (m_new == -inf) must not produce NaN
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / tl.maximum(l_i[:, None], 1e-30)
    o_rows = (q_start + qpos_c).to(tl.int64) * stride_ot + head.to(tl.int64) * stride_oh
    tl.store(o_ptr + o_rows[:, None] + offs_d[None, :], out.to(o_ptr.dtype.element_ty),
             mask=q_valid[:, None])


def gqa_prefill_attention(q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table,
                          softmax_scale, block_q=32, tile_n=128, num_warps=16, num_stages=2,
                          num_prefill_tokens=None):
    """q/out: [T, H, D]; k_cache/v_cache: [num_blocks, block_size, Hkv, D].

    ``cu_seqlens_q`` holds absolute row offsets into ``q``/``out`` (so a suffix
    of a batch can be attended without copying); ``num_prefill_tokens`` is the
    number of query rows covered by those sequences (default: all rows of q).
    """
    T, H, D = q.shape
    num_blocks, block_size, Hkv, _ = k_cache.shape
    G = H // Hkv
    num_seqs = seq_lens.shape[0]
    BLOCK_M = block_q * G
    if num_prefill_tokens is None:
        num_prefill_tokens = T
    grid = (num_prefill_tokens // block_q + num_seqs, Hkv)
    _gqa_prefill_kernel[grid](
        q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table,
        softmax_scale * 1.4426950408889634, num_seqs,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0),
        NUM_KV_HEADS=Hkv, G=G, HEAD_DIM=D, BLOCK_SIZE=block_size,
        BLOCK_Q=block_q, BLOCK_M=BLOCK_M, TILE_N=tile_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out

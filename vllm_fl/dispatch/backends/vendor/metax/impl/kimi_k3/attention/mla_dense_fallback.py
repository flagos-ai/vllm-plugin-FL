# SPDX-License-Identifier: Apache-2.0
# Derived from FlagGems-vllm flash_mla_with_kvcache (validated 3e9e47c base).
"""Accepted interleaved KV layout fallback with explicit C550 launch geometry."""

import triton
import triton.language as tl
from flaggems_vllm.ops.flash_mla_with_kvcache import (
    _can_use_tle_dense_decode,
    _tle_dense_decode_launch,
)


@triton.jit
def _dense_decode_kernel(
    Q_ptr,
    stride_q_b,
    stride_q_sq,
    stride_q_h,
    KV_cache,
    stride_kv_bs,
    stride_kv_block,
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

    # Load Q: NoPE part [BLOCK_H, HEAD_DIM_V] and RoPE part [BLOCK_H, DQK-HEAD_DIM_V]
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

    # Online softmax accumulators
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
        kv_offset = (
            kv_page_number.to(tl.int64) * stride_kv_block
            + (offs_n % PAGE_SIZE).to(tl.int64) * stride_kv_bs
        )

        # Load V (NoPE part)
        offs_v_c = kv_offset[:, None] + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c)
        k_c = tl.trans(v_c)

        # QK = q_nope @ k_nope^T
        qk = tl.dot(q_nope, k_c)

        # Add RoPE contribution
        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        # Online softmax update
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v_c.dtype), v_c)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        offs_n += BLOCK_N

    if remainder:
        mask_kvsplit = offs_n < cur_batch_seq_len
        kv_page_number = tl.load(
            Block_table + offs_n // PAGE_SIZE, mask=mask_kvsplit, other=0
        )
        kv_offset = (
            kv_page_number.to(tl.int64) * stride_kv_block
            + (offs_n % PAGE_SIZE).to(tl.int64) * stride_kv_bs
        )

        offs_v_c = kv_offset[:, None] + offs_d_nope[None, :]
        v_c = tl.load(KV_cache + offs_v_c, mask=mask_kvsplit[:, None], other=0.0)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)

        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v_c.dtype), v_c)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Store output
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

    # Store LSE
    lse_val = e_max + tl.math.log(e_sum)
    lse_offset = i_b * stride_lse_b + cur_head * stride_lse_h + i_sq
    tl.store(LSE + lse_offset, lse_val, mask=mask_head)


def dense_decode_fallback(
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
    BLOCK_H = 16  # Must match the explicit C550 dense kernel tile.
    num_head_blocks = (num_heads_q + BLOCK_H - 1) // BLOCK_H

    # KV cache: [num_blocks, page_block_size, num_heads_k, head_dim_k]
    # The portable kernel addresses pages directly, including interleaved layers.
    if kv_cache.stride(-1) != 1 or block_table.stride(-1) != 1:
        raise NotImplementedError("Dense MLA requires contiguous entry/table columns")

    # TLE retains its original flat-token layout; this view performs no copy.
    if kv_cache.is_contiguous() and _can_use_tle_dense_decode(
        q, kv_cache, block_table, head_dim_v, page_block_size
    ):
        kv_flat = kv_cache.view(-1, head_dim_k)
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
        kv_cache,
        kv_cache.stride(1),
        kv_cache.stride(0),
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
        BLOCK_H=16,
        BLOCK_N=16,
        num_warps=4,
        num_stages=1,
    )

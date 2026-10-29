"""Shared TP8 staged MLA QK and split merge; arithmetic is unchanged."""

import triton
import triton.language as tl


@triton.jit
def _qk_long_kernel(
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
    PartialMax,
    PartialSum,
    ProbabilityLow,
    Alpha,
    MAX_CHUNKS: tl.constexpr,
    SPLITS: tl.constexpr,
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
    K_TILE: tl.constexpr,
    CAP: tl.constexpr,
    COMPENSATE: tl.constexpr,
    MIN_LENGTH: tl.constexpr = 256,
    SHORT_SPLITS: tl.constexpr = 0,
    SHORT_LIMIT: tl.constexpr = 4096,
):
    """
    Dense decode kernel with paged attention and online softmax.
    Grid: (ceil(HQ / BLOCK_H), batch_size * seq_q)
    """
    split = tl.program_id(2)
    pid_h_block = tl.program_id(0)
    pid_b_sq = tl.program_id(1)
    i_b = pid_b_sq // SQ
    i_sq = pid_b_sq % SQ
    guard_length = tl.load(Seq_lens + i_b)
    if (guard_length <= MIN_LENGTH) | (guard_length > CAP):
        return

    cur_head = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_head = cur_head < HQ

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

    cur_batch_seq_len = guard_length
    Block_table += i_b * stride_bt_b

    if SHORT_SPLITS > 0:
        active_splits = tl.where(cur_batch_seq_len <= SHORT_LIMIT, SHORT_SPLITS, SPLITS)
        if split >= active_splits:
            return
        blocks_per_split = tl.where(
            cur_batch_seq_len <= SHORT_LIMIT,
            tl.cdiv(tl.cdiv(cur_batch_seq_len, BLOCK_N), SHORT_SPLITS),
            tl.cdiv(tl.cdiv(cur_batch_seq_len, BLOCK_N), SPLITS),
        )
    else:
        active_splits = (
            tl.where(cur_batch_seq_len <= 5120, 128, SPLITS)
            if SPLITS == 256
            else SPLITS
        )
        blocks_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, BLOCK_N), active_splits)
    first = split * blocks_per_split * BLOCK_N
    end = tl.minimum((split + 1) * blocks_per_split * BLOCK_N, cur_batch_seq_len)
    count = tl.maximum(end - first, 0)
    offs_n = first + tl.arange(0, BLOCK_N)
    loop_time = count // BLOCK_N
    remainder = count % BLOCK_N

    for i in range(0, loop_time):
        kv_page_number = tl.load(Block_table + offs_n // PAGE_SIZE)
        kv_offset = (
            kv_page_number.to(tl.int64) * stride_kv_block
            + (offs_n % PAGE_SIZE).to(tl.int64) * stride_kv_bs
        )

        qk = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
        for kd in range(0, HEAD_DIM_V, K_TILE):
            col = kd + tl.arange(0, K_TILE)
            qa = tl.load(
                Q_ptr
                + i_b * stride_q_b
                + i_sq * stride_q_sq
                + cur_head[:, None] * stride_q_h
                + col[None, :],
                mask_head[:, None] & (col[None, :] < HEAD_DIM_V),
                0,
            )
            kb = tl.load(
                KV_cache + kv_offset[None, :] + col[:, None],
                (col[:, None] < HEAD_DIM_V) & tl.full((1, BLOCK_N), True, tl.int1),
                0,
            )
            qk = tl.dot(qa, kb, qk)

        # Add RoPE contribution
        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        # Online softmax update
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        probability_hi = p.to(tl.bfloat16)
        probability_lo = (
            (p - probability_hi.to(tl.float32)).to(tl.bfloat16)
            if COMPENSATE
            else tl.full((BLOCK_H, BLOCK_N), 0, tl.bfloat16)
        )
        chunk_slot = (((i_b * SPLITS + split) * HQ + cur_head) * MAX_CHUNKS + i).to(
            tl.int64
        )
        column = tl.arange(0, BLOCK_N)
        tl.store(
            Out + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_hi,
            mask_head[:, None],
        )
        tl.store(
            ProbabilityLow + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_lo,
            mask_head[:, None],
        )
        tl.store(Alpha + chunk_slot, re_scale, mask_head)
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

        qk = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
        for kd in range(0, HEAD_DIM_V, K_TILE):
            col = kd + tl.arange(0, K_TILE)
            qa = tl.load(
                Q_ptr
                + i_b * stride_q_b
                + i_sq * stride_q_sq
                + cur_head[:, None] * stride_q_h
                + col[None, :],
                mask_head[:, None] & (col[None, :] < HEAD_DIM_V),
                0,
            )
            kb = tl.load(
                KV_cache + kv_offset[None, :] + col[:, None],
                (col[:, None] < HEAD_DIM_V) & mask_kvsplit[None, :],
                0,
            )
            qk = tl.dot(qa, kb, qk)

        offs_k_pe = kv_offset[None, :] + offs_d_pe[:, None]
        k_pe = tl.load(KV_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)
        qk += tl.dot(q_pe, k_pe)
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        probability_hi = p.to(tl.bfloat16)
        probability_lo = (
            (p - probability_hi.to(tl.float32)).to(tl.bfloat16)
            if COMPENSATE
            else tl.full((BLOCK_H, BLOCK_N), 0, tl.bfloat16)
        )
        chunk_slot = (
            ((i_b * SPLITS + split) * HQ + cur_head) * MAX_CHUNKS + loop_time
        ).to(tl.int64)
        column = tl.arange(0, BLOCK_N)
        tl.store(
            Out + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_hi,
            mask_head[:, None],
        )
        tl.store(
            ProbabilityLow + chunk_slot[:, None] * BLOCK_N + column[None, :],
            probability_lo,
            mask_head[:, None],
        )
        tl.store(Alpha + chunk_slot, re_scale, mask_head)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Keep unnormalized accumulators/statistics in FP32 for the merge.
    slot = ((i_b * SQ + i_sq) * SPLITS + split) * HQ + cur_head
    tl.store(PartialMax + slot, e_max, mask_head)
    tl.store(PartialSum + slot, e_sum, mask_head)


@triton.jit
def _merge_long_kernel(
    P,
    M,
    S,
    O,
    L,
    Lengths,
    OriginalLengths,
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    SPLITS: tl.constexpr,
    RS: tl.constexpr,
    VT: tl.constexpr,
    MIN_LENGTH: tl.constexpr = 256,
    SHORT_SPLITS: tl.constexpr = 0,
    SHORT_LIMIT: tl.constexpr = 4096,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    length = tl.load(Lengths + b)
    if length <= MIN_LENGTH:
        if tl.load(OriginalLengths + b) == 0:
            zero_d = tl.program_id(2) * VT + tl.arange(0, VT)
            tl.store(O + (b * H + h) * V + zero_d, 0, zero_d < V)
            if tl.program_id(2) == 0:
                tl.store(L + b * H + h, -float("inf"))
        return
    active_splits = (
        tl.where(length <= SHORT_LIMIT, SHORT_SPLITS, SPLITS)
        if SHORT_SPLITS > 0
        else SPLITS
    )
    split = tl.arange(0, RS)
    d = tl.program_id(2) * VT + tl.arange(0, VT)
    slot = (b * SPLITS + split) * H + h
    maximum = tl.load(M + slot, split < active_splits, -float("inf"))
    denom = tl.load(S + slot, split < active_splits, 0)
    overall = tl.max(maximum, 0)
    scale = tl.where(denom > 0, tl.exp(maximum - overall), 0)
    part = tl.load(
        P + slot[:, None] * V + d[None, :],
        (split[:, None] < active_splits) & (d[None, :] < V),
        0,
    )
    numerator = tl.sum(part * scale[:, None], 0)
    total = tl.sum(denom * scale, 0)
    result = tl.where(total > 0, numerator / total, 0)
    tl.store(O + (b * H + h) * V + d, result.to(tl.bfloat16), d < V)
    if tl.program_id(2) == 0:
        tl.store(
            L + b * H + h, tl.where(total > 0, overall + tl.log(total), -float("inf"))
        )

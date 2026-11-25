# SPDX-License-Identifier: Apache-2.0
"""GPU packed-request N-gram hashing with the checkpoint's signed-int64 contract."""

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["NUM_REQS", "NUM_TOKENS"])
def _hash_ids(
    IDS,
    STARTS,
    CONTEXT,
    MULT,
    SIZES,
    OFFSETS,
    OUT,
    NUM_REQS,
    NUM_TOKENS,
    ORDER: tl.constexpr,
    HEADS: tl.constexpr,
    EOS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    pos = tl.program_id(0) * BT + tl.arange(0, BT)
    valid = pos < NUM_TOKENS
    # upper_bound(starts, pos) - 1 skips repeated offsets of empty requests.
    lo = tl.full((BT,), 0, tl.int32)
    hi = tl.zeros((BT,), tl.int32) + NUM_REQS
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        start = tl.load(STARTS + mid, mid < NUM_REQS, 2147483647)
        right = (mid < NUM_REQS) & (start <= pos)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    req = tl.maximum(lo - 1, 0)
    begin = tl.load(STARTS + req)
    local = pos - begin
    token = tl.load(IDS + pos, valid, EOS).to(tl.int64)
    mixed = token * tl.load(MULT).to(tl.int64)
    alive = tl.full((BT,), True, tl.int1)
    head = tl.arange(0, BH)
    for shift in tl.static_range(1, ORDER):
        in_query = local >= shift
        previous_query = tl.load(IDS + pos - shift, valid & in_query, EOS).to(tl.int64)
        col = ORDER - 1 + local - shift
        previous_context = tl.load(
            CONTEXT + req * (ORDER - 1) + col,
            valid & ~in_query,
            EOS,
        ).to(tl.int64)
        previous = tl.where(in_query, previous_query, previous_context)
        alive &= previous != EOS
        previous = tl.where(alive, previous, EOS)
        mixed ^= previous * tl.load(MULT + shift).to(tl.int64)
        global_head = (shift - 1) * HEADS + head
        size = tl.load(SIZES + global_head, head < HEADS, 1).to(tl.int64)
        offset = tl.load(OFFSETS + global_head, head < HEADS, 0).to(tl.int64)
        remainder = mixed[:, None] % size[None, :]
        remainder = tl.where(remainder < 0, remainder + size[None, :], remainder)
        tl.store(
            OUT + pos[:, None] * ((ORDER - 1) * HEADS) + global_head[None, :],
            remainder + offset[None, :],
            valid[:, None] & (head[None, :] < HEADS),
        )


def compute_ngram_ids_gpu(
    input_ids,
    query_start_loc,
    context,
    multipliers,
    vocab_sizes,
    vocab_offsets,
    *,
    heads_per_ngram,
    eos_token_id,
    num_reqs,
    num_tokens,
    out,
):
    """Write exact table row IDs, without reading token values back to CPU."""
    tensors = (
        input_ids,
        query_start_loc,
        context,
        multipliers,
        vocab_sizes,
        vocab_offsets,
        out,
    )
    if any(
        t.device != input_ids.device or t.device.type != "cuda" or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("GPU N-gram hashing requires contiguous tensors on one GPU")
    if any(
        t.dtype not in (torch.int32, torch.int64)
        for t in (input_ids, query_start_loc, context)
    ):
        raise ValueError("Token IDs, offsets and context must be integer tensors")
    if any(t.dtype != torch.int64 for t in (multipliers, vocab_sizes, vocab_offsets)):
        raise ValueError("Hash metadata must use int64")
    order = multipliers.numel()
    heads = (order - 1) * heads_per_ngram
    if order < 2 or heads_per_ngram < 1 or out.dtype != torch.int64:
        raise ValueError("Invalid N-gram hash dtype/order")
    if not (
        0 <= num_reqs < query_start_loc.numel() and 0 <= num_tokens <= input_ids.numel()
    ):
        raise ValueError("Invalid packed request/token counts")
    if (
        context.ndim != 2
        or context.shape[1] != order - 1
        or context.shape[0] < num_reqs
    ):
        raise ValueError("Invalid N-gram context shape")
    if (
        vocab_sizes.numel() != heads
        or vocab_offsets.numel() != heads
        or out.numel() < num_tokens * heads
    ):
        raise ValueError("Invalid hash head metadata or output capacity")
    if num_tokens == 0:
        return out[:0]
    if num_reqs == 0:
        raise ValueError("Nonempty tokens require requests")
    _hash_ids[(tr.cdiv(num_tokens, 64),)](
        input_ids,
        query_start_loc,
        context,
        multipliers,
        vocab_sizes,
        vocab_offsets,
        out,
        num_reqs,
        num_tokens,
        order,
        heads_per_ngram,
        eos_token_id,
        context.shape[0].bit_length() + 1,
        64,
        tr.next_power_of_2(heads_per_ngram),
        num_warps=4,
    )
    return out[:num_tokens]

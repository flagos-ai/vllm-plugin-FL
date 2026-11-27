# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar (BI-V150 / corex) paged-attention kernels, phase 2 ("E4").

Imported only from ``unified_attention_iluvatar.unified_attention`` (itself
installed only when ``current_platform.vendor_name == "iluvatar"``) and only
reached when that wrapper's E2 gate ``_il_applies`` holds: causal, 16-bit
q/kv, no sliding window / chunked attention / mm-prefix / alibi / qq-bias /
sinks / softcap / kv-quant / fp8 output / tensor descriptors.  Everything else
keeps the E1/E2/E3 kernel in that file.  Pure Triton: no vLLM import.

Kernels
-------
``_il_prefill_kernel``  2D path, used for every batch with max_query_len > 1.
    Same work split as the tuned E2/E3 launch (BLOCK_M rows = BLOCK_Q query
    tokens x QPK query heads of one kv head; TILE keys per step), plus
      P1  loop-invariant paged address math: the in-block offset
          ``(key % BLOCK_SIZE) * s1 + kv_head * s2`` is hoisted out of the key
          loop (TILE % BLOCK_SIZE == 0), per tile only ``block * s0`` is added,
          and K and V share one offset vector (unbind views, equal strides);
      P2  mask-free main loop: tiles whose keys are all <= the q-block's first
          query position (all of them but the last 1-2 for a long-context
          chunk) skip the causal / length compare-select and use unmasked loads;
      P3  exp2 with log2(e) folded into the softmax scale;
      P4  CTA order knob: natural / heavy-first / kv-head-interleaved;
      P5  SKIP_DECODE: sequences with query_len == 1 return immediately, so a
          mixed batch can route its decode rows to the split-KV kernel below;
      P6  PREFETCH: explicit double buffering of the mask-free interior loop
          (the compiler's num_stages > 1 pipelining is useless on corex).
``_il_decode_kernel``   flash-decoding (split-KV) kernel for query_len == 1
    rows, grid (num_seqs, num_kv_heads // HPC, NSEG): no binary search and no
    dead q-blocks (upstream launches q.shape[0] // BLOCK_Q + num_seqs q-blocks
    for decode, a third of which exit at once).  The shipped CONFIG uses
    HPC = 1 (one kv head per CTA).  HPC = 2 puts both kv heads of a
    sequence into one CTA: the 16 query heads fill BLOCK_M = 16 exactly
    (upstream: 8 live rows of 16) and each key's K (and V) is one contiguous
    512-B row of the (blocks, 2, 16, 2, 128) cache read by one CTA instead of
    two interleaved 256-B halves read by two CTAs.  The score tile is
    block-diagonal (row head h only sees columns of kv head h // QPK); the
    masked half costs the same MMA work as upstream's 8 dead rows.  Optional
    register prefetch of the next tile (PREFETCH).
    HPC = 2 is NOT usable on this device and is kept only as a knob: measured
    2026-10-23 on BI-V150 it returns wrong rows (row err 1.1-65 on every case,
    including a 1-sequence one) at num_warps 1 and 2, with or without K_TRANS,
    and at num_warps 4 it is correct but only 0.28-0.91x of E1-E3.  Do not
    raise CONFIG["decode"]["hpc"] without re-running work/R4/T2-ilu-attn2
    (bench_t2.py --mode confirm) on the device.
``_il_decode1_kernel``  phase-3 ("E5") decode kernel, the shipped one: the
    specialisation of the above to one KV page per tile (TILE == BLOCK_SIZE,
    HPC == 1), which is the only tile size this device wants.  Same grid,
    same partial layout, same reduce.  See its own docstring for D1-D3; the
    short version is that the device is register/occupancy bound here, so the
    kernel is shaped to run at num_warps = 1 with nothing prefetched into
    registers, and the page-table dependency is taken off the critical path
    with a scalar software pipeline instead.  Measured on BI-V150 against the
    phase-2 kernel above (work/R7/ilu-decattn, real paged cache + scattered
    block table, 42-launch CUDA-graph replay, A/B interleaved on one stream,
    every configuration checked against an fp32 torch reference):
      64 rows / ctx 4864   947 -> 751 us (336 -> 424 GB/s)
      31 rows / ctx 16896 1498 -> 1123 us (358 -> 478 GB/s)
    and with the then-shipped NSEG policy (16 segments at 64 rows, 32 at 31
    rows) 1.268x / 1.355x of phase 2 end to end including the reduce.  R8/T3
    then retuned that policy to 2 / 4 segments (about 250 CTAs instead of
    2048) for a further 1.178x / 1.030x -- see the CONFIG comment below.  The
    kernel source itself is unchanged by R8/T3: the measurements that could
    have changed it all lost (see "measured and rejected" below).
``_il_reduce_kernel``   merges the NSEG partials of each (seq, q head); grid
    (num_seqs, num_q_heads); sequences with query_len != 1 return.

All kernels start the softmax denominator at 0 (E1: immune to the corex
``exp(-inf)`` first-tile miscompilation) and keep running maxima in the log2
domain.  Decode partials live in the attention builder's ``softmax_segm_*``
buffers used as flat storage indexed by ``(seq * num_q_heads + head) * NSEG +
seg``; NSEG is picked on the host from num_seqs and the buffer capacity, so
nothing is allocated per step and CUDA-graph capture sees fixed shapes (grid =
f(num_seqs) only; per-sequence lengths are read on the device).
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

LOG2E = 1.4426950408889634

# Key tile for the prefill kernel's unmasked interior; <= the diagonal TILE
# turns the whole R9/T6 feature off (qmma with it) and restores the shipped
# kernel exactly.  See the "prefill" entry of CONFIG below.
_PF_TILE2 = int(os.environ.get("VLLM_FL_IL_PF_TILE2", "256"))

# ---------------------------------------------------------------------------
# Launch configuration (BI-V150).  These are the values measured on the device
# (work/R4/T2-ilu-attn2, 2026-10-23: full sweep + a same-process A/B confirm
# run against the installed E1-E3 path, bf16, MiniCPM5-2B shapes 16 q / 2 kv
# heads, head_size 128, block_size 16); bench_t2.py prints its recommendation
# in exactly this shape.  Measured per-layer speedups vs E1-E3: decode 1.31x
# (64 seqs, 4.6k ctx) / 1.28x (30 seqs, 16.9k ctx), prefill chunks 1.41-1.52x,
# mixed steps 1.54-1.73x.
# The decode entry was re-measured 2026-10-24 (work/R7/ilu-decattn, see
# results.json there): the shipped num_warps=2 + register prefetch was giving
# 336-358 GB/s of a ~550 GB/s KV-read ceiling, and the device turns out to
# want the opposite shape -- one warp per CTA, nothing prefetched into
# registers, as many CTAs as the scratch allows.  TILE above 16 loses at every
# warp count (16 -> 32 -> 64 -> 128 is monotonically worse), maxnreg below 128
# spills, and num_stages=2 costs 2x.  Hence num_warps 1 / prefetch off /
# one_page on.  R8/T3 (work/R8/T3-ilu-decode-bw) then retuned the segment
# policy below and re-measured the alternatives that this shape suggests;
# measured and rejected on this device, do not retry:
#   * the K access pattern is NOT the limiter.  The paged gather alone (same
#     grid, same page-id pipeline, softmax and both MMAs deleted) runs at
#     639 GB/s at 4k and 651 at 16k, above the 575 GB/s this device gives for
#     a plain copy, and K and V gather at the same rate (623 vs 622 GB/s).  A
#     contiguous page table is only 0.5% faster than a scattered one.  So the
#     kernel's 415-487 GB/s is 65-75% memory and 25-35% non-overlapped work,
#     and the ceiling is occupancy, not the access pattern.
#   * indexing the stride-kv_s1 axis fastest instead of loading K
#     head-contiguous and transposing in registers (k_trans False, which is
#     what _il_kv_tile does): 0.844x / 0.811x.  Keep k_trans True.
#   * a scalar base pointer plus one loop-invariant int32 index tile instead
#     of rebuilding the int64 pointer tile each iteration, with or without
#     tl.max_contiguous/tl.multiple_of hints: 0.997x.  The compiler already
#     strength-reduces this.
#   * consuming 2 or 4 pages per loop iteration while keeping the page ids
#     scalar (so the wider tile costs no block-table gather), which widens the
#     QK dot to N = 32/64 and halves/quarters the acc rescale: 0.944x / 0.639x,
#     and 0.803x at the retuned nseg.  The device wants a 16-key tile for
#     register reasons, not because of the address math.
#   * making the kv head the fastest grid axis so the two CTAs that split a
#     512-B KV slot are adjacent program ids: 1.011x / 0.998x, i.e. noise.
# VLLM_FL_IL_ATTN_V2=0 turns the whole E4 path off (dev A/B only).
# ---------------------------------------------------------------------------
CONFIG = {
    "enable": os.environ.get("VLLM_FL_IL_ATTN_V2", "1") != "0",
    # 2D / prefill kernel.  Retuned 2026-10-24 (work/R7/ilu-prefill): with
    # BLOCK_M = 256 and num_warps = 16 fixed (0.25 rows per thread is the
    # measured sweet spot on 16 SMs), the two remaining levers are the key
    # tile and the pipeline depth, and they interact through the 128 KiB
    # shared-memory budget:
    #   * TILE = 128 makes the QK dot BN = 128 instead of 64 and halves the
    #     per-key cost of the acc rescale; tl.dot on corex reaches 85.5 TF/s
    #     at 256x128x128 against 68.6 at 256x64x128.
    #   * num_stages = 2 is what the extra tile size pays for: Q and P live
    #     in registers here, so shared holds only the pipelined K/V buffers
    #     (2 x 64 KiB at TILE = 128, 65 KiB reported).  num_stages = 3 asks
    #     for 129 KiB and does not fit; TILE = 256 asks for 192 KiB.
    #   * no maxnreg cap: the register budget this kernel wants is 152, and
    #     capping at 128 costs 22 spills and 4% of the gain.
    # Measured per layer against the previous (tile 64, stages 1, maxnreg
    # 128) values on the real paged layout: 2048-token chunk at context 0
    # 0.635 -> 0.559 ms, 2k 1.610 -> 1.329, 6k 3.567 -> 2.886, 14k 7.557 ->
    # 6.081 (34.1 -> 42.4 TF/s), fp32-reference row error unchanged at
    # 3.2e-3.
    #   * qmma / tile2 were added 2026-10-25 (work/R9/T6-ilu-prefill-pipeline).
    #     R8/T5 attributed 73% of this kernel's time to non-overlapped load
    #     latency, 9.73 s (4k) / 81.88 s (16k) of it K's alone, because the
    #     pipeliner gave K a SINGLE shared slot: Q 64 KiB + K 32 KiB + V
    #     32 KiB is exactly the 128 KiB cap, so K's async copy is issued at
    #     the end of body j and drained at the top of body j+1.  Three things
    #     were measured and all lost: a K-only register prefetch, 0.87x at
    #     every prefix (a loop-carried K leaves the async path and pays a
    #     global -> register -> shared round trip, which costs more than the
    #     latency it hides -- the same reason prefetch=True lost); more
    #     pipeline depth, 0.62-0.72x, because num_stages 3 and 4 add only the
    #     two 512-B block-table slots (+1024 B, which is the whole of the
    #     "129 KiB" num_stages=3 refusal) and never a second K slot, however
    #     much budget they are given; and a hand-unrolled interior that puts
    #     two K copies in flight, 0.67x at 256 registers with 86 spills
    #     (0.94x at num_stages=1, no spills).  BLOCK_M = 128 does not invert
    #     under any of this either: 0.42-0.46x, because BLOCK_Q halves to 16
    #     and the KV tile traffic doubles.
    #     What does work is spending the freed budget on a wider key tile.
    #     qmma=True multiplies Q by the identity once so Q reaches the MMA in
    #     an #mma layout instead of through a shared buffer that stays live
    #     across the loop (131072 -> 98304 B, 152 -> 175 registers, 0 spills,
    #     row error unchanged); that is worth about +2% on its own at depth
    #     and -2.8% at prefix 0 (one extra dot per q-block).  tile2=256 then
    #     spends the 32 KiB on a 256-key unmasked interior -- the shape the
    #     cap refused at 192 KiB -- while the masked diagonal stays at
    #     TILE=128 so no MMA work is spent on keys the causal mask discards
    #     (a uniform TILE=256 is 1.049x at prefix 14k but 0.880x at prefix 0;
    #     the split is better at every prefix).  Measured per layer on a
    #     2048-token chunk, 42-launch CUDA-graph replay, A/B interleaved
    #     against the installed kernel, scattered page table, every
    #     configuration checked against an fp32 reference:
    #       prefix     0     2k     6k    14k
    #       kernel  0.974  1.040  1.064  1.075
    #       forward 0.977  1.044  1.068  1.079   (42-layer il_attn_v2.forward)
    #     42.8 -> 46.0 TF/s at prefix 14k.  VLLM_FL_IL_PF_TILE2=0 restores the
    #     shipped kernel exactly (both knobs off together).
    "prefill": dict(block_m=256, tile=128, num_warps=16, num_stages=2,
                    split_mask=True, order=1, k_trans=False, prefetch=False,
                    extra=None, qmma=_PF_TILE2 > 128, tile2=_PF_TILE2),
    # split-KV decode kernel (+ reduce)
    #   nseg       base number of KV segments per sequence (power of 2)
    #   nseg_max   upper bound when growing nseg for small batches
    #   min_ctas   grow nseg (<= nseg_max, within buffer capacity) until
    #              num_seqs * kv_groups * nseg >= min_ctas
    #   one_page    use _il_decode1_kernel (TILE == BLOCK_SIZE, HPC == 1)
    #   split_mask  unmasked interior loop, masked last tile only
    #   bt_pipe     page-id software pipeline depth (0 = none)
    # The segment policy was retuned 2026-10-25 (work/R8/T3-ilu-decode-bw, see
    # DESIGN.md there).  R7 left nseg at the phase-2 value because its sweep
    # used a uniform context length per batch, which flatters few-segment
    # configurations; redone on the real spread (64 rows at ctx 4096..5120, 31
    # rows at 16384..17408, fp32-reference-checked, 42-launch CUDA-graph
    # replay, A/B interleaved) the optimum is not 2048 CTAs but about 250:
    #     4k, 64 rows   nseg 1/2/4/8/16  -> 0.675 / 1.178 / 1.078 / 1.120 / 1.0
    #     16k, 31 rows  nseg 2/4/8/16/32 -> 0.586 / 1.030 / 0.934 / 0.980 / 1.0
    # 2048 one-warp CTAs run only ~19 tiles each and pay their prologue (Q
    # load) and epilogue (an [BLOCK_M, HEAD] fp32 partial) over far too little
    # work; ~250 CTAs of ~150 tiles amortise it and still saturate the paged
    # gather (489 GB/s at 4k against a 639 GB/s gather-only ceiling measured
    # with the same grid and page pipeline but no softmax and no MMA).  The
    # reduce is not the reason: it is 3.8 us of 734.9 at 4k.
    # min_ctas = 248 is pinned from both ends -- it is the lowest CTA count
    # measured healthy (248 is the 31-row optimum at both contexts) and the
    # highest threshold that still lets 64 rows use 2 segments (64*2*2 = 256).
    # 192 would give 48-56 rows only 192-224 CTAs, 256 would give 31 rows 8
    # segments (0.934x).  nseg_max stays 32 so batches of <= 4 rows keep
    # exactly today's behaviour.  Off-diagonal check, since the policy keys on
    # rows: 64 rows at 16k context -> nseg 2, 1.063x; 31 rows at 4k context ->
    # nseg 4, 1.009x (nseg 2 there would be 0.587x, which is what min_ctas
    # exists to prevent).  Whole-step (42 layers, qkv/attn/o/MLP/lm head, A/B
    # interleaved CUDA graphs): 4k 42.406 -> 37.735 ms (1.124x), 16k 56.612 ->
    # 55.203 ms (1.026x).  Uses *less* scratch than before (128 of vLLM's 1024
    # (seq, segment) pairs at 64 rows instead of 1024), so the memory
    # high-water mark that sizes the KV cache is untouched, and the grid still
    # depends only on num_seqs, so FULL_DECODE_ONLY capture is unaffected.
    # Re-tuned at the new nseg, every other knob is unchanged and still best:
    # num_warps 2/4 = 0.82/0.50x, two pages per tile = 0.80x, block_m 16 =
    # 1.109x vs 1.179x, bt_pipe 1 = 1.178x, K/V register prefetch = 1.088x.
    "decode": dict(hpc=1, block_m=0, tile=16, num_warps=1, num_stages=1,
                   nseg=2, nseg_max=32, min_ctas=248, prefetch=False,
                   k_trans=True, reduce_warps=1, extra=None,
                   one_page=True, split_mask=True, bt_pipe=2),
    # route the query_len == 1 rows of a mixed batch to the decode kernel
    "split_mixed": True,
}


def _np2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


# ===========================================================================
# shared helpers
# ===========================================================================
@triton.jit
def _il_find_seq(qsl_ptr, target, num_seqs, BLOCK_Q: tl.constexpr):
    """Sequence owning q-block ``target`` (q-blocks laid out as
    ``qsl[s] // BLOCK_Q + s``, identical to upstream ``find_seq_idx``)."""
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(qsl_ptr + mid) // BLOCK_Q + mid
        if val <= target:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _il_kv_tile(k_ptr, v_ptr, bt_row, j, c_t, offs_d, inb, c_head_off,
                key_lim, ok_extra, kv_s0, kv_s1,
                BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr,
                MASKED: tl.constexpr, HOIST: tl.constexpr,
                K_TRANS: tl.constexpr):
    """Load one paged K^T (HEAD, C) and V (C, HEAD) tile.

    Column c holds key ``j * TILE + c_t[c]`` of the kv head whose cache offset
    is ``c_head_off[c]``; ``inb`` = hoisted in-block offset (valid when
    TILE % BLOCK_SIZE == 0).  MASKED loads zero keys >= key_lim (and
    everything when ok_extra is false); unmasked loads assume all keys valid.
    """
    pos = j * TILE + c_t
    if MASKED:
        key_ok = (pos < key_lim) & ok_extra
        blk = tl.load(bt_row + pos // BLOCK_SIZE, mask=key_ok, other=0)
    else:
        key_ok = pos < key_lim  # unused by callers in this mode
        blk = tl.load(bt_row + pos // BLOCK_SIZE)
    if HOIST:
        off = blk.to(tl.int64) * kv_s0 + inb
    else:
        off = blk.to(tl.int64) * kv_s0 + ((pos % BLOCK_SIZE) * kv_s1 + c_head_off)
    if MASKED:
        if K_TRANS:
            K = tl.trans(tl.load(k_ptr + off[:, None] + offs_d[None, :],
                                 mask=key_ok[:, None], other=0.0))
        else:
            K = tl.load(k_ptr + off[None, :] + offs_d[:, None],
                        mask=key_ok[None, :], other=0.0)
        V = tl.load(v_ptr + off[:, None] + offs_d[None, :],
                    mask=key_ok[:, None], other=0.0)
    else:
        if K_TRANS:
            K = tl.trans(tl.load(k_ptr + off[:, None] + offs_d[None, :]))
        else:
            K = tl.load(k_ptr + off[None, :] + offs_d[:, None])
        V = tl.load(v_ptr + off[:, None] + offs_d[None, :])
    return K, V, pos, key_ok


@triton.jit
def _il_softmax_pv(acc, M, L, Q, K, V, S_mask, scale_log2,
                   USE_MASK: tl.constexpr):
    """One online-softmax step in the log2 domain; L starts at 0 (E1)."""
    S = tl.dot(Q, K) * scale_log2
    if USE_MASK:
        S = tl.where(S_mask, S, float("-inf"))
    m_new = tl.maximum(M, tl.max(S, axis=1))
    if USE_MASK:
        # a fully masked row (padding rows only) must not produce NaN
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
    P = tl.exp2(S - m_new[:, None])
    alpha = tl.exp2(M - m_new)
    L = L * alpha + tl.sum(P, axis=1)
    acc = acc * alpha[:, None]
    acc = tl.dot(P.to(V.dtype), V, acc)
    return acc, m_new, L


# ===========================================================================
# 2D (prefill / mixed) kernel
# ===========================================================================
@triton.jit(do_not_specialize=["num_seqs", "n_qblocks"])
def _il_prefill_kernel(
    out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, seq_lens_ptr, qsl_ptr,
    scale_log2, num_seqs, n_qblocks,
    q_s0, q_s1, o_s0, o_s1, bt_s0, kv_s0, kv_s1, kv_s2,
    NUM_KV: tl.constexpr, QPK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_Q: tl.constexpr, HEAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr,
    SKIP_DECODE: tl.constexpr, SPLIT_MASK: tl.constexpr,
    ORDER: tl.constexpr, K_TRANS: tl.constexpr,
    PREFETCH: tl.constexpr, QMMA: tl.constexpr, TILE2: tl.constexpr,
):
    # ORDER: 0 natural (q-block fastest, kv head = axis 1), 1 heavy-first,
    #        2 kv-head-interleaved 1D grid, 3 = 2 + heavy-first
    # QMMA:  convert Q to an #mma layout once so its 64 KiB shared staging is
    #        dead inside the loop (see the CONFIG comment and R9/T6)
    # TILE2: key tile for the unmasked interior (0 / <= TILE: use TILE); the
    #        masked diagonal always runs at TILE.  Needs QMMA to fit.
    if ORDER >= 2:
        flat = tl.program_id(0)
        kvh = flat % NUM_KV
        pid = flat // NUM_KV
    else:
        pid = tl.program_id(0)
        kvh = tl.program_id(1)
    if ORDER % 2 == 1:
        pid = n_qblocks - 1 - pid

    seq = _il_find_seq(qsl_ptr, pid, num_seqs, BLOCK_Q)
    q_start = tl.load(qsl_ptr + seq)
    q_len = tl.load(qsl_ptr + seq + 1) - q_start
    qb = pid - (q_start // BLOCK_Q + seq)
    if qb * BLOCK_Q >= q_len:
        return
    if SKIP_DECODE:
        if q_len == 1:
            return
    seq_len = tl.load(seq_lens_ptr + seq)
    ctx = seq_len - q_len

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    offs_t = tl.arange(0, TILE)
    tok = qb * BLOCK_Q + offs_m // QPK
    head = kvh * QPK + offs_m % QPK
    row_ok = tok < q_len
    q_abs = ctx + tok
    row_off = (q_start + tok).to(tl.int64)

    Q = tl.load(q_ptr + row_off[:, None] * q_s0 + head[:, None] * q_s1
                + offs_d[None, :], mask=row_ok[:, None], other=0.0)

    if QMMA:
        # Take Q's 64 KiB shared staging out of the loop's live set.
        #
        # The MMA on this backend wants its A operand in a #ttg.dot_op
        # layout, and a *loaded* (#blocked) tensor can only reach that layout
        # through shared memory: the compiler local_allocs Q once and
        # local_loads it on every iteration, so the buffer is live across the
        # whole loop and eats half of the 128 KiB per-CTA budget.  An
        # #mma-layout tensor converts to dot_op in registers for free -- that
        # is how P reaches the PV dot below.  Multiplying Q by the identity
        # once produces exactly the same bf16 values in an #mma layout (an
        # fp32 MMA of bf16 values against exact 1.0 / 0.0 is exact, and
        # fp32 -> bf16 of an exactly representable bf16 value is the
        # identity), after which the only shared buffers live inside the loop
        # are the K and V tiles.  Costs one extra BLOCK_M x HEAD x HEAD dot
        # per q-block; what it buys is TILE2 (below), which the 128 KiB cap
        # refused before.
        eye = tl.where(offs_d[:, None] == offs_d[None, :], 1.0, 0.0)
        Q = tl.dot(Q, eye.to(q_ptr.dtype.element_ty)).to(q_ptr.dtype.element_ty)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)  # E1
    acc = tl.zeros([BLOCK_M, HEAD], dtype=tl.float32)

    bt_row = bt_ptr + seq * bt_s0  # seq * stride < 2**31 (<= max_num_seqs rows)
    p_first = ctx + qb * BLOCK_Q
    p_last = tl.minimum(p_first + BLOCK_Q - 1, seq_len - 1)
    n_tiles = p_last // TILE + 1
    if SPLIT_MASK:
        n_full = (p_first + 1) // TILE
    else:
        n_full = 0
    WIDE: tl.constexpr = TILE2 > TILE
    if WIDE:
        # the unmasked part of the row runs on TILE2-key tiles -- one wide
        # async copy and one wide MMA per iteration instead of TILE2 / TILE
        # narrow ones -- while the diagonal keeps the TILE-key granularity, so
        # no MMA work is spent on keys the causal mask throws away.  Keys
        # between the last whole TILE2 tile and p_first are unmasked but are
        # picked up by the masked loop, which is correct for any tile: at most
        # TILE2 / TILE - 1 tiles per q-block pay the mask for nothing.
        n_wide = (p_first + 1) // TILE2
        j_lo = n_wide * (TILE2 // TILE)
    else:
        n_wide = 0
        j_lo = n_full

    HOIST: tl.constexpr = (TILE % BLOCK_SIZE) == 0
    head_off = kvh * kv_s2
    inb = (offs_t % BLOCK_SIZE) * kv_s1 + head_off

    # unmasked interior: every key < (p_first + 1) is visible to every row
    if WIDE:
        offs_t2 = tl.arange(0, TILE2)
        HOIST2: tl.constexpr = (TILE2 % BLOCK_SIZE) == 0
        inb2 = (offs_t2 % BLOCK_SIZE) * kv_s1 + head_off
        for j2 in range(0, n_wide):
            K, V, pos, key_ok = _il_kv_tile(
                k_ptr, v_ptr, bt_row, j2, offs_t2, offs_d, inb2, head_off,
                seq_len, seq_len > 0, kv_s0, kv_s1,
                BLOCK_SIZE, TILE2, False, HOIST2, K_TRANS)
            acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V, row_ok[:, None],
                                       scale_log2, False)
    elif PREFETCH:
        # explicit double buffering: corex gets nothing from the compiler's
        # software pipelining (num_stages > 1 always lost in phase 1), so the
        # next tile's loads are issued by hand before this tile's two dots.
        # The j + 1 index is clamped to the last tile, so no load ever leaves
        # the key range the unmasked path is allowed to touch.
        if n_full > 0:
            K, V, pos, key_ok = _il_kv_tile(
                k_ptr, v_ptr, bt_row, 0, offs_t, offs_d, inb, head_off,
                seq_len, seq_len > 0, kv_s0, kv_s1,
                BLOCK_SIZE, TILE, False, HOIST, K_TRANS)
            for j in range(0, n_full):
                jn = tl.minimum(j + 1, n_full - 1)
                Kn, Vn, posn, okn = _il_kv_tile(
                    k_ptr, v_ptr, bt_row, jn, offs_t, offs_d, inb, head_off,
                    seq_len, seq_len > 0, kv_s0, kv_s1,
                    BLOCK_SIZE, TILE, False, HOIST, K_TRANS)
                acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V, row_ok[:, None],
                                           scale_log2, False)
                K = Kn
                V = Vn
    else:
        for j in range(0, n_full):
            K, V, pos, key_ok = _il_kv_tile(
                k_ptr, v_ptr, bt_row, j, offs_t, offs_d, inb, head_off,
                seq_len, seq_len > 0, kv_s0, kv_s1,
                BLOCK_SIZE, TILE, False, HOIST, K_TRANS)
            acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V, row_ok[:, None],
                                       scale_log2, False)
    # diagonal / tail tiles: causal + length masks
    for j in range(j_lo, n_tiles):
        K, V, pos, key_ok = _il_kv_tile(
            k_ptr, v_ptr, bt_row, j, offs_t, offs_d, inb, head_off,
            seq_len, seq_len > 0, kv_s0, kv_s1,
            BLOCK_SIZE, TILE, True, HOIST, K_TRANS)
        acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V,
                                   pos[None, :] <= q_abs[:, None],
                                   scale_log2, True)

    acc = acc / L[:, None]
    tl.store(out_ptr + row_off[:, None] * o_s0 + head[:, None] * o_s1
             + offs_d[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=row_ok[:, None])


# ===========================================================================
# split-KV decode kernel + reduce
# ===========================================================================
@triton.jit
def _il_decode1_kernel(
    part_ptr, pmax_ptr, psum_ptr,
    q_ptr, k_ptr, v_ptr, bt_ptr, seq_lens_ptr, qsl_ptr,
    scale_log2, q_s0, q_s1, bt_s0, kv_s0, kv_s1, kv_s2,
    NUM_Q_HEADS: tl.constexpr, QPK: tl.constexpr,
    BLOCK_M: tl.constexpr, HEAD: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    NSEG: tl.constexpr, PREFETCH: tl.constexpr, SPLIT_MASK: tl.constexpr,
    BT_PIPE: tl.constexpr,
):
    """One-page-per-tile split-KV decode (TILE == BLOCK_SIZE, HPC == 1).

    Differences from ``_il_decode_kernel``:
      D1 the page number is a *scalar* load (all TILE lanes of the generic
         kernel load the same block-table entry, a 16-lane gather of one
         value), so the address math costs one uniform load and one broadcast;
      D2 BT_PIPE lifts that scalar load out of the critical path: the page
         number for tile j+1 (and j+2) is issued before tile j's K/V loads,
         which removes the ``block_table -> kv_cache`` dependent-load chain
         from the per-iteration latency without the register cost of a full
         K/V prefetch (one scalar vs two TILE x HEAD tiles);
      D3 SPLIT_MASK: only the last tile of a sequence can be partial, so the
         interior loop loads and scores unmasked (no predicate registers, no
         ``where`` on the score tile, no ``-inf`` fixup).
    The device prefers many tiny CTAs here (num_warps=1, no K/V prefetch):
    everything above is aimed at keeping per-thread register use low enough
    for that.
    """
    seq = tl.program_id(0)
    grp = tl.program_id(1)
    seg = tl.program_id(2)
    q_start = tl.load(qsl_ptr + seq)
    q_len = tl.load(qsl_ptr + seq + 1) - q_start
    if q_len != 1:
        return
    seq_len = tl.load(seq_lens_ptr + seq)
    n_tiles = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    tps = tl.maximum((n_tiles + NSEG - 1) // NSEG, 1)
    lo = seg * tps
    hi = tl.minimum(lo + tps, n_tiles)
    if lo >= hi:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    offs_c = tl.arange(0, BLOCK_SIZE)
    head = grp * QPK + offs_m
    row_ok = offs_m < QPK

    Q = tl.load(q_ptr + q_start.to(tl.int64) * q_s0 + head[:, None] * q_s1
                + offs_d[None, :], mask=row_ok[:, None], other=0.0)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)  # E1
    acc = tl.zeros([BLOCK_M, HEAD], dtype=tl.float32)

    inb = offs_c * kv_s1 + grp * kv_s2
    bt_row = bt_ptr + seq * bt_s0
    # last tile is partial iff seq_len % BLOCK_SIZE != 0; everything before it
    # is fully inside the sequence
    if SPLIT_MASK:
        hi_full = tl.minimum(hi, seq_len // BLOCK_SIZE)
    else:
        hi_full = lo

    # bt_row index clamp: never read outside [lo, hi_full-1] (and never below
    # lo, which matters when the interior loop is empty)
    jmax = tl.maximum(lo, hi_full - 1)
    dummy = row_ok[:, None]

    # ---- interior: every key valid -------------------------------------
    if BT_PIPE > 0:
        b0 = tl.load(bt_row + lo)
        b1 = tl.load(bt_row + tl.minimum(lo + 1, jmax)) if BT_PIPE > 1 else b0
        for j in range(lo, hi_full):
            if BT_PIPE > 1:
                bn = tl.load(bt_row + tl.minimum(j + 2, jmax))
            else:
                bn = tl.load(bt_row + tl.minimum(j + 1, jmax))
            off = b0.to(tl.int64) * kv_s0 + inb
            K = tl.trans(tl.load(k_ptr + off[:, None] + offs_d[None, :]))
            V = tl.load(v_ptr + off[:, None] + offs_d[None, :])
            acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V, dummy, scale_log2,
                                       False)
            if BT_PIPE > 1:
                b0 = b1
                b1 = bn
            else:
                b0 = bn
    elif PREFETCH:
        off = tl.load(bt_row + lo).to(tl.int64) * kv_s0 + inb
        K = tl.trans(tl.load(k_ptr + off[:, None] + offs_d[None, :]))
        V = tl.load(v_ptr + off[:, None] + offs_d[None, :])
        for j in range(lo, hi_full):
            offn = (tl.load(bt_row + tl.minimum(j + 1, jmax)).to(tl.int64)
                    * kv_s0 + inb)
            Kn = tl.trans(tl.load(k_ptr + offn[:, None] + offs_d[None, :]))
            Vn = tl.load(v_ptr + offn[:, None] + offs_d[None, :])
            acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V, dummy, scale_log2,
                                       False)
            K = Kn
            V = Vn
    else:
        for j in range(lo, hi_full):
            off = tl.load(bt_row + j).to(tl.int64) * kv_s0 + inb
            K = tl.trans(tl.load(k_ptr + off[:, None] + offs_d[None, :]))
            V = tl.load(v_ptr + off[:, None] + offs_d[None, :])
            acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V, dummy, scale_log2,
                                       False)

    # ---- tail: at most one partial tile --------------------------------
    for j in range(hi_full, hi):
        pos = j * BLOCK_SIZE + offs_c
        key_ok = pos < seq_len
        off = tl.load(bt_row + j).to(tl.int64) * kv_s0 + inb
        K = tl.trans(tl.load(k_ptr + off[:, None] + offs_d[None, :],
                             mask=key_ok[:, None], other=0.0))
        V = tl.load(v_ptr + off[:, None] + offs_d[None, :],
                    mask=key_ok[:, None], other=0.0)
        acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V, key_ok[None, :],
                                   scale_log2, True)

    idx = (seq * NUM_Q_HEADS + head) * NSEG + seg
    tl.store(part_ptr + idx.to(tl.int64)[:, None] * HEAD + offs_d[None, :],
             acc, mask=row_ok[:, None])
    tl.store(pmax_ptr + idx, M, mask=row_ok)
    tl.store(psum_ptr + idx, L, mask=row_ok)


@triton.jit
def _il_decode_kernel(
    part_ptr, pmax_ptr, psum_ptr,
    q_ptr, k_ptr, v_ptr, bt_ptr, seq_lens_ptr, qsl_ptr,
    scale_log2, q_s0, q_s1, bt_s0, kv_s0, kv_s1, kv_s2,
    NUM_Q_HEADS: tl.constexpr, QPK: tl.constexpr, HPC: tl.constexpr,
    BLOCK_M: tl.constexpr, HEAD: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr, NSEG: tl.constexpr,
    PREFETCH: tl.constexpr, K_TRANS: tl.constexpr,
):
    seq = tl.program_id(0)
    grp = tl.program_id(1)
    seg = tl.program_id(2)
    q_start = tl.load(qsl_ptr + seq)
    q_len = tl.load(qsl_ptr + seq + 1) - q_start
    if q_len != 1:
        return
    seq_len = tl.load(seq_lens_ptr + seq)
    n_tiles = (seq_len + TILE - 1) // TILE
    tps = tl.maximum((n_tiles + NSEG - 1) // NSEG, 1)
    lo = seg * tps
    hi = tl.minimum(lo + tps, n_tiles)
    if lo >= hi:
        return

    C: tl.constexpr = HPC * TILE
    ROWS: tl.constexpr = HPC * QPK
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    offs_c = tl.arange(0, C)
    head = grp * ROWS + offs_m
    row_ok = offs_m < ROWS
    c_t = offs_c % TILE
    c_head_off = (grp * HPC + offs_c // TILE) * kv_s2
    # row r (q head grp*ROWS + r) belongs to local kv head r // QPK; rows >= ROWS
    # are padding and match no column
    diag = (offs_m // QPK)[:, None] == (offs_c // TILE)[None, :]

    Q = tl.load(q_ptr + q_start.to(tl.int64) * q_s0 + head[:, None] * q_s1
                + offs_d[None, :], mask=row_ok[:, None], other=0.0)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)  # E1
    acc = tl.zeros([BLOCK_M, HEAD], dtype=tl.float32)

    HOIST: tl.constexpr = (TILE % BLOCK_SIZE) == 0
    inb = (c_t % BLOCK_SIZE) * kv_s1 + c_head_off
    bt_row = bt_ptr + seq * bt_s0

    if PREFETCH:
        K, V, pos, key_ok = _il_kv_tile(
            k_ptr, v_ptr, bt_row, lo, c_t, offs_d, inb, c_head_off,
            seq_len, hi > lo, kv_s0, kv_s1, BLOCK_SIZE, TILE, True, HOIST,
            K_TRANS)
        for j in range(lo, hi):
            Kn, Vn, posn, key_okn = _il_kv_tile(
                k_ptr, v_ptr, bt_row, j + 1, c_t, offs_d, inb, c_head_off,
                seq_len, (j + 1) < hi, kv_s0, kv_s1, BLOCK_SIZE, TILE, True,
                HOIST, K_TRANS)
            acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V,
                                       diag & key_ok[None, :], scale_log2,
                                       True)
            K = Kn
            V = Vn
            key_ok = key_okn
    else:
        for j in range(lo, hi):
            K, V, pos, key_ok = _il_kv_tile(
                k_ptr, v_ptr, bt_row, j, c_t, offs_d, inb, c_head_off,
                seq_len, hi > lo, kv_s0, kv_s1, BLOCK_SIZE, TILE, True, HOIST,
                K_TRANS)
            acc, M, L = _il_softmax_pv(acc, M, L, Q, K, V,
                                       diag & key_ok[None, :], scale_log2,
                                       True)

    idx = (seq * NUM_Q_HEADS + head) * NSEG + seg
    tl.store(part_ptr + idx.to(tl.int64)[:, None] * HEAD + offs_d[None, :],
             acc, mask=row_ok[:, None])
    tl.store(pmax_ptr + idx, M, mask=row_ok)
    tl.store(psum_ptr + idx, L, mask=row_ok)


@triton.jit
def _il_reduce_kernel(
    out_ptr, part_ptr, pmax_ptr, psum_ptr, seq_lens_ptr, qsl_ptr,
    o_s0, o_s1, n_tokens,
    NUM_Q_HEADS: tl.constexpr, HEAD: tl.constexpr, TILE: tl.constexpr,
    NSEG: tl.constexpr, ZERO_PAD: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    q_start = tl.load(qsl_ptr + seq)
    q_len = tl.load(qsl_ptr + seq + 1) - q_start
    offs_d = tl.arange(0, HEAD)
    if ZERO_PAD:
        # pure-decode CUDA-graph padding: padded request s (query_len 0) owns
        # the orphan token row s; write zeros there instead of leaving stale
        # graph-pool memory (upstream writes garbage).  Never used for mixed
        # batches, where seq index != token index.
        pad_row = (q_len == 0) & (seq < n_tokens)
        tl.store(out_ptr + seq.to(tl.int64) * o_s0 + h * o_s1 + offs_d,
                 tl.zeros([HEAD], dtype=tl.float32).to(out_ptr.dtype.element_ty),
                 mask=(offs_d < HEAD) & pad_row)
    if q_len != 1:
        return
    seq_len = tl.load(seq_lens_ptr + seq)
    n_tiles = (seq_len + TILE - 1) // TILE
    tps = tl.maximum((n_tiles + NSEG - 1) // NSEG, 1)
    n_act = (n_tiles + tps - 1) // tps
    s = tl.arange(0, NSEG)
    s_ok = s < n_act
    base = (seq * NUM_Q_HEADS + h) * NSEG
    m = tl.load(pmax_ptr + base + s, mask=s_ok, other=float("-inf"))
    m_all = tl.max(m, axis=0)
    w = tl.where(s_ok, tl.exp2(m - m_all), 0.0)
    lsum = tl.load(psum_ptr + base + s, mask=s_ok, other=0.0)
    L = tl.sum(lsum * w, axis=0)
    o = tl.load(part_ptr + (base + s).to(tl.int64)[:, None] * HEAD
                + offs_d[None, :], mask=s_ok[:, None], other=0.0)
    acc = tl.sum(o * w[:, None], axis=0) / L
    tl.store(out_ptr + q_start.to(tl.int64) * o_s0 + h * o_s1 + offs_d,
             acc.to(out_ptr.dtype.element_ty))


# ===========================================================================
# host-side launchers
# ===========================================================================
def _launch_kwargs(num_warps, num_stages, extra):
    kw = {"num_warps": int(num_warps), "num_stages": int(num_stages)}
    if extra:
        kw.update(extra)
    return kw


# Shapes whose wide-interior form did not fit this device's shared memory.
# The wide interior asks for exactly 128 KiB on BI-V150, which is the cap, so
# a device (or a future TILE2 / BLOCK_M) with less room must fall back to the
# narrow interior rather than fail the step.  The Triton JIT raises at compile
# time, i.e. on the first launch of a given constexpr set, so this costs one
# dictionary lookup afterwards and the retry never sees a half-run kernel.
_PF_NARROW: set = set()


def launch_prefill(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                   softmax_scale, *, block_m=128, tile=64, num_warps=8,
                   num_stages=1, split_mask=True, order=0, k_trans=False,
                   skip_decode=False, prefetch=False, extra=None,
                   num_seqs=None, qmma=False, tile2=0):
    """2D kernel over every q-block (``q.shape[0] // BLOCK_Q + num_seqs``)."""
    num_tokens, nq, hd = q.shape
    nkv = k.shape[2]
    qpk = nq // nkv
    block_q = block_m // qpk
    assert block_q >= 1 and block_q * qpk == block_m, (block_m, qpk)
    num_seqs = seqused_k.shape[0] if num_seqs is None else num_seqs
    n_qb = num_tokens // block_q + num_seqs
    grid = (n_qb * nkv,) if order >= 2 else (n_qb, nkv)
    t2 = int(tile2)
    if t2 <= tile or (t2 % tile) or (t2 % k.shape[1]):
        t2 = 0
    key = (block_m, tile, t2, int(num_warps), int(num_stages), bool(qmma), hd)
    if t2 and key in _PF_NARROW:
        t2 = 0

    def _go(t2_, qmma_):
        _il_prefill_kernel[grid](
            out, q, k, v, block_table, seqused_k, cu_seqlens_q,
            float(softmax_scale) * LOG2E, num_seqs, n_qb,
            q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            block_table.stride(0), k.stride(0), k.stride(1), k.stride(2),
            NUM_KV=nkv, QPK=qpk, BLOCK_M=block_m, BLOCK_Q=block_q, HEAD=hd,
            BLOCK_SIZE=k.shape[1], TILE=tile,
            SKIP_DECODE=bool(skip_decode), SPLIT_MASK=bool(split_mask),
            ORDER=int(order), K_TRANS=bool(k_trans), PREFETCH=bool(prefetch),
            QMMA=bool(qmma_), TILE2=int(t2_),
            **_launch_kwargs(num_warps, num_stages, extra),
        )

    if not t2:
        _go(0, bool(qmma))
        return
    try:
        _go(t2, True)
    except Exception as exc:                                   # noqa: BLE001
        _PF_NARROW.add(key)
        logger.warning(
            "il_attn_v2: prefill TILE2=%d does not fit this device (%s); "
            "falling back to the TILE=%d interior", t2, exc, tile)
        _go(0, bool(qmma))


def launch_decode(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                  softmax_scale, part, pmax, psum, *, hpc=2, block_m=0,
                  tile=32, nseg=16, num_warps=2, num_stages=1, prefetch=False,
                  k_trans=False, reduce_warps=1, extra=None, num_seqs=None,
                  do_reduce=True, zero_pad=False, one_page=False,
                  split_mask=True, bt_pipe=0):
    """Split-KV decode for every sequence with query_len == 1.

    ``part``/``pmax``/``psum`` are flat-addressed fp32 buffers holding at least
    ``num_seqs * nq * nseg`` (x head_size for ``part``) elements.
    """
    num_tokens, nq, hd = q.shape
    nkv = k.shape[2]
    qpk = nq // nkv
    assert nkv % hpc == 0, (nkv, hpc)
    num_seqs = seqused_k.shape[0] if num_seqs is None else num_seqs
    rows = hpc * qpk
    bm = block_m or _np2(rows)
    assert bm >= rows
    need = num_seqs * nq * nseg
    assert pmax.numel() >= need and psum.numel() >= need \
        and part.numel() >= need * hd, "decode scratch too small"
    if one_page and hpc == 1 and tile == k.shape[1]:
        _il_decode1_kernel[(num_seqs, nkv, nseg)](
            part, pmax, psum, q, k, v, block_table, seqused_k, cu_seqlens_q,
            float(softmax_scale) * LOG2E, q.stride(0), q.stride(1),
            block_table.stride(0), k.stride(0), k.stride(1), k.stride(2),
            NUM_Q_HEADS=nq, QPK=qpk, BLOCK_M=bm, HEAD=hd,
            BLOCK_SIZE=k.shape[1], NSEG=nseg, PREFETCH=bool(prefetch),
            SPLIT_MASK=bool(split_mask), BT_PIPE=int(bt_pipe),
            **_launch_kwargs(num_warps, num_stages, extra),
        )
    else:
        _il_decode_kernel[(num_seqs, nkv // hpc, nseg)](
            part, pmax, psum, q, k, v, block_table, seqused_k, cu_seqlens_q,
            float(softmax_scale) * LOG2E, q.stride(0), q.stride(1),
            block_table.stride(0), k.stride(0), k.stride(1), k.stride(2),
            NUM_Q_HEADS=nq, QPK=qpk, HPC=hpc, BLOCK_M=bm, HEAD=hd,
            BLOCK_SIZE=k.shape[1], TILE=tile, NSEG=nseg,
            PREFETCH=bool(prefetch), K_TRANS=bool(k_trans),
            **_launch_kwargs(num_warps, num_stages, extra),
        )
    if do_reduce:
        _il_reduce_kernel[(num_seqs, nq)](
            out, part, pmax, psum, seqused_k, cu_seqlens_q,
            out.stride(0), out.stride(1), num_tokens,
            NUM_Q_HEADS=nq, HEAD=hd, TILE=tile, NSEG=nseg,
            ZERO_PAD=bool(zero_pad),
            num_warps=int(reduce_warps), num_stages=1,
        )


# ===========================================================================
# wrapper entry point
# ===========================================================================
def supported(q, k, v, out, block_table) -> bool:
    """Layout preconditions of the kernels above (checked on the host)."""
    if q.dim() != 3 or k.dim() != 4 or out.dim() != 3:
        return False
    nq, hd = q.shape[1], q.shape[2]
    nkv = k.shape[2]
    bs = k.shape[1]
    return (
        q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype and v.dtype == q.dtype
        and nkv > 0 and nq % nkv == 0
        and hd >= 16 and (hd & (hd - 1)) == 0
        and bs > 0 and (bs & (bs - 1)) == 0
        and tuple(k.shape) == tuple(v.shape)
        and k.stride() == v.stride()
        and k.stride(3) == 1 and q.stride(2) == 1 and out.stride(2) == 1
        and block_table.dtype == torch.int32
    )


def pick_nseg(num_seqs: int, kv_groups: int, cap: int, dc: dict) -> int:
    """Power-of-two segment count for ``num_seqs`` decode rows that fits the
    scratch capacity ``cap`` (in (seq, segment) pairs per q head); 0 = none."""
    if num_seqs <= 0 or cap < num_seqs:
        return 0
    ex = dc.get("nseg_exact", 0)
    if ex:
        return max(1, min(int(ex), cap // num_seqs))
    nseg = max(1, _np2(dc.get("nseg", 16)))
    nmax = max(nseg, _np2(dc.get("nseg_max", nseg)))
    min_ctas = int(dc.get("min_ctas", 0))
    while (nseg < nmax and num_seqs * kv_groups * nseg < min_ctas
           and num_seqs * nseg * 2 <= cap):
        nseg *= 2
    while nseg > 1 and num_seqs * nseg > cap:
        nseg //= 2
    return nseg


def forward(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, block_table,
            softmax_scale, seq_threshold_3D=None, softmax_segm_output=None,
            softmax_segm_max=None, softmax_segm_expsum=None,
            cfg=None) -> bool:
    """Run the E4 kernels; returns False (nothing launched) when the caller
    must fall back to the E1/E2/E3 kernel.  Causal attention only."""
    cfg = CONFIG if cfg is None else cfg
    if not cfg.get("enable", True) or not supported(q, k, v, out, block_table):
        return False
    nq, hd = q.shape[1], q.shape[2]
    nkv = k.shape[2]
    num_seqs = seqused_k.shape[0]
    dc = cfg["decode"]
    hpc = dc["hpc"] if nkv % dc["hpc"] == 0 else 1

    have_buf = (softmax_segm_output is not None and softmax_segm_max is not None
                and softmax_segm_expsum is not None
                and softmax_segm_output.dtype == torch.float32
                and softmax_segm_max.dtype == torch.float32
                and softmax_segm_expsum.dtype == torch.float32
                and softmax_segm_output.is_contiguous()
                and softmax_segm_max.is_contiguous()
                and softmax_segm_expsum.is_contiguous())
    cap = 0
    if have_buf:
        cap = min(softmax_segm_max.numel(), softmax_segm_expsum.numel(),
                  softmax_segm_output.numel() // hd) // nq

    def _decode(nseg, zero_pad):
        launch_decode(
            q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
            softmax_segm_output, softmax_segm_max, softmax_segm_expsum,
            hpc=hpc, block_m=dc.get("block_m", 0), tile=dc["tile"], nseg=nseg,
            num_warps=dc["num_warps"], num_stages=dc["num_stages"],
            prefetch=dc.get("prefetch", False), k_trans=dc.get("k_trans", False),
            reduce_warps=dc.get("reduce_warps", 1), extra=dc.get("extra"),
            zero_pad=zero_pad, one_page=dc.get("one_page", False),
            split_mask=dc.get("split_mask", True),
            bt_pipe=dc.get("bt_pipe", 0))

    if max_seqlen_q <= 1:
        # pure decode: same eligibility as upstream's 3D path
        if not have_buf or seq_threshold_3D is None \
                or num_seqs > seq_threshold_3D:
            return False
        nseg = pick_nseg(num_seqs, nkv // hpc, cap, dc)
        if nseg < 1:
            return False
        _decode(nseg, True)
        return True

    pc = cfg["prefill"]
    if (pc["block_m"] % (nq // nkv)) != 0:
        return False
    nseg = 0
    if cfg.get("split_mixed", False) and have_buf:
        nseg = pick_nseg(num_seqs, nkv // hpc, cap, dc)
    split = nseg >= 1
    launch_prefill(
        q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
        block_m=pc["block_m"], tile=pc["tile"], num_warps=pc["num_warps"],
        num_stages=pc["num_stages"], split_mask=pc.get("split_mask", True),
        order=pc.get("order", 0), k_trans=pc.get("k_trans", False),
        prefetch=pc.get("prefetch", False), skip_decode=split,
        extra=pc.get("extra"), qmma=pc.get("qmma", False),
        tile2=pc.get("tile2", 0))
    if split:
        _decode(nseg, False)
    return True

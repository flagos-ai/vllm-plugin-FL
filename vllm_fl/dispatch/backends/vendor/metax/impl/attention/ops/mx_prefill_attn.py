# SPDX-License-Identifier: Apache-2.0
# 2026 - FlagOS S2 contest work on top of the MetaX Maca attention backend.
"""Calling-side attention policy for the vendor flash-attention on MetaX.

Used by ../flash_attn.py (MacaFlashAttention).  The attention math stays the
vendor's flash_attn 2.6.3+metax kernels; this module only changes *how* they
are called, plus one Triton page-gather copy kernel.

Measured on the C500 (work/R4/T4-mx-prefill-attn/bench_t4.py, policy P1,
logs/gpu/t4_P1.json), us per layer against the calls minicpm-opt makes
(flash_attn_varlen_func / flash_attn_with_kvcache + ``output[...] = ...``),
every variant checked against an fp64 reference (row err <= 2.4e-3):

* ``out=`` straight into the layer output (C entry points, no autograd
  Function): the slice assignment is a DtoD memcpy of 36-65 us even for a
  256 KB decode output.  Prefill -45 us/layer, decode -31..48 us/layer, also
  in FULL-graph replay (dec4k_64 276.2 -> 244.7, dec16k_64 826.1 -> 797.2).
* one prefill row with k_len >= 4096 (a chunk with context): gather its paged
  K/V into a contiguous buffer (8 KiB per page) and run the non-paged varlen
  kernel (kernel<128x64w4>, 101-107 TF/s vs 98 TF/s for the paged split-KV
  kernel): 16k_c7 2495 -> 2309 us.  ctx-0 chunks (k < 4096) stay paged.
* 2+ prefill rows with max k_len >= 10241 (the step that ends one 16k prompt
  and starts the next): one fwd_kvcache per row with explicit KV splits over a
  block table cut to the pages in use; 16k_tail prefill 1276 -> 819 us.
* everything else: the paged varlen call with ``out=``.

Mixed-step attention (prefill + decode rows) per layer: 16k 1.07-1.12x
(tail step 1.39x), 4k 1.11-1.21x; 42 layers: 16k 84.3 -> 75.1 ms, 4k
26.9 -> 23.1 ms per mixed step.

Rejected on the box (do not resurrect without new data): own Triton paged
prefill kernel (0.46-0.71x), explicit KV splits for single chunks (0.50-0.88x),
narrowed decode block tables, explicit decode splits, ks_set_solution steering
(sticky, breaks the decode kernel), MHA_* env knobs.

Mixed-step overlap (R5/P1, section "mixed-step overlap" below): the decode rows of a
mixed step run on a side stream while the prefill kernel runs.  T4 rejected
this because it gained nothing on top of the gather path (t4_pol_P2: 75.5 vs
75.1 ms), although it hid 290-490 of ~600 us/layer when the prefill was the
paged kernel (t4_full2, section overlap, base_overlap_prefirst, 16k_c4..c7).  The difference is
where the side stream starts.  T4 forked right after the KV write, so the decode
kernel became runnable while the short gather kernel ran.  Its ~770 one-warp
CTAs (8 KiB smem each, about 7 per AP) all went resident at once and kept the
2-CTA/AP prefill kernel (32 KiB smem) off the APs until the decode was done,
so the two ran one after the other.  Here the side stream waits for an event
recorded immediately before the main prefill kernel.  The prefill CTAs take
the APs first, and the decode CTAs fill the ~160 of 208 slots that the
prefill's 1.23-wave tail leaves empty.  Same kernels, same arguments, so the
outputs are bitwise identical.

Measured on the C500 (work/R5/P1-mx-mixed-attn-overlap, bench_p1.py and
bench_p1x.py, logs/gpu/p1x_base36_*.json, 36 decode rows, us per layer):
16k chunks c3..c7 1585/1888/2202/2505/2819 -> 1434/1669/1922/2151/2389,
16k split tail 1301 -> 1128 with go before the first per-row launch (1365
with go before the last).  With the qkv/o_proj GEMMs around it the gain
shrinks by 30-50 us/layer, and c0..c2 (k <= ~6000) and every 4k case lose
(-11..-130 us/layer), hence OVL_MIN_K = 8000.  Policy total for a 16k mixed
step: -6.4 ms (layer bench) to -7.5 ms (attention only), hiding 32-38% of
the decode-row time; 24/46 decode rows: -5.9 / -5.4..-7.1 ms.  4k: off.

Nothing here touches the GPU at import time.  setup() runs once from the
metadata builder: it self-tests every call form against the vendor's own
Python API on tiny inputs (argument order, the GQA out= layout) and compiles
the gather kernel for the pointer alignments it will see; on any failure the
policy stays off and the backend keeps its original calls.
"""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl

# ┌------------------------  Metax Modification -------------------------┐
# R8/Q1 self-developed INT8 KV cache (DEFAULT OFF).  When it has latched on, the
# vendor prefill kernels cannot read the cache, so prefill_attention() below hands
# the whole step to mx_kv_int8, which dequantises inside the paged -> contiguous
# gather this module already does and then calls back into the same vendor C entry
# points (varlen_fwd / kvcache_fwd) with ordinary bf16 buffers.  mx_kv_int8 never
# imports this module, so there is no cycle.
from . import mx_kv_int8 as _kv8
# └------------------------- Metax Modification -------------------------┘

# --------------------------------------------------------------------------
# Policy P1 (bench_t4.py section `policy`, logs/gpu/t4_P1.json)
# --------------------------------------------------------------------------
GATHER_MIN_K = 4096     # single prefill row: gather + contiguous varlen from here
GATHER_MAX_K = 32768    # bound on the per-layer scratch (2 * k * H_kv * D * 2 B)
TAIL_MIN_K = 10241      # 2+ prefill rows: per-row KV-split launches from here
TAIL_MAX_ROWS = 4       # measured with 2 rows; more rows keep the varlen call
TAIL_SCALE = 0.5        # multiplies seq_splits() (best of x0.5 / x1 / x2)

# seq_splits(): enough CTAs for ~2 waves of the 2-CTA/AP hdim-128 kernel
_SPLIT_SLOTS = 208      # 104 APs x 2 CTAs
_SPLIT_BLOCK_M = 128
_SPLIT_MIN_KEYS = 1024
_SPLIT_CAP = 16

# Mixed-step overlap (R5/P1; bench_p1.py, bench_p1x.py).  The metadata builder
# decides it once per step (overlap_wanted); forward() then calls
# mixed_overlapped().
OVL_MIN_K = 8000        # one prefill row with k_len >= this (the gather path);
                        # 16k_c2 (k ~6000) and 4k_c1 (k ~4000) gain attention-only
                        # but lose 17-31 us/layer between the GEMMs, so 4k stays off
OVL_MULTI_MIN_K = 10241  # 2..TAIL_MAX_ROWS prefill rows with max k_len >= this
                         # (the per-row split path, tail_rows_split)
OVL_MIN_DECODES = 1     # at least this many decode rows
OVL_MODE = "dec_side"   # dec_side: decode rows on the side stream, prefill on main
                        # pre_side: prefill rows on the side stream, decode on main
OVL_GO_AT = "first"     # main: go event right before the main prefill kernel
                        #   (tail path: before the last per-row launch)
                        # first: as main, except the tail path fires before the
                        #   first per-row launch (16k tail 1128 vs 1365 us/layer)
                        # kv: before anything (T4's rejected variant, bench only)
OVL_PRIORITY = 0        # side stream priority (0 = default, <0 = higher)
OVL_DELAY_ITERS = 0     # >0: the stream that waits on go spins this many loop
                        # iterations (1 CTA) before its kernels
_OVL_RING = 8           # (kv, go, done) event triples, reused round-robin

# State set by setup(); read by flash_attn.py on every call.
ENABLED = False         # all call forms passed the self-test
GATHER_OK = False       # gather kernel compiled and matched
OVERLAP_OK = False      # side stream created and the overlapped step matched bitwise
STATUS = "not set up"
_SETUP_DONE = False

_FAC = None


def vendor():
    """flash_attn_2_cuda, the pybind module behind flash_attn_interface."""
    global _FAC
    if _FAC is None:
        import flash_attn_2_cuda as fac  # noqa: PLC0415

        _FAC = fac
    return _FAC


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------------------
# vendor C entry points (flash_attn 2.6.3+metax3.7.0.7 argument order)
# --------------------------------------------------------------------------
def varlen_fwd(q, k, v, cu_q, cu_k, max_q, max_k, scale, *, block_table=None,
               out=None, causal=True):
    """flash_attn_cuda.varlen_fwd, same 23 arguments as
    flash_attn_interface._flash_attn_varlen_forward, with out= filled in."""
    res = vendor().varlen_fwd(
        q, k, v, out, cu_q, cu_k, None, None, block_table, None,
        int(max_q), int(max_k), 0.0, float(scale), False, bool(causal), -1, -1,
        0.0, False, None, None, False)
    return res[0]


def kvcache_fwd(q4, k_cache, v_cache, cache_seqlens, block_table, scale, *,
                num_splits=0, out=None, causal=True):
    """flash_attn_cuda.fwd_kvcache, same 21 arguments as
    flash_attn_interface.flash_attn_with_kvcache, with out= filled in.

    q4: (batch, seqlen_q, H_q, D); out: same shape.  For seqlen_q == 1 with
    GQA the vendor packs the H_q/H_kv query heads of a kv head into the row
    dimension and checks out= against that swapped shape (batch, group, H_kv,
    D), so the output is passed as that strided view and the kernel writes
    straight into the caller's tensor."""
    if out is not None and q4.shape[1] == 1 and q4.shape[2] > k_cache.shape[-2]:
        b, _, hq, d = q4.shape
        hkv = k_cache.shape[-2]
        out = out.view(b, hkv, hq // hkv, d).transpose(1, 2)
    res = vendor().fwd_kvcache(
        q4, k_cache, v_cache, None, None, cache_seqlens, None, None, None, None,
        block_table, None, out, float(scale), bool(causal), -1, -1, 0.0, True,
        int(num_splits), None)
    return res[0]


# --------------------------------------------------------------------------
# paged -> contiguous K/V gather (whole pages, masked tail)
# --------------------------------------------------------------------------
@triton.jit
def _gather_kv_pages(k_cache, v_cache, k_out, v_out, block_table, seq_lens, cu_k,
                     bt_stride, stride_cb, stride_ct,
                     PAGE_: tl.constexpr, ROW: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    L = tl.load(seq_lens + s)
    if b * PAGE_ >= L:
        return
    blk = tl.load(block_table + s * bt_stride + b).to(tl.int64)
    dst0 = tl.load(cu_k + s).to(tl.int64) + b * PAGE_
    r = tl.arange(0, PAGE_)
    c = tl.arange(0, ROW)
    m = (b * PAGE_ + r) < L
    src = blk * stride_cb + r[:, None] * stride_ct + c[None, :]
    dst = (dst0 + r)[:, None] * ROW + c[None, :]
    tl.store(k_out + dst, tl.load(k_cache + src, mask=m[:, None], other=0.0), mask=m[:, None])
    tl.store(v_out + dst, tl.load(v_cache + src, mask=m[:, None], other=0.0), mask=m[:, None])


def gather_layout_ok(k_cache, v_cache) -> bool:
    """NHD pages with packed rows and a power-of-two row (the kernel's tile)."""
    if k_cache.dim() != 4 or v_cache.stride() != k_cache.stride():
        return False
    _, page, hkv, d = k_cache.shape
    row = hkv * d
    return (k_cache.stride(3) == 1 and k_cache.stride(2) == d
            and k_cache.stride(1) == row
            and row & (row - 1) == 0 and page & (page - 1) == 0)


def gather_kv(k_cache, v_cache, block_table, seq_lens, cu_k, max_seq_len, total_k):
    """Copy the first seq_lens[s] tokens of every row s of block_table into
    contiguous (total_k, H_kv, D) buffers at offsets cu_k[s].  Sync-free:
    max_seq_len / total_k are host ints the metadata builder already has."""
    _, page, hkv, d = k_cache.shape
    k_out = torch.empty(total_k, hkv, d, dtype=k_cache.dtype, device=k_cache.device)
    v_out = torch.empty(total_k, hkv, d, dtype=v_cache.dtype, device=v_cache.device)
    grid = (triton.cdiv(int(max_seq_len), page), seq_lens.shape[0])
    _gather_kv_pages[grid](k_cache, v_cache, k_out, v_out, block_table, seq_lens, cu_k,
                           block_table.stride(0), k_cache.stride(0), k_cache.stride(1),
                           PAGE_=page, ROW=hkv * d, num_warps=4, num_stages=1)
    return k_out, v_out


# --------------------------------------------------------------------------
# host-side decisions (metadata builder, once per step)
# --------------------------------------------------------------------------
def host_lens(query_start_loc_cpu, seq_lens_cpu, num_decodes: int, num_reqs: int):
    """(query lens, kv lens) of the prefill rows as python ints, from the CPU
    tensors the builder already reads -- no device sync."""
    qsl = query_start_loc_cpu[num_decodes:num_reqs + 1]
    q = tuple((qsl[1:] - qsl[:-1]).tolist())
    k = tuple(seq_lens_cpu[num_decodes:num_reqs].tolist())
    return q, k


def tail_rows_split(host_k) -> bool:
    """True when the prefill rows go out as per-row KV-split launches (the
    builder then also cuts their block table to the pages in use)."""
    return 2 <= len(host_k) <= TAIL_MAX_ROWS and max(host_k) >= TAIL_MIN_K


def seq_splits(q_len: int, k_len: int, heads: int = 16) -> int:
    """KV splits for one prefill row issued through fwd_kvcache."""
    if k_len < 2 * _SPLIT_MIN_KEYS:
        return 1
    ctas = heads * _cdiv(q_len, _SPLIT_BLOCK_M)
    ns = _cdiv(2 * _SPLIT_SLOTS, ctas)
    return int(max(1, min(ns, k_len // _SPLIT_MIN_KEYS, _SPLIT_CAP)))


# --------------------------------------------------------------------------
# per-layer calls (FlashAttentionImpl.forward)
# --------------------------------------------------------------------------
def prefill_attention(q, key_cache, value_cache, out, block_table, cu_q, cu_k, seq_lens,
                      host_q, host_k, scale, bt_narrow=None, go=None, kv8=None):
    """Attention of the prefill rows, written into out in place.
    q/out: (n_prefill_tokens, H_q, D); block_table/seq_lens/cu_q/cu_k: device
    tensors of the prefill rows (cu_q rebased to 0); host_q/host_k: their
    lengths as python ints; bt_narrow: block table cut to the pages in use,
    set by the builder exactly when tail_rows_split(host_k).
    go: mixed-step overlap only (_Go, see mixed_overlapped); go.fire() is
    called once, on the stream the prefill runs on, right before the kernel
    the decode rows should run next to (after the gather copy).
    kv8: does THIS LAYER's cache hold int8 pages?  The caller knows (the impl's
    per-layer _kv8_ok and _kv8.ENABLED); None falls back to the global flag, which
    is what the bench and the self-test use."""
    nseq = len(host_q)
    max_q = max(host_q)
    max_k = max(host_k)
    if go is not None and go.at == "kv":
        go.fire()
    # ┌------------------------  Metax Modification -------------------------┐
    # R8/Q1: an INT8 cache has no vendor-readable paged branch, so EVERY
    # composition goes through our gather.  mx_kv_int8.prefill_plan() covers all
    # of them (a single row of any length, the ctx-0 chunks that are paged on the
    # bf16 path, and the 2+ row prompt-ending step, which keeps the T4 per-row
    # KV-split launches by viewing the gathered buffer as pages) and raises rather
    # than falling through to a paged reader.  bt_narrow is not used: the plan
    # derives its own page view.  go.fire() is idempotent, so the "kv" mode above
    # and the post-gather fire inside cannot double-record.
    # R9/F: keyed on the CALLER's per-layer answer, not on the global flag.  A layer
    # that shares another layer's KV cache keeps bf16 pages even when the feature is on
    # (the owner decides the format), and sending its rows to the int8 gather would read
    # bf16 bytes as int8.  _kv8.setup() now also refuses the feature outright on a model
    # with KV sharing, so this is the second of two locks on the same door.
    if (_kv8.ENABLED if kv8 is None else kv8):
        return _kv8.prefill_attention(
            sys.modules[__name__], q,
            _kv8.view_of_halves(key_cache, value_cache), out,
            block_table, cu_q, cu_k, seq_lens, host_q, host_k, scale, go=go,
        )
    # └------------------------- Metax Modification -------------------------┘
    if nseq == 1:
        if GATHER_OK and GATHER_MIN_K <= max_k <= GATHER_MAX_K:
            kc, vc = gather_kv(key_cache, value_cache, block_table, seq_lens, cu_k,
                               max_k, max_k)
            if go is not None:
                go.fire()
            varlen_fwd(q, kc, vc, cu_q, cu_k, max_q, max_k, scale, out=out)
            return out
    elif bt_narrow is not None:
        # one fwd_kvcache per row: the row that ends a prompt (few queries,
        # 16k keys) is split along KV instead of running 3 m-blocks x 16 heads
        # over 16k keys on 48 of 104 APs.
        hq, d = q.shape[1], q.shape[2]
        fire_at = 0 if go is not None and go.at == "first" else nseq - 1
        a = 0
        for s in range(nseq):
            ql, kl = host_q[s], host_k[s]
            b = a + ql
            ns = max(1, int(round(seq_splits(ql, kl, hq) * TAIL_SCALE)))
            if go is not None and s == fire_at:
                go.fire()
            kvcache_fwd(q[a:b].view(1, ql, hq, d), key_cache, value_cache,
                        seq_lens[s:s + 1], bt_narrow[s:s + 1], scale, num_splits=ns,
                        out=out[a:b].view(1, ql, hq, d))
            a = b
        return out
    if go is not None:
        go.fire()
    varlen_fwd(q, key_cache, value_cache, cu_q, cu_k, max_q, max_k, scale,
               block_table=block_table, out=out)
    return out


def decode_attention(q_dec, key_cache, value_cache, out, block_table, seq_lens, scale):
    """Decode rows, identical vendor call to flash_attn_with_kvcache (vendor
    split heuristic, full-width block table, CUDA-graph safe) but written
    straight into out.  q_dec: (n, q_len, H_q, D); out: (n * q_len, H_q, D)."""
    kvcache_fwd(q_dec, key_cache, value_cache, seq_lens, block_table, scale,
                num_splits=0, out=out.view(q_dec.shape))
    return out


# --------------------------------------------------------------------------
# mixed-step overlap (R5/P1): decode rows next to the prefill kernel
# --------------------------------------------------------------------------
# Stream protocol for one layer of a mixed step (OVL_MODE "dec_side"):
#
#   main : KV write .. [gather] .. go.record | prefill kernel ......| wait(done) -> o_proj
#   side :                     wait(go) ---> | decode + combine | done.record
#
# The decode kernel is queued after the prefill kernel on the host, and the
# side stream only waits for `go`, so both kernels become runnable together.
# The prefill CTAs get the APs first and the decode CTAs fill the slots
# the prefill tail wave leaves free.  No record_stream() is needed:
#   * every tensor the side stream reads (q, KV cache, block table, seq lens)
#     or writes (output rows) was allocated on main, and main waits for
#     `done` before it queues anything else, so any reuse of those blocks
#     happens after the side stream's work;
#   * the vendor's split-KV scratch is allocated while the side stream is
#     current, so the caching allocator only hands it out again to later
#     side-stream work, which is queued after it.
# A ring of event triples is reused round-robin, and `main` joins every
# layer, so no event is re-recorded before the wait that uses it has been
# queued.  Prefill rows never run under CUDA-graph capture (>512-token steps
# have no graph, PIECEWISE graphs run attention eagerly), and the caller
# checks is_current_stream_capturing() as well.


@triton.jit
def _spin_kernel(x_ptr, n):
    acc = tl.load(x_ptr)
    for _ in range(n):
        acc = acc * 0.9990234375 + 0.0009765625
    tl.store(x_ptr, acc)


class _OvlState:
    """Side stream + event ring, created once by setup() (or by a bench)."""

    def __init__(self, device, priority=0, ring=_OVL_RING):
        self.device = device
        self.priority = int(priority)
        self.stream = torch.cuda.Stream(device=device, priority=self.priority)
        self.events = [(torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event())
                       for _ in range(ring)]
        self.i = 0
        self.spin_main = torch.zeros(1, dtype=torch.float32, device=device)
        self.spin_side = torch.zeros(1, dtype=torch.float32, device=device)

    def next(self):
        e = self.events[self.i]
        self.i = (self.i + 1) % len(self.events)
        return e


class _Go:
    """Handed to prefill_attention(go=...): fire() records the go event on the
    current stream (once) right before the prefill kernel the decode rows
    should run next to; `then` (bench: host-order variant) runs right after."""

    __slots__ = ("ev", "at", "fired", "then")

    def __init__(self, ev, at, then=None):
        self.ev, self.at, self.fired, self.then = ev, at, False, then

    def fire(self):
        if self.fired:
            return
        self.fired = True
        self.ev.record()
        if self.then is not None:
            self.then()


_OVL: _OvlState | None = None


def _spin(buf, iters):
    _spin_kernel[(1,)](buf, int(iters), num_warps=1, num_stages=1)


def overlap_wanted(host_q, host_k, num_decodes: int) -> bool:
    """Metadata builder, once per step: run this step's decode rows next to
    its prefill rows?  Only the shapes bench_p1.py measured to gain."""
    if not (OVERLAP_OK and host_k) or num_decodes < OVL_MIN_DECODES:
        return False
    if len(host_k) == 1:
        return host_k[0] >= OVL_MIN_K
    # 2+ rows: only the per-row split tail was measured (the varlen call that
    # 5+ rows keep was not).
    return tail_rows_split(host_k) and max(host_k) >= OVL_MULTI_MIN_K


def mixed_overlapped(prefill, decode, *, state=None, mode=None, at=None, delay=None,
                     early=False):
    """One layer of a mixed step with the decode rows overlapped.

    prefill(go): queues the prefill rows (prefill_attention(..., go=go));
    decode():    queues the decode rows (decode_attention(...)).
    Returns once everything is queued; the current stream then waits for both.
    Keyword overrides (state/mode/at/delay/early) exist for bench_p1.py; the
    plugin uses the module defaults (OVL_*)."""
    st = _OVL if state is None else state
    mode = OVL_MODE if mode is None else mode
    at = OVL_GO_AT if at is None else at
    delay = OVL_DELAY_ITERS if delay is None else delay
    main = torch.cuda.current_stream()
    side = st.stream
    ev_kv, ev_go, ev_done = st.next()
    if mode == "pre_side":
        # prefill rows on the (higher-priority) side stream, decode on main
        ev_kv.record(main)
        side.wait_event(ev_kv)
        go = _Go(ev_go, at)
        with torch.cuda.stream(side):
            prefill(go)
            go.fire()
        ev_done.record(side)
        main.wait_event(ev_go)
        if delay:
            _spin(st.spin_main, delay)
        decode()
        main.wait_event(ev_done)
        return

    def launch_decode():
        side.wait_event(ev_go)
        with torch.cuda.stream(side):
            if delay:
                _spin(st.spin_side, delay)
            decode()
        ev_done.record(side)

    go = _Go(ev_go, at, launch_decode if early else None)
    prefill(go)
    go.fire()  # no-op if prefill_attention already fired it
    if not early:
        launch_decode()
    main.wait_event(ev_done)


# --------------------------------------------------------------------------
# start-up self-test
# --------------------------------------------------------------------------
def _close(got, ref, tol):
    g = got.detach().float().cpu()
    r = ref.detach().float().cpu()
    if not torch.isfinite(g).all():
        return False, float("nan")
    err = float((g - r).abs().max())
    return err <= tol, err


def _self_test(device, hq, hkv, d, page, dtype, bt_width, ref_varlen, ref_kvcache):
    gen = torch.Generator().manual_seed(0)

    def rnd(*shape):
        # .contiguous(): on MACA torch a host->device .to() of a 4-D tensor
        # comes back channels_last (e.g. (8, 16, 2, 128) -> strides
        # (4096, 1, 2048, 16)), which the C entry points reject ("Input tensor
        # must have contiguous last dimension").  The served KV cache and q
        # are allocated on the device and are not affected.
        return torch.randn(*shape, generator=gen).to(dtype).to(device).contiguous()

    nb = 8
    kc, vc = rnd(nb, page, hkv, d), rnd(nb, page, hkv, d)
    width = max(int(bt_width), 4)
    bt_cpu = torch.zeros(2, width, dtype=torch.int32)
    bt_cpu[0, :4] = torch.tensor([3, 1, 6, 0], dtype=torch.int32)
    bt_cpu[1, :4] = torch.tensor([5, 2, 7, 4], dtype=torch.int32)
    bt = bt_cpu.to(device)
    qlens, klens = (7, 20), (3 * page - 5, 2 * page + 3)
    seq = torch.tensor(klens, dtype=torch.int32, device=device)
    cu_q = torch.tensor([0, qlens[0], sum(qlens)], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, klens[0], sum(klens)], dtype=torch.int32, device=device)
    q = rnd(sum(qlens), hq, d)
    scale = d ** -0.5
    tol = 3e-2
    report = []

    def check(name, got, ref):
        ok, err = _close(got, ref, tol)
        report.append(f"{name}={err:.1e}")
        if not ok:
            raise RuntimeError(f"self-test {name}: max abs err {err} > {tol}")

    ref = ref_varlen(q=q, k=kc, v=vc, block_table=bt, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                     max_seqlen_q=max(qlens), max_seqlen_k=max(klens), softmax_scale=scale,
                     causal=True, window_size=(-1, -1), alibi_slopes=None, softcap=0,
                     s_aux=None)
    # paged varlen, both rows
    out = torch.full_like(q, float("nan"))
    varlen_fwd(q, kc, vc, cu_q, cu_k, max(qlens), max(klens), scale, block_table=bt, out=out)
    check("varlen_out", out, ref)
    # per-row KV-split launches over a narrowed table
    out = torch.full_like(q, float("nan"))
    narrow = bt[:, :_cdiv(max(klens), page)].contiguous()
    a = 0
    for s in range(2):
        b = a + qlens[s]
        kvcache_fwd(q[a:b].view(1, qlens[s], hq, d), kc, vc, seq[s:s + 1], narrow[s:s + 1],
                    scale, num_splits=2, out=out[a:b].view(1, qlens[s], hq, d))
        a = b
    check("rows_split", out, ref)
    # decode rows, GQA out= view
    qd = rnd(2, 1, hq, d)
    ref_d = ref_kvcache(q=qd, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=seq,
                        softmax_scale=scale, causal=True, window_size=(-1, -1),
                        alibi_slopes=None, softcap=0, s_aux=None)
    out_d = torch.full((2, hq, d), float("nan"), dtype=dtype, device=device)
    decode_attention(qd, kc, vc, out_d, bt, seq, scale)
    check("decode_out", out_d.view(2, 1, hq, d), ref_d)
    # gather + contiguous varlen, one row at a time.  Row 1 reads seq_lens /
    # block_table at an offset, i.e. the pointer alignments a step with
    # num_decodes % 4 != 0 produces, so both Triton specialisations of the
    # gather kernel compile here and not in the middle of a served step.  A
    # failure only turns the gather off.
    if not gather_layout_ok(kc, vc):
        report.append("gather=layout")
        return False, report
    try:
        a = 0
        for s in range(2):
            b = a + qlens[s]
            cu1 = torch.tensor([0, klens[s]], dtype=torch.int32, device=device)
            cq1 = torch.tensor([0, qlens[s]], dtype=torch.int32, device=device)
            kg, vg = gather_kv(kc, vc, bt[s:s + 1], seq[s:s + 1], cu1, klens[s], klens[s])
            o1 = torch.full_like(q[a:b], float("nan"))
            varlen_fwd(q[a:b], kg, vg, cq1, cu1, qlens[s], klens[s], scale, out=o1)
            check(f"gather_row{s}", o1, ref[a:b])
            a = b
    except Exception as exc:  # noqa: BLE001
        report.append(f"gather=FAILED({type(exc).__name__}: {str(exc)[:120]})")
        return False, report
    return True, report


def _self_test_overlap(device, hq, hkv, d, page, dtype, bt_width):
    """Creates the side stream, then runs one tiny mixed step (2 decode rows +
    1 prefill row) overlapped and in series on one stream: the outputs must be
    bitwise equal (same kernels, same arguments, only the streams differ)."""
    global _OVL
    gen = torch.Generator().manual_seed(1)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen).to(dtype).to(device).contiguous()

    nb = 8
    kc, vc = rnd(nb, page, hkv, d), rnd(nb, page, hkv, d)
    width = max(int(bt_width), 4)
    bt_cpu = torch.zeros(3, width, dtype=torch.int32)
    bt_cpu[0, :3] = torch.tensor([3, 1, 6], dtype=torch.int32)
    bt_cpu[1, :2] = torch.tensor([5, 2], dtype=torch.int32)
    bt_cpu[2, :4] = torch.tensor([7, 4, 0, 2], dtype=torch.int32)
    bt = bt_cpu.to(device)
    pre_q, pre_k = 20, 3 * page + 3
    i32 = dict(dtype=torch.int32, device=device)
    seq_dec = torch.tensor([3 * page - 5, 2 * page - 1], **i32)
    seq_pre = torch.tensor([pre_k], **i32)
    cu_q = torch.tensor([0, pre_q], **i32)
    cu_k = torch.tensor([0, pre_k], **i32)
    q = rnd(2 + pre_q, hq, d)
    scale = d ** -0.5

    def run(overlap):
        out = torch.full_like(q, float("nan"))

        def pre(go=None):
            prefill_attention(q[2:], kc, vc, out[2:], bt[2:3], cu_q, cu_k, seq_pre,
                              (pre_q,), (pre_k,), scale, None, go)

        def dec():
            decode_attention(q[:2].view(2, 1, hq, d), kc, vc, out[:2], bt[:2], seq_dec,
                             scale)

        if overlap:
            mixed_overlapped(pre, dec)
        else:
            pre()
            dec()
        return out

    _OVL = _OvlState(device, OVL_PRIORITY)
    ref = run(False)
    got = run(True)
    torch.cuda.synchronize(device)
    if not torch.isfinite(ref.float()).all():
        raise RuntimeError("overlap self-test: serial reference not finite")
    if not torch.equal(ref, got):
        err = float((ref.float() - got.float()).abs().max())
        raise RuntimeError(f"overlap self-test: overlapped != serial (max abs {err})")


def setup(device, num_heads_q, num_heads_kv, head_size, block_size, dtype, bt_width,
          ref_varlen, ref_kvcache) -> str:
    """Self-test once per process; sets ENABLED / GATHER_OK / OVERLAP_OK.
    Returns a status line for the caller's log."""
    global ENABLED, GATHER_OK, OVERLAP_OK, STATUS, _SETUP_DONE
    if _SETUP_DONE:
        return STATUS
    _SETUP_DONE = True
    ENABLED = GATHER_OK = OVERLAP_OK = False
    try:
        if dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"kv cache dtype {dtype}")
        vendor()
        gather_ok, report = _self_test(device, num_heads_q, num_heads_kv, head_size,
                                       block_size, dtype, bt_width, ref_varlen,
                                       ref_kvcache)
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001
        STATUS = f"off ({type(exc).__name__}: {str(exc)[:200]})"
        return STATUS
    ENABLED, GATHER_OK = True, gather_ok
    # A failure here only turns the mixed-step overlap off.
    try:
        _self_test_overlap(device, num_heads_q, num_heads_kv, head_size, block_size,
                           dtype, bt_width)
        OVERLAP_OK = True
        ovl = f"on ({OVL_MODE}, go at {OVL_GO_AT}, priority {OVL_PRIORITY})"
    except Exception as exc:  # noqa: BLE001
        ovl = f"off ({type(exc).__name__}: {str(exc)[:120]})"
    STATUS = (f"on (gather {'on' if gather_ok else 'off'}; self-test max err "
              f"{' '.join(report)}; mixed-step overlap {ovl})")
    return STATUS

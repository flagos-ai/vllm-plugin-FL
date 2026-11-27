# SPDX-License-Identifier: Apache-2.0
# 2026 - FlagOS S2 contest work on top of the MetaX Maca attention backend.
"""Self-developed INT8 KV cache for the MetaX C500 (R8/Q1, rewritten in R9).  DEFAULT OFF.

Target file in the plugin:
    vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/mx_kv_int8.py
(next to mx_prefill_attn.py, whose vendor C entry points it calls, and mx_dec_native.py,
whose native kernel reads this layout with its Q8 = 1 instantiations).

WHY
---
A 4k decode step reads 17.1 GB, of which 12.6 GB (73%) is bf16 KV; a 16k decode step
reads 51.1 GB, of which 46.6 GB (91%).  INT8 data plus one fp32 scale per (token, kv
head) for K and for V is 528 B per token per layer instead of 1024 B (x0.516), so the
decode attention -- 37.1 s of a 106 s 4k run and 80.1 s of a 185 s 16k run -- is the
largest remaining lever on both cases.

WHY THE TWO EARLIER ATTEMPTS LOST, AND WHAT IS DIFFERENT
--------------------------------------------------------
R4/T5 and R5/K4 read the INT8 KV with a TRITON kernel that reaches about 1.03 TB/s
against the vendor bf16 kernel's 1.24-1.39, so half the bytes with a 40% slower kernel
was a net loss (4k +1.8% / 16k -5.8%, and 0.69x of the vendor at kernel level).  This
round reads it with the native MACA C++ kernel of R6/N1 + R7/N4, which is at or above
vendor parity on bf16, through one more template parameter (Q8) -- see
csrc/q1_int8.cuh for the layout and the register dequantisation, and DESIGN.md for the
measured budget.  Accuracy was measured in R6/X3: top-1 agreement 99.2%, perplexity
x1.005.

WHAT R9 MEASURED (the numbers this module is now tuned to)
----------------------------------------------------------
R8/Q1 shipped this cache and projected only +1.6% average; the R8/Q1 audit attributed
87-92% of the missing gain to the implementation, and R9 rebuilt both halves:
  * the READER (csrc/q1_int8.cuh, R9/R) dequantises through an fp16 magic number
    instead of a bf16 one -- 134 ALU ops per lane per tile instead of 418 -- which is
    worth x1.555 / x1.782 of the vendor bf16 kernel at 64 rows over a 4k / 16k context,
    against cfg 28's x1.171 / x1.245, and its error FALLS to the quantisation floor;
  * the WRITER (R9/W + R9/G) launches one program per (token block, kv head) with int32
    addresses and a 3-D K tile: 4.87 us/layer at n = 64 against the R8 kernel's 7.74 and
    the vendor bf16 reshape_and_cache's 4.42, and 9.87 against 11.17 / 11.98 at n = 2048.
The whole-step measurement (write + 42 layers of attention in one graph) gives 7.15 ms
against the deployed 10.42 at 4k and 20.00 against 33.16 at 16k, which projects to
4k x2.886 / 16k x2.078 = +17.6% average score over the deployed r7_n6 stack
(/data/exp/R9/G/logs/r9_full_20261025_075543.json, 168/168 checks).  The audit's
headline claim that the writer cost 145.3 us/layer was a host-timer artefact: it costs
11.17.  THE FEATURE STILL SHIPS OFF -- see COMPLIANCE.

THE FOUR PIECES (all of them are needed; the interlock is the critical one)
--------------------------------------------------------------------------
(a) STORAGE  int8 data + fp32 scale per (token, kv head), carved out of the SAME bf16
    pages vLLM allocated (4224 of 8192 B per half page), so num_blocks, the KV budget
    and therefore the scheduler's behaviour are bit-for-bit unchanged.  At concurrency
    64 the freed space would buy nothing: the client, not the cache, caps the batch
    (16k needs 64 x 17408 = 1.11 M tokens against the ~1.17 M the pool already holds).
(b) WRITE    our own quantising KV write: this module's Triton kernel
    (write_kv_int8, a drop-in for reshape_and_cache_flash) and, when the fused glue
    path is on, the same arithmetic inside the FlagGems rope_kv_cache kernel and the
    native n6 rope kernel (impl/mx_glue.py, QUANT constexpr).
(c) DECODE READ  the native kernel's Q8 = 1 configs (mx_dec_native.decode_attention
    with view=...), covering the decode-only steps (FULL graphs) and the decode rows of
    mixed steps.
(d) PREFILL READ  the vendor prefill kernel CANNOT read int8 and must not be replaced
    (our own prefill FA measured 73-77 TF/s against the vendor's 100-111, twice), so the
    dequantisation is folded into the paged -> contiguous GATHER the T4 path already
    does, and the policy guarantees that EVERY prefill reader goes through that gather:
    prefill_plan() covers every (host_q, host_k) composition, including the ctx-0 chunks
    that are paged today (k < 4096) and the 2+ row prompt-ending steps, which keep their
    per-row KV-split launches by viewing the gathered buffer as pages (fake_block_table).

NOTHING MAY SEE THE INT8 PAGES EXCEPT OUR OWN KERNELS.  enable() is refused unless
every reader is covered: cascade attention off (the platform disables it anyway),
dcp_world_size == 1, kv_sharing_target_layer_name None, cache_dtype "auto" (bf16
allocation), decode rows with q_len == 1, and the native library self-test green.  The
decision is latched once per process BEFORE the first KV write (the metadata builder's
setup, which runs before the first forward and before CUDA-graph capture), so a cache
can never hold a mix of formats.

COMPLIANCE
----------
Self-developed quantisation strategy (RULE:89) inside vllm-plugin-FL, reached through
the backend's own op-selection path, with the bf16 vendor path kept intact and used
whenever the feature is off or any gate fails.  It is NOT enabled by a hyper-parameter:
--kv-cache-dtype / cache_config.cache_dtype stay "auto" and are never read; the storage
format is decided by this module alone.  POLICY_DEFAULT stays "0" (feature off) until
the organisers answer question 6 of notes/organiser_questions.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
QMAX = 127.0

# Master switch.  POLICY_DEFAULT is what a plain server does; the env var overrides it.
ENV_FLAG = "VLLM_FL_MX_INT8_KV"
POLICY_DEFAULT = "0"
MODE = os.environ.get(ENV_FLAG, POLICY_DEFAULT).strip().lower()
REQUESTED = MODE in ("1", "on", "true", "yes")

# ---------------------------------------------------------------------------
# layout "Q1-L1" (mirrors csrc/q1_int8.cuh; one vLLM page = 16 tokens = one MMA tile)
# ---------------------------------------------------------------------------
#   half page (the K or the V part of one block), bytes:
#     [0, HKV*PAGE*D)                int8 data, one PAGE*D block per kv head
#     [sc, sc + 4*HKV*PAGE)          fp32 scale[h][t]
#   inside a head's block:
#     V (token major)     t*D + d
#     K (dim-group major) (d // 8)*(PAGE*8) + t*8 + (d % 8)
# K is swizzled so that the 8 bytes a lane needs as an MMA x fragment are contiguous and
# a warp instruction covers 512 contiguous bytes; that removes the identity-MMA transpose
# the bf16 path needs (16 MMAs per tile instead of 24) -- an int8 tile is half the bytes
# and therefore twice as compute-dense per byte.
PAGE_TILE = 16          # tokens per page the layout is defined for (vLLM block_size)
DIMG = 8                # bytes per K dim-group


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class CacheView:
    """Where the int8 data and the fp32 scales of one layer live (element units).

    data   : int8 view of the engine's bf16 allocation
    scales : fp32 view of the same storage
    Element (block b, kv head h, token t in the page, dim d):
        data[k_base + b*s_blk + h*s_head + K_OFF(t, d)]        (K)
        data[v_base + b*s_blk + h*s_head + t*d_dim + d]        (V)
    Scale:  scales[sk_base + b*ss_blk + h*page + t]            (sv_base for V)
    """

    data: torch.Tensor
    scales: torch.Tensor
    k_base: int
    v_base: int
    s_blk: int            # bytes per half page
    s_head: int           # bytes per (page, kv head) data block
    sk_base: int          # fp32 index of the K scale block
    sv_base: int
    ss_blk: int           # fp32 stride between pages
    page: int
    hkv: int
    d: int
    num_blocks: int
    sc_off: int           # byte offset of the scale block inside a half page

    def args(self):
        return (self.k_base, self.v_base, self.s_blk, self.s_head,
                self.sk_base, self.sv_base, self.ss_blk)

    def native_args(self):
        """(q8, q8_sc) for mx_dec_native.launch(); kc / vc stay the bf16 tensors (the
        kernel reinterprets them), so the strides it is given are the bf16 ones."""
        return 1, self.sc_off

    def bytes_per_token_layer(self) -> int:
        return 2 * self.hkv * (self.d + 4)


def int8_inplace_fits(kv: torch.Tensor) -> bool:
    """kv = (2, NB, PAGE, HKV, D) bf16, dense inside a page (what the backend allocates),
    page large enough for int8 data + fp32 scales, page == one tile."""
    if kv.dim() != 5 or kv.shape[0] != 2 or kv.element_size() != 2:
        return False
    _, nb, page, hkv, d = kv.shape
    if page != PAGE_TILE or d % DIMG or not kv.is_contiguous():
        return False
    need = hkv * page * d + 4 * hkv * page
    return 2 * kv.stride(1) >= need and nb >= 1


def int8_view(kv: torch.Tensor) -> CacheView:
    assert int8_inplace_fits(kv), f"kv cache {tuple(kv.shape)} / {kv.stride()} unsupported"
    _, nb, page, hkv, d = kv.shape
    # element-indexed FLAT views (the CacheView contract): kv.view(dtype) keeps the 5-D
    # shape, which would make data[idx] index dim 0.  kv is contiguous (int8_inplace_fits
    # asserts it), so reshape(-1) is a view -- no copy, the writes stay in place.
    flat = kv.view(torch.int8).reshape(-1)
    f32 = kv.view(torch.float32).reshape(-1)
    half = 2 * kv.stride(1)          # bytes per half page
    vb = 2 * kv.stride(0)            # bytes from the K half to the V half
    sc = hkv * page * d              # scales right after the data
    return CacheView(flat, f32, 0, vb, half, page * d,
                     sc // 4, (vb + sc) // 4, half // 4,
                     page, hkv, d, nb, sc)


_VIEWS: dict = {}


def view_of(kv: torch.Tensor) -> CacheView:
    """Per-layer view, built once (the attention forward runs eagerly or at capture)."""
    key = (kv.data_ptr(), tuple(kv.shape))
    v = _VIEWS.get(key)
    if v is None:
        v = int8_view(kv)
        _VIEWS[key] = v
    return v


_KVS: dict = {}


def kv_of_halves(key_cache: torch.Tensor, value_cache: torch.Tensor) -> torch.Tensor:
    """The (2, NB, PAGE, HKV, D) tensor the two halves are ``unbind(0)`` views of.

    The attention impl carries the halves, not the allocation, but every layout
    expression here and in csrc/q1_int8.cuh is written against the 5-D tensor, and the
    native launch wants its two base pointers plus the bf16 page strides.  Rebuilding
    it from the K half (the V half starts exactly ``key_cache.numel()`` elements later,
    which is checked) keeps the call sites to one line and costs nothing: the result is
    cached per allocation, and the attention forward runs eagerly or at capture.
    """
    key = (key_cache.data_ptr(), tuple(key_cache.shape), value_cache.data_ptr())
    kv = _KVS.get(key)
    if kv is not None:
        return kv
    if (key_cache.dim() != 4 or key_cache.shape != value_cache.shape
            or key_cache.dtype != value_cache.dtype or key_cache.element_size() != 2
            or not key_cache.is_contiguous() or not value_cache.is_contiguous()):
        raise RuntimeError(f"INT8 KV: unsupported cache halves {tuple(key_cache.shape)}")
    gap = value_cache.data_ptr() - key_cache.data_ptr()
    if gap != key_cache.element_size() * key_cache.numel():
        # not the single (2, NB, PAGE, HKV, D) allocation the layout is carved out of
        raise RuntimeError(f"INT8 KV: K and V halves are {gap} B apart, expected "
                           f"{key_cache.element_size() * key_cache.numel()}")
    kv = key_cache.new_empty(0).set_(
        key_cache.untyped_storage(),
        key_cache.storage_offset(),
        (2,) + tuple(key_cache.shape),
        (key_cache.numel(),) + tuple(key_cache.stride()),
    )
    _KVS[key] = kv
    return kv


def view_of_halves(key_cache: torch.Tensor, value_cache: torch.Tensor) -> CacheView:
    """view_of() for the two halves the attention impl carries around."""
    return view_of(kv_of_halves(key_cache, value_cache))


def _maca_backend() -> bool:
    if INTERPRET:
        return False
    try:
        from triton.backends.metax import compiler as _mc  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


_IS_MACA = None


def launch_kwargs(warps, stages=1, pipeline=None):
    global _IS_MACA
    if INTERPRET:
        return {}
    kw = dict(num_warps=warps, num_stages=stages)
    if _IS_MACA is None:
        _IS_MACA = _maca_backend()
    if _IS_MACA and pipeline:
        kw["pipeline"] = pipeline
    return kw


# ===========================================================================
# (b) WRITE: fused K+V absmax-quantise-and-write (drop-in reshape_and_cache_flash)
# ===========================================================================
@triton.jit
def _q1_rne(y):
    """Round-to-nearest-EVEN of a float32 tile into int8, for |y| <= 127.

    R9/F: deliberately independent of what the backend's float -> int cast does.
    ``t`` is that cast and ``f = y - t`` is EXACT (t and y are within a factor of two
    for |y| >= 1, so Sterbenz applies; below 1 the cast gives 0 or +-1 and the
    subtraction is exact anyway), so the correction below lands on the RNE result
    whether the cast truncates toward zero OR already rounds to nearest even -- which
    is precisely the ambiguity the R8 write self-test could not settle.  The old form,
    ``tl.where(y >= 0, y + 0.5, y - 0.5).to(tl.int8)``, is correct only for a
    truncating cast and even then rounds half AWAY FROM ZERO, where the torch
    reference (torch.round) rounds half to even; it also adds 0.5 to the product,
    which lets the compiler contract the two into one FMA and skip the rounding of
    ``y`` itself.  Here ``y`` is rounded before anything is added to it.
    """
    t = y.to(tl.int32)
    f = y - t.to(tl.float32)
    a = tl.abs(f)
    up = (a > 0.5) | ((a == 0.5) & ((t & 1) != 0))
    s = tl.where(f >= 0, 1, -1)
    return (t + tl.where(up, s, 0)).to(tl.int8)


@triton.jit
def _q1_write_kv(K, V, SLOT, D8, D32, n_tok,
                 stride_kt, stride_kh, stride_vt, stride_vh,
                 k_base, v_base, s_blk, s_head, sk_base, sv_base, ss_blk,
                 HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                 BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK_T * HKV)
    tok = pid * BLOCK_T + r // HKV
    h = r % HKV
    d = tl.arange(0, D)
    dg = d // 8                      # K dim-group
    dr = d % 8
    tok_ok = tok < n_tok
    slot = tl.load(SLOT + tok, mask=tok_ok, other=-1).to(tl.int64)
    ok = slot >= 0
    slot = tl.where(ok, slot, 0)
    blk = slot // PAGE
    t = slot - blk * PAGE
    m2 = ok[:, None]
    tok64 = tok.to(tl.int64)
    k = tl.load(K + tok64[:, None] * stride_kt + h[:, None] * stride_kh + d[None, :],
                mask=m2, other=0.0).to(tl.float32)
    v = tl.load(V + tok64[:, None] * stride_vt + h[:, None] * stride_vh + d[None, :],
                mask=m2, other=0.0).to(tl.float32)
    ka = tl.max(tl.abs(k), axis=1)
    va = tl.max(tl.abs(v), axis=1)
    kinv = tl.where(ka > 0, 127.0 / ka, 0.0)
    vinv = tl.where(va > 0, 127.0 / va, 0.0)
    kq = tl.minimum(tl.maximum(k * kinv[:, None], -127.0), 127.0)
    vq = tl.minimum(tl.maximum(v * vinv[:, None], -127.0), 127.0)
    # round half to even, the same rule torch.round() uses (see _q1_rne)
    kq = _q1_rne(kq)
    vq = _q1_rne(vq)
    hbase = blk * s_blk + h * s_head
    kdst = k_base + hbase[:, None] + dg[None, :] * (PAGE * 8) + t[:, None] * 8 + dr[None, :]
    vdst = v_base + hbase[:, None] + t[:, None] * D + d[None, :]
    tl.store(D8 + kdst, kq, mask=m2)
    tl.store(D8 + vdst, vq, mask=m2)
    sdst = blk * ss_blk + h * PAGE + t
    # scale = absmax / 127 as a DIVISION, bit-identical to quantize_rows() in torch and
    # to the numpy layout reference (absmax * (1/127) differs by 1 ulp)
    tl.store(D32 + sk_base + sdst, ka / 127.0, mask=ok)
    tl.store(D32 + sv_base + sdst, va / 127.0, mask=ok)


WRITE_CFG = dict(block_t=8, warps=2, stages=1)


# ---------------------------------------------------------------------------
# R9/W: the rewritten quantising write
# ---------------------------------------------------------------------------
# WHY A SECOND KERNEL.  _q1_write_kv above measured 145.3 us/layer at n = 2048 against a
# 19.8 us roofline (it moves 3.18 MB where the vendor's bf16 reshape_and_cache moves
# 4.19 MB; R8/Q1-audit section 1).  Its host wrapper costs about 30 us -- that is what
# the same measurement reports at n = 64, where the GPU cannot be spending more than a
# few us on 99 KB -- so the 145.3 us is GPU time: 22 GB/s, 1.4% of peak.
#
# An OFFLINE AOT compile for this card (GPUTarget("maca", 80, 64), no device,
# work/R9/W-q1-writer/offline/aot_triton.py, with the runtime's divisible_by_16
# specialisation, which every address argument of this path satisfies) answers the one
# question the audit left open -- and NOT the way it guessed.  Triton DOES vectorise the
# deployed kernel: 8 B stores for K, 16 B for V, 32 B loads.  "Per-element scatter" is
# refuted (it is what the compiler reports only if the divisibility hints are left out,
# which is an artefact of compiling by hand).  What the deployed config pays instead,
# read off its ttgir:
#
#   * 2304 B of SHARED MEMORY and 3 convert_layout per program: the K store wants
#     sizePerThread 8 and the V store 16 in one elementwise chain, so the whole tile
#     makes a shared-memory round trip.  (The same kernel at 4 or 8 warps compiles to
#     64-128 B and 2 convert_layout -- the deployed block_t 8 / 2 warps is the worst
#     bucket of its own kernel.)
#   * 41 two-dimensional int64 index tensors: at 16 elements per lane that is 32 VGPRs
#     for kdst plus 32 for vdst, plus two int64 adds per element.
#   * 8 programs at n = 64 (7.7% of the 104 APs) and 256 at n = 2048, 2 warps each.
#   * K and V both live at once (load K, load V, quantise both, store both), so the
#     register peak is doubled.
#
# The rewrite below is not a tuning knob; it removes all four.  The K tile is 3-D,
# [BLOCK_T, D/DIMG, DIMG], so the innermost axis of BOTH the load and the store index is
# a plain ``tl.arange(0, DIMG)`` with stride 1: the store keeps its 8 B width and the
# load its 16 B with no ``d // 8`` / ``d % 8`` anywhere, and -- the point -- K's and V's
# tiles now agree on sizePerThread, so the shared-memory round trip disappears (64 B at
# block_t 8 / 2 warps, for the two absmax reductions alone).  K is written, then V, so
# only one tile is live.  Further changes, all of them from the audit's list:
#   * one program per (token block, kv head) -- grid (cdiv(n, BLOCK_T), HKV) -- so the
#     grid fills the 104 APs at n = 64 as well (128 programs instead of 8), and ``h`` is
#     a scalar instead of a per-row tensor (the ``r // HKV`` / ``r % HKV`` rows go away);
#   * every address int32 (the caller asserts the bound), so the [16, 128] int64 index
#     tensors (32 VGPRs each for kdst and vdst) become int32 ones;
#   * the absmax reduction of K is over the two INNER axes of the 3-D tile, i.e. inside a
#     thread (axis 2, DIMG values held in one register group) and then across the
#     DIMG-group axis, instead of across 8 lanes of a 128-wide row.
# The K STORAGE LAYOUT IS UNCHANGED, bit for bit: the reader's swizzle
# (d>>3)*128 + t*8 + (d&7) (csrc/q1_int8.cuh:20-30) and its 512 B per warp instruction
# are exactly what they were, so the measured 1.392/1.501 TB/s read floor is untouched.
# Writing K page-major instead (one program per page, 2048 contiguous B) would allow
# 16 B stores, but at n = 64 every token sits in its own page, so it would help only the
# prefill write, and the audit's byte count says the extra width is worth at most a few
# hundred ns: K int8 data is 0.5 MB of the 3.18 MB a layer moves at n = 2048.  The
# sibling Iluvatar card measured a pre-transposed K store at 0.153x of its vendor because
# a column store there had stride page_size; that failure mode does NOT exist here, and
# the check is in the offsets above: within one (page, kv head) block K is dim-group
# major, so the 16 tokens of a page write the SAME 128 B run (stride DIMG = 8 B inside
# the run), not one run each -- whenever a program covers a whole page, every line it
# touches is written in full even at 8 B per instruction.
@triton.jit
def _q1_write_kv2(K, V, SLOT, D8, D32, n_tok,
                  stride_kt, stride_kh, stride_vt, stride_vh,
                  k_base, v_base, s_blk, s_head, sk_base, sv_base, ss_blk,
                  HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                  BLOCK_T: tl.constexpr, DIMG: tl.constexpr, RNE: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.arange(0, BLOCK_T)
    tok = pid * BLOCK_T + r
    ok = tok < n_tok
    slot = tl.load(SLOT + tok, mask=ok, other=-1).to(tl.int32)
    ok = ok & (slot >= 0)
    slot = tl.where(ok, slot, 0)
    blk = slot // PAGE
    t = slot - blk * PAGE
    hb = blk * s_blk + h * s_head                       # int32 [BLOCK_T]
    dg = tl.arange(0, D // DIMG)
    dr = tl.arange(0, DIMG)
    d = tl.arange(0, D)
    # ---- K: [BLOCK_T, D/DIMG, DIMG]; innermost axis contiguous on both sides --------
    m3 = ok[:, None, None]
    ksrc = (tok[:, None, None] * stride_kt + h * stride_kh
            + dg[None, :, None] * DIMG + dr[None, None, :])
    k = tl.load(K + ksrc, mask=m3, other=0.0).to(tl.float32)
    # Per (token, head) absmax over the WHOLE 128-dim row, expressed on a 2-D view of
    # the tile.  R9/G measured that the nested form -- tl.max(tl.max(|k|, axis=2),
    # axis=1), i.e. reducing the innermost DIMG axis first -- returns a PARTIAL maximum
    # when BLOCK_T is 1 or 2 on this backend: the scale came out up to 0.93 relative too
    # small, so 69% of the K codes at n = 2048 were 1.26-1.29x too large and clamped at
    # 127 (diag_w.py / diag_w2.py, work/R9/G-gpu/logs/).  BLOCK_T >= 4 was exact, which
    # is why the offline program-by-program numpy check could not see it and why the bug
    # sat exactly on WRITE_PLAN's decode bucket.  Reducing a reshaped [BLOCK_T, D] tile
    # is exact at BLOCK_T 1, 2, 4, 8, 32 and never slower (it is 0.9 us/layer FASTER at
    # block_t 8 / 2 warps, n = 2048).
    ka = tl.max(tl.reshape(tl.abs(k), [BLOCK_T, D]), axis=1)
    kinv = tl.where(ka > 0, 127.0 / ka, 0.0)
    kc = tl.minimum(tl.maximum(k * kinv[:, None, None], -127.0), 127.0)
    kq = _q1_rne(kc) if RNE else kc.to(tl.int8)
    kdst = (hb[:, None, None] + dg[None, :, None] * (PAGE * DIMG)
            + t[:, None, None] * DIMG + dr[None, None, :])
    tl.store(D8 + k_base + kdst, kq, mask=m3)
    # ---- V: [BLOCK_T, D] token-major on both sides ----------------------------------
    m2 = ok[:, None]
    v = tl.load(V + tok[:, None] * stride_vt + h * stride_vh + d[None, :],
                mask=m2, other=0.0).to(tl.float32)
    va = tl.max(tl.abs(v), axis=1)
    vinv = tl.where(va > 0, 127.0 / va, 0.0)
    vc = tl.minimum(tl.maximum(v * vinv[:, None], -127.0), 127.0)
    vq = _q1_rne(vc) if RNE else vc.to(tl.int8)
    vdst = hb[:, None] + t[:, None] * D + d[None, :]
    tl.store(D8 + v_base + vdst, vq, mask=m2)
    sdst = blk * ss_blk + h * PAGE + t
    # scale = absmax / 127 as a DIVISION, bit-identical to quantize_rows() in torch and
    # to the numpy layout reference (absmax * (1/127) differs by 1 ulp)
    tl.store(D32 + sk_base + sdst, ka / 127.0, mask=ok)
    tl.store(D32 + sv_base + sdst, va / 127.0, mask=ok)


# (block_t, warps) by token count.  FEATURE-DRIVEN: the only input is n, the number of
# slots this call writes, which is a property of the step (1 per decode row, the chunk
# size on a prefill step).  Targets, for HKV = 2 on a 104-AP card:
#   n <=  128  -> block_t 1: 2*n programs (128 at n = 64), one warp each, so the grid
#                 covers the device in one wave and the cost is one memory round trip;
#   n <=  512  -> block_t 4: 2*cdiv(n,4) programs (256 at n = 512);
#   else       -> block_t 8: 2*cdiv(n,8) programs (512 at n = 2048) x 2 warps = 1024
#                 warps = 4.9 per AP, i.e. 4.92 CTA waves, so wave quantisation costs
#                 5/4.92 = 1.6% (block_t 16 would be 2.46 waves = 22%).
# R9/G, measured on the C500 under graph replay (us per layer, 42 calls in one graph,
# /data/exp/R9/G/logs/r9_full_20261025_075543.json; "v1" is the R8 kernel above, "v2"
# this one, "floor" a pure store of the same bytes):
#
#   n     v2 bt1/w1  bt2/w1  bt4/w1  bt8/w1  bt8/w2 | v1 bt8/w2  bt8/w4 | vendor  floor
#     16       4.84    5.44    6.62    9.08    9.37 |      7.94    5.97 |   4.15   2.97
#     64       4.87    5.74    6.61    9.27    9.35 |      7.74    5.97 |   4.42   2.97
#    512       6.57    6.38    6.60    9.07    9.38 |      7.94    5.92 |   6.02   3.69
#   2048      13.51   11.11   10.69   12.44   13.83 |     11.17    9.87 |  11.98   4.76
#
# Two facts, both of them measured rather than argued:
#   * v2 owns the SMALL counts, because the grid is what matters there -- at n = 64 it
#     launches 128 programs (100% of the 104 APs, 1.23 waves) where v1 launches 8
#     (7.7%), and 4.87 us against 7.74 is the whole of it;
#   * v1 at FOUR warps owns the large ones (5.92 / 9.87 against v2's best 6.38 / 10.69,
#     i.e. 7.7% at n = 2048), because there the device is full either way and v1's one
#     2-D [BLOCK_T*HKV, D] tile issues both stores from one elementwise chain.
# So the shipped plan takes each where it wins.  One warp per program is right for v2 at
# every size: the wave-quantisation argument that put (8, 2) in this table was right
# about the CTA count and wrong about the warp -- at 2 warps each program carries twice
# the live tile and the extra width buys nothing (bt8/w2 is the worst v2 bucket at every
# n).  The split point is the step's own token count and nothing else; n <= 128 is every
# decode step at this concurrency and n = 2048 every prefill chunk.
#
# The two kernels are BIT-IDENTICAL (offline/test_w_write.py --long executes both over
# numpy at BLOCK_T 1..32 for ten shapes; the start-up self-test writes the same input
# with both on the device and compares the whole cache), so this plan changes no stored
# byte -- it is a launch policy, not a format.
WRITE_PLAN = ((128, 1, 1, "v2"), (1 << 62, 8, 4, "v1"))
# ... and each kernel's own measured best, for a forced VLLM_FL_MX_KV8_WRITER and for
# the int32 fallback.  WRITE_CFG (block_t 8 / 2 warps) stays what it was: the config R8
# deployed, kept as the A/B reference the bench asks for explicitly.
WRITE_PLAN_V2 = ((128, 1, 1), (1 << 62, 4, 1))
WRITE_PLAN_V1 = ((1 << 62, 8, 4),)
# "plan" = the hybrid above (the default); "v1" / "v2" force one kernel for every n.
WRITE_KERNEL = os.environ.get("VLLM_FL_MX_KV8_WRITER", "plan").strip().lower()
INT32_MAX = (1 << 31) - 1


def write_cfg_for(n: int, kernel: str | None = None) -> dict:
    """Launch config for writing n slots: (block_t, warps, kernel).

    kernel None  -> the shipped hybrid plan;
    "v1" / "v2"  -> that kernel's own measured best for n (a forced writer must not be
                    handed the other one's block_t, which is what the hybrid table holds).
    """
    table = {None: WRITE_PLAN, "plan": WRITE_PLAN,
             "v1": WRITE_PLAN_V1, "v2": WRITE_PLAN_V2}.get(kernel, WRITE_PLAN)
    for row in table:
        hi, bt, w = row[0], row[1], row[2]
        if n <= hi:
            return dict(block_t=bt, warps=w, stages=1,
                        kernel=(row[3] if len(row) > 3 else kernel or "v2"))
    raise AssertionError("unreachable")


def write_int32_safe(view: CacheView, n: int) -> bool:
    """Every address _q1_write_kv2 computes fits in int32.

    Largest int8 element index: the last byte of the V half of the last page,
    ``v_base + (num_blocks-1)*s_blk + s_head*(hkv-1) + page*d``, bounded by
    ``2*num_blocks*s_blk`` = the whole allocation in bytes.  Largest fp32 index:
    ``sv_base + num_blocks*ss_blk`` = that / 4.  Largest source index:
    ``n*stride_kt`` <= n*hkv*d.  For MiniCPM5-2B on this card the allocation is about
    1.2 GB per layer (73k pages x 16 KiB), i.e. 56% of the int32 range -- it fits, but
    only by 1.8x, so it is asserted per call rather than assumed.
    """
    return (2 * view.num_blocks * view.s_blk <= INT32_MAX
            and n * max(view.hkv * view.d, 1) <= INT32_MAX)


def write_kv_int8(key, value, view: CacheView, slot_mapping, *, cfg=None, kernel=None,
                  rne=True):
    """Drop-in for reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping,
    ...): the token count is slot_mapping.shape[0] (key / value may be padded), slots < 0
    are skipped.  No allocation, no host sync, grid from the token count only, so a FULL
    decode graph can capture it."""
    n = int(slot_mapping.shape[0])
    if n == 0:
        return
    k3 = key if key.dim() == 3 else key.view(key.shape[0], -1, view.d)
    v3 = value if value.dim() == 3 else value.view(value.shape[0], -1, view.d)
    which = (kernel or WRITE_KERNEL)
    forced = which in ("v1", "v2")
    c = cfg or write_cfg_for(n, which if forced else None)
    if not forced:
        which = c.get("kernel", "v2")
    if which == "v2" and not write_int32_safe(view, n):
        which = "v1"                      # addresses would overflow int32: use the R8 one
        if cfg is None:
            c = write_cfg_for(n, "v1")
    if not rne and which == "v1":
        # rne is a v2-only measurement knob (v1 always rounds half to even), so silently
        # honouring the plan here would report a number for a kernel nobody asked for
        raise RuntimeError(f"rne=False needs kernel='v2' (the plan chose v1 for n = {n})")
    if which == "v2":
        grid = (_cdiv(n, c["block_t"]), view.hkv)
        _q1_write_kv2[grid](k3, v3, slot_mapping, view.data, view.scales, n,
                            k3.stride(0), k3.stride(1), v3.stride(0), v3.stride(1),
                            *view.args(), HKV=view.hkv, D=view.d, PAGE=view.page,
                            BLOCK_T=c["block_t"], DIMG=DIMG, RNE=1 if rne else 0,
                            **launch_kwargs(c["warps"], c["stages"]))
        return
    grid = (_cdiv(n, c["block_t"]),)
    _q1_write_kv[grid](k3, v3, slot_mapping, view.data, view.scales, n,
                       k3.stride(0), k3.stride(1), v3.stride(0), v3.stride(1),
                       *view.args(), HKV=view.hkv, D=view.d, PAGE=view.page,
                       BLOCK_T=c["block_t"], **launch_kwargs(c["warps"], c["stages"]))


# ===========================================================================
# (d) PREFILL READ: paged int8 -> contiguous bf16 gather with the dequantisation folded in
# ===========================================================================
@triton.jit
def _q1_gather_dequant(D8, D32, KO, VO, BT, SEQ, CUK, stride_bt,
                       k_base, v_base, s_blk, s_head, sk_base, sv_base, ss_blk,
                       HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                       FILL: tl.constexpr):
    pb = tl.program_id(0)
    s = tl.program_id(1)
    L = tl.load(SEQ + s)
    if pb * PAGE >= L:
        return
    blk = tl.load(BT + s.to(tl.int64) * stride_bt + pb).to(tl.int64)
    dst0 = tl.load(CUK + s).to(tl.int64) + pb * PAGE
    r = tl.arange(0, PAGE * HKV)
    t = r // HKV
    h = r % HKV
    d = tl.arange(0, D)
    m = (pb * PAGE + t) < L
    hbase = blk * s_blk + h * s_head
    dg = d // 8
    dr = d % 8
    ksrc = k_base + hbase[:, None] + dg[None, :] * (PAGE * 8) + t[:, None] * 8 + dr[None, :]
    vsrc = v_base + hbase[:, None] + t[:, None] * D + d[None, :]
    k = tl.load(D8 + ksrc, mask=m[:, None], other=0).to(tl.float32)
    v = tl.load(D8 + vsrc, mask=m[:, None], other=0).to(tl.float32)
    sidx = blk * ss_blk + h * PAGE + t
    ks = tl.load(D32 + sk_base + sidx, mask=m, other=0.0)
    vs = tl.load(D32 + sv_base + sidx, mask=m, other=0.0)
    k = (k * ks[:, None]).to(KO.dtype.element_ty)
    v = (v * vs[:, None]).to(VO.dtype.element_ty)
    dst = (dst0 + t)[:, None] * (HKV * D) + h[:, None] * D + d[None, :]
    if FILL:
        # page-aligned destination (the per-row KV-split readers see whole pages): write
        # zeros into the slots past the row's end instead of leaving them uninitialised
        tl.store(KO + dst, tl.where(m[:, None], k, 0.0))
        tl.store(VO + dst, tl.where(m[:, None], v, 0.0))
    else:
        tl.store(KO + dst, k, mask=m[:, None])
        tl.store(VO + dst, v, mask=m[:, None])


GATHER_WARPS = 4


def gather_dequant(view: CacheView, block_table, seq_lens, cu_k, max_seq_len, total_k, *,
                   dtype=torch.bfloat16, warps=GATHER_WARPS, fill=False, out=None):
    """First seq_lens[s] tokens of block_table row s -> contiguous (total_k, HKV, D)
    bf16 buffers at offsets cu_k[s], dequantising on the way.  max_seq_len / total_k are
    host ints the metadata builder already has, so there is no device sync.  fill=True
    also zeroes the tail of the last page of every row (page-aligned destinations)."""
    if out is None:
        k_out = torch.empty(total_k, view.hkv, view.d, dtype=dtype, device=view.data.device)
        v_out = torch.empty_like(k_out)
    else:
        k_out, v_out = out
    if total_k == 0:
        return k_out, v_out
    grid = (_cdiv(int(max_seq_len), view.page), int(seq_lens.shape[0]))
    _q1_gather_dequant[grid](view.data, view.scales, k_out, v_out, block_table, seq_lens,
                             cu_k, block_table.stride(0), *view.args(), HKV=view.hkv,
                             D=view.d, PAGE=view.page, FILL=fill,
                             **launch_kwargs(warps, 1))
    return k_out, v_out


# ===========================================================================
# (d) prefill policy: every composition is covered by our own gather
# ===========================================================================
ROWSPLIT = True          # keep the T4 per-row KV-split launches for prompt-ending steps
TAIL_MIN_K = 10241       # same feature as mx_prefill_attn.TAIL_MIN_K
TAIL_MAX_ROWS = 4
CAP_TOKENS = 32768       # per-group gather bound (set from max_model_len at setup)


def pad16(n: int) -> int:
    return (n + 15) & ~15


def prefill_plan(host_q, host_k, *, cap=None, rowsplit=None):
    """Groups of consecutive prefill rows, each gathered into ONE contiguous dequantised
    K/V buffer, with the vendor call to use on it.  Feature-driven: row count, kv lengths
    and the scratch bound only.  Returns [{lo, hi, reader, pad, total, kv}], where
    reader "varlen" = one non-paged varlen_fwd (today's gather path, extended to every
    chunk) and "rowsplit" = one fwd_kvcache per row over the page VIEW of the gathered
    buffer (the T4 tail policy, which needs page-aligned offsets).

    Every group's reader reads the dequantised buffer, so no vendor kernel ever sees the
    int8 pages.  A single row always fits one group because cap >= pad16(max_model_len).
    """
    cap = CAP_TOKENS if cap is None else cap
    rs = ROWSPLIT if rowsplit is None else rowsplit
    n = len(host_k)
    out = []
    lo = 0
    while lo < n:
        hi, tot = lo, 0
        while hi < n:
            need = pad16(host_k[hi])
            if hi > lo and tot + need > cap:
                break
            if hi == lo and need > cap:
                raise ValueError(f"prefill row k={host_k[hi]} exceeds the gather cap {cap}")
            tot += need
            hi += 1
        kv = host_k[lo:hi]
        tail = rs and 2 <= (hi - lo) <= TAIL_MAX_ROWS and max(kv) >= TAIL_MIN_K
        out.append(dict(lo=lo, hi=hi, reader="rowsplit" if tail else "varlen",
                        pad=bool(tail), total=tot if tail else sum(kv), kv=tuple(kv)))
        lo = hi
    return out


def plan_offsets(kv, pad: bool):
    """Destination offsets of the rows of one group inside its gather buffer."""
    off, out = 0, []
    for k in kv:
        out.append(off)
        off += pad16(k) if pad else k
    return out, off


def fake_block_table(offsets, kv, page: int, device, dtype=torch.int32):
    """Block table of the gathered buffer seen as pages: row s holds the page ids
    offsets[s]//page + 0, 1, ...  Built once per step (host ints -> one small H2D), never
    per layer; only needed by the "rowsplit" reader."""
    width = max(1, max(_cdiv(k, page) for k in kv))
    bt = torch.zeros(len(kv), width, dtype=dtype)
    for s, (o, k) in enumerate(zip(offsets, kv)):
        m = _cdiv(k, page)
        bt[s, :m] = torch.arange(o // page, o // page + m, dtype=dtype)
    return bt.to(device, non_blocking=True)


def prefill_attention(mxa, q, view: CacheView, out, block_table, cu_q, cu_k, seq_lens,
                      host_q, host_k, scale, *, plan=None, bt_fake=None, cu_pad=None,
                      go=None):
    """Prefill rows against the INT8 cache.  mxa = ops/mx_prefill_attn (vendor C entry
    points, split policy).  plan / bt_fake / cu_pad come from the metadata builder (one
    per step); without them they are built here (bench / self-test path).

    q / out: (n_prefill_tokens, HQ, D).  cu_k: [0, cumsum(k_lens)] (nseq + 1 entries, the
    builder's prefill_cu_seqlens_k); seq_lens: the prefill rows' kv lengths.  go: the P1 overlap event, fired right before the kernel the decode rows should
    run next to (after the gather), exactly as in mx_prefill_attn.prefill_attention.
    """
    plan = plan or prefill_plan(host_q, host_k)
    hq, d = q.shape[1], q.shape[2]
    qo = 0
    fired = False
    for gi, grp in enumerate(plan):
        lo, hi = grp["lo"], grp["hi"]
        kv = grp["kv"]
        offs, total = plan_offsets(kv, grp["pad"])
        rows = hi - lo
        nq = sum(host_q[lo:hi])
        if grp["pad"]:
            # page-aligned destinations: the gather only needs the row starts
            cu = cu_pad[gi] if cu_pad is not None else torch.tensor(
                offs, dtype=torch.int32, device=q.device)
        else:
            # cu_k carries nseq + 1 entries (the vendor varlen wants the closing one);
            # the gather kernel reads the first `rows` of them as the row starts
            cu = cu_k[lo:hi + 1] - cu_k[lo] if lo else cu_k[:hi + 1]
        kc, vc = gather_dequant(view, block_table[lo:hi], seq_lens[lo:hi], cu,
                                max(kv), total, fill=grp["pad"])
        if go is not None and not fired:
            go.fire()                      # after the gather, before the first kernel
            fired = True
        if grp["reader"] == "rowsplit":
            bt = bt_fake[gi] if bt_fake is not None else fake_block_table(
                offs, kv, view.page, q.device)
            kp = kc.view(-1, view.page, view.hkv, d)
            vp = vc.view(-1, view.page, view.hkv, d)
            a = qo
            for s in range(rows):
                ql, kl = host_q[lo + s], kv[s]
                ns = max(1, int(round(mxa.seq_splits(ql, kl, hq) * mxa.TAIL_SCALE)))
                mxa.kvcache_fwd(q[a:a + ql].view(1, ql, hq, d), kp, vp,
                                seq_lens[lo + s:lo + s + 1], bt[s:s + 1], scale,
                                num_splits=ns, out=out[a:a + ql].view(1, ql, hq, d))
                a += ql
        else:
            cuq = cu_q[lo:hi + 1] - cu_q[lo] if lo else cu_q[:hi + 1]
            mxa.varlen_fwd(q[qo:qo + nq], kc, vc, cuq, cu, max(host_q[lo:hi]), max(kv),
                           scale, out=out[qo:qo + nq])
        qo += nq
    return out


def decode_varlen(mxa, q, view: CacheView, out, block_table, cu_q, cu_k, seq_lens,
                  host_q, host_k, scale, *, plan=None):
    """R9/F: the step's DECODE rows when they carry more than one query token each.

    The native int8 kernel reads one query token per row, but the backend routes every
    request with query_len <= reorder_batch_threshold (128) through the decode region,
    so a short prompt -- any MATH-500 prompt, or the 14-token smoke request that made
    the R8 ON bench raise -- is a decode row with q_len > 1.  Such a row is a prefill
    row in everything that matters to a reader: q tokens at the END of a kv range of
    seq_lens[r], causal.  So it takes the same gather-and-dequantise reader as the
    prefill rows, which is the only int8 reader that serves q_len > 1, and the vendor
    kernel again never sees an int8 page.  Feature-driven on the step's q lengths
    (num_decode_tokens != num_decodes), decided once per step by the metadata builder.

    This is a correctness route, not a fast path: at concurrency 64 over 4k/16k prompts
    every decode-only step has q_len == 1 and goes to the native kernel as before.
    """
    return prefill_attention(mxa, q, view, out, block_table, cu_q, cu_k, seq_lens,
                             host_q, host_k, scale, plan=plan)


# ===========================================================================
# torch references (self-test, bench, offline tests)
# ===========================================================================
@torch.no_grad()
def quantize_rows(x, qmax=QMAX):
    """Per-row symmetric absmax int8 of x[..., D] -> (int8, fp32 scale).

    Reference for the write kernel, and it has to use the SAME rounding rule: round
    half to EVEN (torch.round), not the half-away-from-zero the kernel used to do.
    ``inv = qmax / a`` and the clamp mirror _q1_write_kv exactly, so the only
    remaining source of disagreement is the last-bit rounding of ``xf * inv`` itself
    (see write_codes_report)."""
    xf = x.float()
    a = xf.abs().amax(dim=-1)
    inv = torch.where(a > 0, qmax / a, torch.zeros_like(a))
    y = (xf * inv.unsqueeze(-1)).clamp(-qmax, qmax)
    return torch.round(y).to(torch.int8), (a / qmax).float()


@torch.no_grad()
def write_codes_report(x, codes, scales, qmax=QMAX, tie=2e-4):
    """R9/F: exactness of a quantising write, measured on the STORED (code, scale) pair.

    x: the written rows (..., D) as they were handed to the writer; codes / scales: what
    the cache now holds, as read by ``read_cache_torch(..., raw=True)`` -- codes (..., D)
    and the per-row scale (..., 1), both fp32.

    Comparing DEQUANTISED values instead (what bench_q1.py:754 does) folds a 1-ulp
    difference in the stored scale into every element of that row, which is why that
    check reported 4.6% of elements differing while the code-level check in the
    start-up self-test reported 0.032%: those are two different measurements, and only
    the code-level one says anything about the rounding.

    Returns a dict of floats:
      scale_ulp  max |stored scale - absmax/qmax|, in ulps of the reference scale
      moved      fraction of codes != torch.round(x * qmax/absmax)
      far        max |stored code - reference code|
      off_tie    fraction of codes that differ and are NOT excused.  A difference is
                 excused only when the exact product lies within `tie` of a rounding
                 tie but not exactly ON it: there the two sides may have computed the
                 product one ulp apart (an FMA contraction or a reciprocal multiply
                 costs about that) and both codes are the correct RNE of what each
                 computed.  A difference exactly AT a tie is never excused, because
                 both sides compute that product exactly -- a value x with absmax a is
                 an exact tie only when a's bf16 mantissa is 254, and then 127/a is a
                 power of two and the product is exact on any implementation.  So
                 off_tie == 0 pins the rounding RULE: a truncating writer fails it on
                 about half the elements, the old half-away-from-zero writer on the
                 ties.
      err        max |dequant - x| in units of the stored quantisation step (<= 0.5)
    """
    xf = x.float()
    a = xf.abs().amax(dim=-1)
    s_ref = a / qmax
    inv = torch.where(a > 0, qmax / a, torch.zeros_like(a))
    y = (xf * inv.unsqueeze(-1)).clamp(-qmax, qmax)
    q_ref = torch.round(y)
    s_got = scales.reshape(s_ref.shape).float()
    eps = torch.finfo(torch.float32).eps
    ulp = (s_ref.abs() * eps).clamp_min(torch.finfo(torch.float32).tiny)
    diff = (codes.float() - q_ref).abs()
    moved = diff > 0
    # distance of the exact product from the nearest rounding tie (n + 1/2)
    tie_dist = ((y - torch.floor(y)) - 0.5).abs()
    excused = (tie_dist > 0) & (tie_dist < tie)
    step = s_got.unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
    n = max(1, diff.numel())
    return dict(
        scale_ulp=float(((s_got - s_ref).abs() / ulp).max()),
        moved=float(moved.float().mean()),
        far=float(diff.max()),
        off_tie=float((moved & ~excused).float().sum()) / n,
        err=float(((codes.float() * s_got.unsqueeze(-1) - xf).abs() / step).max()),
    )


@torch.no_grad()
def fake_quant(x, qmax=QMAX):
    """dequant(quant(x)) in fp32: what every reader of the INT8 cache sees."""
    q, s = quantize_rows(x, qmax)
    return q.float() * s.unsqueeze(-1)


@torch.no_grad()
def read_cache_torch(view: CacheView, blocks, which="k", raw=False):
    """Dequantised (len(blocks), PAGE, HKV, D) fp32 straight from the int8 view.

    raw=True returns the stored pair instead: (codes fp32, scales (..., 1) fp32), which
    is what an exact check of the write needs -- comparing dequantised values folds a
    1-ulp difference in the stored scale into every element of a row."""
    dev = view.data.device
    page, hkv, d = view.page, view.hkv, view.d
    base = view.k_base if which == "k" else view.v_base
    sbase = view.sk_base if which == "k" else view.sv_base
    blk = torch.as_tensor(blocks, dtype=torch.int64, device=dev)
    t = torch.arange(page, device=dev)
    h = torch.arange(hkv, device=dev)
    dd = torch.arange(d, device=dev)
    if which == "k":
        inner = (dd // 8) * (page * 8) + (dd % 8)
        off = (base + blk[:, None, None, None] * view.s_blk
               + h[None, None, :, None] * view.s_head
               + t[None, :, None, None] * 8 + inner[None, None, None, :])
    else:
        off = (base + blk[:, None, None, None] * view.s_blk
               + h[None, None, :, None] * view.s_head
               + t[None, :, None, None] * d + dd[None, None, None, :])
    q = view.data[off.reshape(-1)].view(len(blk), page, hkv, d).float()
    sidx = (sbase + blk[:, None, None] * view.ss_blk + h[None, None, :] * page
            + t[None, :, None])
    s = view.scales[sidx.reshape(-1)].view(len(blk), page, hkv, 1)
    if raw:
        return q, s
    return q * s


@torch.no_grad()
def ref_decode(q3, kc_f32, vc_f32, block_table, seq_lens, scale):
    """fp32 attention from dequantised page tensors (kc/vc: (NB, PAGE, HKV, D) fp32)."""
    n, hq, d = q3.shape
    hkv = kc_f32.shape[2]
    g = hq // hkv
    out = torch.zeros(n, hq, d, dtype=torch.float32, device=q3.device)
    bs = kc_f32.shape[1]
    for r in range(n):
        k = int(seq_lens[r])
        if k <= 0:
            continue
        pos = torch.arange(k, device=q3.device)
        blk = block_table[r][pos // bs].long()
        K = kc_f32[blk, pos % bs]
        V = vc_f32[blk, pos % bs]
        Q = q3[r].float().view(hkv, g, d)
        S = torch.einsum("hgd,khd->hgk", Q, K) * scale
        P = torch.softmax(S, dim=-1)
        out[r] = torch.einsum("hgk,khd->hgd", P, V).reshape(hq, d)
    return out


# ===========================================================================
# policy / interlock
# ===========================================================================
ENABLED = False
STATUS = "not set up"
_SETUP_DONE = False
SELFTEST_CTX = (0, 1, 15, 16, 17, 333, 1500, 4100)
# second context set for the graph replay test (clamped to each row's written context)
SELFTEST_CTX2 = (2048, 0, 16, 1, 4099, 48, 0, 700)
# R9/F: capture the int8 decode-only config in a CUDA graph and replay it (0 = skip)
GRAPH_SELFTEST = os.environ.get("VLLM_FL_MX_INT8_GRAPHTEST", "1") not in ("0", "off", "no")
GATE_REASONS: list = []


def gates_ok(*, cache_dtype_str: str, dcp_world_size: int, kv_sharing: bool,
             cascade_enabled: bool, block_size: int, num_heads_q: int, num_heads_kv: int,
             head_size: int, dtype) -> list:
    """Everything that would let a reader other than ours touch the cache.  Returns the
    list of failed gates (empty = ok)."""
    bad = []
    if cache_dtype_str not in ("auto", "bfloat16"):
        bad.append(f"cache_dtype={cache_dtype_str}")
    if dcp_world_size and dcp_world_size > 1:
        bad.append(f"dcp={dcp_world_size}")
    if kv_sharing:
        bad.append("kv_sharing")
    if cascade_enabled:
        bad.append("cascade")
    if block_size != PAGE_TILE:
        bad.append(f"block_size={block_size}")
    if head_size % DIMG or head_size != 128:
        bad.append(f"head_size={head_size}")
    if num_heads_kv < 1 or num_heads_q % num_heads_kv:
        bad.append(f"heads={num_heads_q}/{num_heads_kv}")
    if dtype != torch.bfloat16:
        bad.append(f"dtype={dtype}")
    return bad


def decode_supported(num_decodes: int, num_decode_tokens: int) -> bool:
    """True when the native int8 decode kernel can serve this step's decode rows, i.e.
    q_len == 1 on every one of them (one query token per row is what it reads).

    R9/F: this is NOT a property the feature can gate on at start-up, which is the bug
    that killed the R8 ON bench.  The backend classifies every request with
    query_len <= reorder_batch_threshold (128, flash_attn.py:432) as a "decode" row, so
    any prompt of 2..128 tokens -- every MATH-500 prompt, and the 14-token smoke
    request that raised at flash_attn.py:1554 -- arrives here as a decode row with
    q_len > 1.  Those rows now go through decode_varlen() instead."""
    return num_decodes > 0 and num_decode_tokens == num_decodes


# R9/F: the decode rows of an OVERLAPPED mixed step run on a side stream next to the
# prefill kernel, and there the int8 kernel is measured SLOWER than the vendor bf16 one
# at few rows.  R9/G re-measured the gate on the SHIPPED readers (cfg 42 + PV fold on the
# main stream, cfg 47 on the side stream; us/layer over 42 layers,
# /data/exp/R9/G/logs/r9_full_20261025_075543.json):
#
#   rows / ctx     vendor bf16   side cfg 47      main cfg 42+fold   (R8: side 38 / main 28)
#   16 @ 4k            80.4       94.1 x0.855       75.7 x1.063       x0.786 / x0.931
#   48 @ 4k           183.7      150.8 x1.218      125.9 x1.459       x1.008 / x1.119
#   64 @ 4k           243.8      182.8 x1.334      156.8 x1.555       x1.099 / x1.171
#   48 @ 16k          601.5      388.1 x1.550      349.4 x1.721       x1.159 / x1.209
#   64 @ 16k          784.7      469.7 x1.670      440.3 x1.782       x1.227 / x1.245
#
# With an int8 cache there is no vendor fallback for a row, so the gate cannot pick
# another reader -- it picks another SCHEDULE: below the threshold the builder does not
# overlap the step at all and the rows go out on the main stream with cfg 42, which now
# beats the vendor at EVERY measured shape, 16 rows included (that is the R8 defect the
# fp16 dequantisation fixed: cfg 28 was x0.931 there).  The side-stream config is still
# below parity at 16 rows (x0.855) and above it at 48 (x1.218), and 24 rows was not
# re-measured in R9/G, so the threshold stays where R8's 24-row reading put it.  At the
# contest's concurrency 64 every overlapped step has 64 decode rows, so this gate is
# about not regressing a small-batch workload, not about the score.  Feature-driven on
# the decode row count only.
SIDE_MIN_ROWS = int(os.environ.get("VLLM_FL_MX_INT8_SIDE_MIN_ROWS", "24"))


def side_ok(num_decodes: int) -> bool:
    """May this step's decode rows be overlapped onto the side stream (int8 only)?"""
    return num_decodes >= SIDE_MIN_ROWS


def step_policy(num_decodes: int, num_decode_tokens: int, overlap: bool):
    """The INT8 cache's whole per-step decision, as one pure function of the step's
    shape so that it can be checked offline (offline/test_f_onpath.py).  Returns
    (dec_varlen, overlap), where

      dec_varlen  the decode rows carry more than one query token each, so they go to
                  decode_varlen() (the gather reader) instead of the native kernel;
      overlap     may this step still be overlapped?  `overlap` in is what the bf16
                  policy decided (_mxa.overlap_wanted); it is withdrawn when the rows
                  need the gather reader (which has no side-stream site) or when there
                  are too few of them for the side-stream int8 config to pay.

    Feature-driven on the row and token counts only -- never on the case.  At
    concurrency 64 over 4k/16k prompts the answer is (False, overlap) on every step.
    """
    dec_varlen = num_decodes > 0 and num_decode_tokens != num_decodes
    return dec_varlen, bool(overlap and not dec_varlen and side_ok(num_decodes))


# ===========================================================================
# start-up self-test (runs once, before the first KV write and before graph capture)
# ===========================================================================
def _self_test(device, mxn, mxa, hq, hkv, d, page, max_rows, cfg_dec, cfg_side) -> str:
    """Write -> both readers -> references, on a private little cache."""
    gen = torch.Generator().manual_seed(0)
    ctx = [c for c in SELFTEST_CTX]
    n = len(ctx)
    nb = sum(_cdiv(c, page) for c in ctx) + 5
    width = max(1, max(_cdiv(c, page) for c in ctx)) + 2
    kv = torch.zeros(2, nb, page, hkv, d, dtype=torch.bfloat16, device=device)
    # poison every slot: a never-written page must not be able to produce NaN
    kv.view(torch.int32).random_(-(2 ** 31), 2 ** 31 - 1, generator=None)
    view = int8_view(kv)
    bt_cpu = torch.zeros(n, width, dtype=torch.int32)
    perm = torch.randperm(nb, generator=gen).to(torch.int32)
    i = 0
    for r, c in enumerate(ctx):
        m = _cdiv(c, page)
        bt_cpu[r, :m] = perm[i:i + m]
        i += m
    bt = bt_cpu.to(device)
    seq = torch.tensor(ctx, dtype=torch.int32, device=device)
    total = sum(ctx)
    report = []

    # ---- write path -------------------------------------------------------------
    k_in = (torch.randn(total, hkv, d, generator=gen) * 1.3).to(torch.bfloat16).to(device)
    v_in = (torch.randn(total, hkv, d, generator=gen) * 0.8).to(torch.bfloat16).to(device)
    slots = torch.empty(total, dtype=torch.int64)
    j = 0
    for r, c in enumerate(ctx):
        for t in range(c):
            slots[j] = int(bt_cpu[r, t // page]) * page + (t % page)
            j += 1
    slots = slots.to(device, dtype=torch.int64)
    write_kv_int8(k_in, v_in, view, slots)
    torch.cuda.synchronize(device)
    blocks = sorted({int(s) // page for s in slots.cpu().tolist()})
    kc = read_cache_torch(view, blocks, "k")
    vc = read_cache_torch(view, blocks, "v")
    bmap = {b: i for i, b in enumerate(blocks)}
    idx = torch.tensor([[bmap[int(s) // page], int(s) % page] for s in slots.cpu().tolist()])
    got_k = kc[idx[:, 0], idx[:, 1]]
    # R9/F: the write is checked on the STORED (code, scale) pair, which is the only
    # level at which "exact" means anything (see write_codes_report).  Three facts must
    # hold, and all three are exact -- no tolerance is fitted to the measurement:
    #   * the stored scale is absmax/127 to within two ulps, which is what a
    #     reciprocal multiply instead of a correctly-rounded division can cost (one
    #     rounding each) and nothing else may: a 1% scale error measures 8.4e4 ulps;
    #   * every stored code is torch.round(x * 127/absmax), i.e. round half to EVEN,
    #     except where the exact product lies within 2e-4 of a tie -- there the two
    #     codes are both the correct RNE of a product two correctly-rounded
    #     implementations may compute one ulp apart.  off_tie must be 0: a truncating
    #     or half-away-from-zero writer fails it at ~50% / ~1e-3 of the entries;
    #   * no dequantised value is more than half a quantisation step from the input.
    moved = 0.0
    for nm, which, src in (("k", "k", k_in), ("v", "v", v_in)):
        cds, scs = read_cache_torch(view, blocks, which, raw=True)
        rep = write_codes_report(src, cds[idx[:, 0], idx[:, 1]], scs[idx[:, 0], idx[:, 1]])
        moved = max(moved, rep["moved"])
        if rep["scale_ulp"] > 2.0:
            raise RuntimeError(f"{nm} write: the stored scale is {rep['scale_ulp']:.1f} "
                               f"ulps from absmax/{QMAX:.0f}")
        if rep["off_tie"] > 0.0 or rep["far"] > 1.0:
            raise RuntimeError(
                f"{nm} write: {100 * rep['off_tie']:.3f}% of the codes differ from "
                f"round-half-to-even away from a tie (max {rep['far']:.0f} steps); the "
                f"rounding rule is wrong, not the last bit")
        if rep["err"] > 0.5 + 2e-4:
            raise RuntimeError(f"{nm} write: a dequantised value is {rep['err']:.4f} "
                               f"quantisation steps from the input")
    # and the write must be deterministic: the same input twice must leave the same bytes
    before = view.data.clone()
    write_kv_int8(k_in, v_in, view, slots)
    torch.cuda.synchronize(device)
    if not torch.equal(before, view.data):
        raise RuntimeError("write is not deterministic (two writes of one input differ)")
    # R9/G: and the two writers must agree BIT FOR BIT on the DEVICE, not only in the
    # offline numpy execution of the same programs -- the BLOCK_T 1/2 absmax lowering bug
    # was invisible to the latter.  WRITE_PLAN uses both (v2 at <= 128 slots, v1 above),
    # so this runs whichever one the calls above did not.
    other = "v1" if write_cfg_for(len(slots))["kernel"] == "v2" else "v2"
    write_kv_int8(k_in, v_in, view, slots, kernel=other)
    torch.cuda.synchronize(device)
    if not torch.equal(before, view.data):
        raise RuntimeError(f"the two write kernels disagree: writer {other} does not "
                           f"store the bytes WRITE_PLAN's choice stores (n = {len(slots)})")
    qerr = float((got_k - k_in.float()).abs().max() / k_in.float().abs().max())
    report.append(f"write=exact moved={100 * moved:.3f}% qmax={qerr:.3f}")

    # ---- decode read path (native kernel, Q8 = 1) --------------------------------
    kc_full = torch.zeros(nb, page, hkv, d, device=device)
    vc_full = torch.zeros(nb, page, hkv, d, device=device)
    kc_full[blocks] = kc
    vc_full[blocks] = vc
    q = (torch.randn(n, hq, d, generator=gen) * 2.0).to(torch.bfloat16).to(device)
    scale = d ** -0.5
    ref = ref_decode(q, kc_full, vc_full, bt, seq, scale)
    for name, cfg in (("dec", cfg_dec), ("side", cfg_side)):
        side = name == "side"
        got = mxn.decode_attention_int8(q, kv, view, bt, seq, scale, cfg=cfg, side=side)
        if got is None:
            raise RuntimeError(f"native int8 decode rejected the {name} site")
        torch.cuda.synchronize(device)
        if not torch.isfinite(got.float()).all():
            raise RuntimeError(f"native int8 decode {name}: non-finite output")
        e = float((got.float() - ref).abs().max() / max(float(ref.abs().max()), 1e-6))
        if e > 2.5e-2:
            raise RuntimeError(f"native int8 decode {name}: rel err {e:.3e}")
        again = mxn.decode_attention_int8(q, kv, view, bt, seq, scale, cfg=cfg, side=side)
        if not torch.equal(got, again):
            raise RuntimeError(f"native int8 decode {name}: two launches differ")
        report.append(f"{name}={e:.1e}")

    # R9/F: every decode ROW COUNT the scheduler can produce must be accepted.  There
    # is no vendor fallback once the pages are int8, so a rejection at run time is a
    # crash; nd = 1 (one short request still decoding) is the count the R8 ON run never
    # reached, and the 8-row batch above does not prove it.
    for nr in sorted({1, 2, max(1, n // 2), n}):
        got = mxn.decode_attention_int8(q[:nr], kv, view, bt[:nr], seq[:nr], scale,
                                        cfg=cfg_dec, side=False)
        if got is None:
            raise RuntimeError(f"native int8 decode rejected {nr} row(s)")
        torch.cuda.synchronize(device)
        e = float((got.float() - ref[:nr]).abs().max() / max(float(ref.abs().max()), 1e-6))
        if not torch.isfinite(got.float()).all() or e > 2.5e-2:
            raise RuntimeError(f"native int8 decode, {nr} row(s): rel err {e:.3e}")
    report.append("rows=1..%d" % n)

    # ---- the decode-only site inside a CUDA graph --------------------------------
    # R9/F: the decode-only site is the one a FULL decode graph captures, and there the
    # launch configuration is frozen while the contexts keep growing, so capture it once
    # and replay it with two different context sets.  decode_attention_int8() reads
    # torch.cuda.current_stream() at CALL time (mx_dec_native.py:499), which is what puts
    # the launch on the CAPTURE stream rather than the default one -- the R8/N4 bug that
    # produced an empty graph and a nan replay.  A failure here raises inside
    # _kv8.setup()'s own try, so it disables the INT8 cache ONLY: _mxn.ENABLED and N1's
    # bf16 mixed-step site are not touched by this module at all.
    if GRAPH_SELFTEST:
        seq_g = seq.clone()
        out_g = torch.full((n, hq, d), float("nan"), dtype=torch.bfloat16, device=device)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            # Deliberately no raise inside the capture: a rejection here enqueues
            # nothing, and an empty graph is reported by capture_end() on the way out
            # (or, failing that, by the replay comparison below).
            mxn.decode_attention_int8(q, kv, view, bt, seq_g, scale, cfg=cfg_dec,
                                      side=False, out=out_g)
        # never longer than the row's own written context: a longer replay would read
        # pages this test never wrote, whose poison fill can hold a NaN scale
        ctx2 = [min(a, b) for a, b in zip((SELFTEST_CTX2 * 4)[:n], ctx)]
        for tag, cx in (("A", list(ctx)), ("B", ctx2), ("A2", list(ctx))):
            seq_g.copy_(torch.tensor(cx, dtype=torch.int32, device=device))
            out_g.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            eager = mxn.decode_attention_int8(q, kv, view, bt, seq_g, scale, cfg=cfg_dec,
                                              side=False)
            torch.cuda.synchronize(device)
            if eager is None or not torch.equal(out_g, eager):
                bad = float("nan") if eager is None else float(
                    (out_g.float() - eager.float()).abs().max())
                raise RuntimeError(f"int8 graph self-test {tag}: replay != eager "
                                   f"(max {bad:.3e})")
        del graph
        torch.cuda.synchronize(device)
        report.append("graph=bitstable")

    # ---- prefill read path (gather + the vendor kernels) --------------------------
    live = [r for r, c in enumerate(ctx) if c > 0]
    host_k = tuple(ctx[r] for r in live)
    bt_l = bt[live]
    seq_l = seq[live]
    cu = torch.tensor([0] + list(torch.tensor(host_k).cumsum(0)), dtype=torch.int32,
                      device=device)
    kg, vg = gather_dequant(view, bt_l, seq_l, cu[:-1], max(host_k), sum(host_k))
    torch.cuda.synchronize(device)
    for s, r in enumerate(live):
        c = ctx[r]
        pos = torch.arange(c, device=device)
        blk = bt[r][pos // page].long()
        # the gather writes bf16: the reference is bf16(dequant), and both sides do the
        # same single fp32 multiply + RTNE, so the comparison is exact
        want = kc_full[blk, pos % page].to(torch.bfloat16).float()
        got = kg[int(cu[s]):int(cu[s]) + c].float()
        if float((got - want).abs().max()) > 0:
            raise RuntimeError(f"gather K row {r} differs from the cache")
        want = vc_full[blk, pos % page].to(torch.bfloat16).float()
        got = vg[int(cu[s]):int(cu[s]) + c].float()
        if float((got - want).abs().max()) > 0:
            raise RuntimeError(f"gather V row {r} differs from the cache")
    report.append("gather=exact")

    # ---- multi-token decode rows (R9/F: decode_varlen, the short-request route) ----
    # The shape that made the R8 ON bench raise: rows in the DECODE region carrying more
    # than one query token (any prompt of 2..128 tokens).  They go through the same
    # gather + vendor varlen reader as the prefill rows, so run the real call once here,
    # over a composition the server can produce (q = min(ctx, 14) at the end of the
    # row's own context), and check it against the fp32 reference of the LAST query
    # token of each row -- which is exactly a decode of that row.
    dq = tuple(min(14, c) for c in host_k)
    cu_q8 = torch.tensor([0] + list(torch.tensor(dq).cumsum(0)), dtype=torch.int32,
                         device=device)
    qv = (torch.randn(sum(dq), hq, d, generator=gen) * 2.0).to(torch.bfloat16).to(device)
    ov = torch.zeros_like(qv)
    decode_varlen(mxa, qv, view, ov, bt_l, cu_q8, cu, seq_l, dq, host_k, scale)
    torch.cuda.synchronize(device)
    if not torch.isfinite(ov.float()).all():
        raise RuntimeError("decode_varlen: non-finite output")
    last = torch.tensor([int(cu_q8[s + 1]) - 1 for s in range(len(dq))], device=device)
    ref_last = ref_decode(qv[last], kc_full, vc_full, bt_l, seq_l, scale)
    ev = float((ov[last].float() - ref_last).abs().max()
               / max(float(ref_last.abs().max()), 1e-6))
    if ev > 2.5e-2:
        raise RuntimeError(f"decode_varlen: rel err {ev:.3e} on the last query token")
    report.append(f"decvarlen={ev:.1e}")

    # every plan the shapes can produce must be covered and must reconstruct the rows
    for comp in ((2048,), (4096,), (1024, 16400), (16400, 1024, 300), tuple(host_k)):
        for grp in prefill_plan(tuple(min(2048, k) for k in comp), comp):
            if grp["reader"] not in ("varlen", "rowsplit"):
                raise RuntimeError(f"plan: uncovered reader {grp}")
    report.append("plan=covered")
    del kv, kc_full, vc_full
    return " ".join(report)


def setup(device, *, mxn, mxa, num_heads_q, num_heads_kv, head_size, block_size, dtype,
          max_rows, max_model_len, cache_dtype_str="auto", dcp_world_size=1,
          kv_sharing=False, cascade_enabled=False, cfg_dec=None, cfg_side=None,
          rope_capture_safe=True) -> str:
    """Once per process, from the metadata builder (before the first forward, before
    CUDA-graph capture).  Latches ENABLED; never flips afterwards."""
    global ENABLED, STATUS, _SETUP_DONE, CAP_TOKENS, GATE_REASONS
    if _SETUP_DONE:
        return STATUS
    _SETUP_DONE = True
    ENABLED = False
    try:
        if not REQUESTED:
            raise RuntimeError(f"{ENV_FLAG}={MODE or POLICY_DEFAULT}")
        GATE_REASONS = gates_ok(
            cache_dtype_str=cache_dtype_str, dcp_world_size=dcp_world_size,
            kv_sharing=kv_sharing, cascade_enabled=cascade_enabled,
            block_size=block_size, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
            head_size=head_size, dtype=dtype)
        if GATE_REASONS:
            raise RuntimeError("gates: " + ", ".join(GATE_REASONS))
        # R10/F: the rotation this path leaves to the glue must have a capture-safe
        # implementation, or the first FULL decode capture dies in FlagGems' generic
        # broadcast multiply (host->device meta copy on the legacy stream).  Refusing
        # here keeps the engine on the deployed bf16 stack instead.
        if not rope_capture_safe:
            raise RuntimeError("the installed RoPE fusion has no capture-safe rotation "
                               "(native rope_only unavailable)")
        if not (mxn is not None and getattr(mxn, "INT8_OK", False)):
            raise RuntimeError(f"native int8 decode unavailable ({getattr(mxn, 'STATUS', '-')})")
        # The prefill reader calls mxa's vendor C entry points, and the whole
        # prefill path has to reach it (there is no paged int8 reader), so the
        # attention policy must be live and self-tested before we commit.
        if not (mxa is not None and getattr(mxa, "ENABLED", False)):
            raise RuntimeError(f"attention policy not live ({getattr(mxa, 'STATUS', '-')})")
        CAP_TOKENS = max(pad16(int(max_model_len)), 32768)
        report = _self_test(device, mxn, mxa, num_heads_q, num_heads_kv, head_size,
                            block_size, max_rows, cfg_dec or mxn.CFG_DEC_I8,
                            cfg_side or mxn.CFG_SIDE_I8)
    except Exception as exc:  # noqa: BLE001
        STATUS = f"off ({type(exc).__name__}: {str(exc)[:300]})"
        return STATUS
    ENABLED = True
    scratch = 2 * CAP_TOKENS * num_heads_kv * head_size * 2 / 2 ** 20
    STATUS = (f"on (INT8 KV, {2 * num_heads_kv * (head_size + 4)} B/token/layer vs "
              f"{2 * num_heads_kv * head_size * 2}, same pages / same num_blocks; gather "
              f"cap {CAP_TOKENS} tokens = {scratch:.0f} MiB scratch; self-test {report})")
    return STATUS

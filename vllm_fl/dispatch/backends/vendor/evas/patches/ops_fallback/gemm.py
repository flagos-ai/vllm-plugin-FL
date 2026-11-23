# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU GEMM correctness patch.

Bare 2-D GEMM (``F.linear``, ``torch.matmul``, ``torch.mm``) is numerically
wrong on LPU -- verified against the CPU reference even for tiny 2x3x4 fp32
inputs -- so every linear layer (via ``default_unquantized_gemm``) produces
garbage and generation is scrambled. This patch routes those names plus
``torch.bmm`` through ``torch.einsum``, which is correct here, skips the
``mul checker`` exit noise of ``torch.bmm``, and avoids the auto-chunk + sync
needed to dodge the firmware die-copy watchdog. Only applied on LPU.

"""

import atexit
import os
from collections import OrderedDict

import torch
import torch.nn.functional as _F

_PATCHED = False
_ORIG_LINEAR = None
_ORIG_MATMUL = None
_ORIG_MM = None
_ORIG_BMM = None

# --- transposed-weight einsum (microbench-verified) --------------------------
# A microbench on the Qwen3-32B shapes measured `mk,kn->mn` with a
# pre-transposed [K,N] weight at ~2x the `mk,nk->mn` path (FFN up M=4: 119 ->
# 62 ms; lm_head: 388 -> 188 ms; FFN up M=512: 127 -> 72 ms), numerically
# identical (allclose, split_dim preserved). Real-model A/B (TP4 MAX_TOKENS=8):
# model_forward 87.6s -> 22.6s (3.88x), execute_model 718s -> 182s, tokens
# agree on the first 4 then diverge on probability-boundary positions due to
# fp32 accumulation-order tails (~5e-4, baseline itself is unstable there).
# This routes the 2D branch of _einsum_linear through the transposed-weight
# form, via RESIDENT contiguous column slabs (_WT_SLAB_CACHE); a cached
# weight.T.contiguous() sliced with live views per call would re-materialize
# every slice on LPU. ON by default; set 0 to restore the pre-transposed
# einsum behaviour.
_EINSUM_TRANSPOSE_WT = os.environ.get("VLLM_FL_EINSUM_TRANSPOSE_WT", "1") == "1"
_WT_T_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
# Pre-extracted contiguous row-slices for huge-N weights (lm_head).  A live
# view ``weight[i:i+chunk]`` of a >1.5 GiB (double-die) weight makes LPU
# materialize the FULL parent storage as the einsum working copy (+~1 GiB
# transient, OOM at decode on a fragmented heap even with 6-9 GiB free;
# seen on DSV4 4-layer TP1).  Caching explicit ``.clone()`` sub-buffers
# (each chunk*K*2 bytes) keeps the per-call transient at that size.  Keyed by
# weight identity only (fixed 512-row chunk, independent of the batch M so the
# prefill-extracted cache is hit at decode).
_LM_HEAD_SUBS_CACHE: dict[
    tuple, tuple[torch.Tensor, list[torch.Tensor]]
] = {}

# --- resident contiguous column slabs for the transposed branch ---------------
# By default the transposed branch must not slice the cached w_t with live
# column views ``w_t[:, i:i+chunk]``: on LPU every column slice of a split_dim=0
# tensor is non-contiguous, so EACH einsum/_fg_mm call re-materializes the slice
# inside the kernel (_to_copy -> double_die_copy_ -> _EVmemcpyBounceD2H host
# bounce -- the dominant term in the gem stacks). Local probe (double-die):
# pre-built contiguous slabs are BITWISE-identical to the views and faster
# end-to-end, while a PER-CALL clone is NOT a win (the clone loop itself costs
# what the internal materialization did), so the slabs must be pre-built and
# RESIDENT.
# A FIXED column width (NOT the M-dependent formula): being M-independent lets
# prefill AND decode share ONE cache entry per weight -- an M-dependent chunk
# would key each weight 2+ times (2x bytes, LRU phase-flip evictions between
# prefill and decode). Same rationale as _LM_HEAD_SUBS_CACHE above.
#
# A single-op microbenchmark cannot size the width: it finds every chunk
# bitwise- and timing-equal because it cannot see the per-launch cost, and on
# LPU every launch drains the device. The width is therefore a launch-count
# knob, not a memory knob: the resident set holds the same total bytes at any
# width, while one logical linear costs ceil(N / width) `flag_gems.mm`
# launches. 4096 is the knee on the engine-level decode path, is
# bitwise-identical to a narrower chunk, and passed the tiny-model sweep, so it
# is the default. Watch the width when debugging engine-wide
# faults: each slab copy is 4x wider than the old 512, and wide die-copies are
# the risk surface the firmware watchdog has reacted to before (the glm-5
# kv_b_proj crash) -- lower the knob to 512 to take that term out.
# Byte-budget LRU mirrors the fp8-linear cache (fp8_methods): the budget
# must cover the WHOLE working set (~1.75 GiB / 236 linears, same bytes as
# the fp8-linear cache) or a cyclic scan hits ~0% and every call rebuilds.
# Watch the atexit stat: evict>0 -> budget too small, miss~calls
# with evict=0 -> keying broken. Builds land on first touch (prefill), same
# as _LM_HEAD_SUBS: no build on the fragmented decode heap in steady state.
# VLLM_FL_EINSUM_SLAB_MB=0 restores the legacy w_t + live-view path (A/B).


def _slab_chunk_from_env(spec: str) -> int:
    """Columns per slab from ``VLLM_FL_EINSUM_SLAB_CHUNK`` (empty = unset = 4096).

    Empty is treated as unset: a shell that expands ``FOO=$BAR`` with ``BAR``
    undefined must not take the import down. Anything else must be a positive
    int -- a silently ignored typo would hide the width the run actually used.
    """
    spec = (spec or "").strip()
    if not spec:
        return 4096
    try:
        value = int(spec)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(
            f"VLLM_FL_EINSUM_SLAB_CHUNK must be a positive int, got {spec!r}"
        )
    return value


_WT_SLAB_CHUNK = _slab_chunk_from_env(os.environ.get("VLLM_FL_EINSUM_SLAB_CHUNK", ""))
_WT_SLAB_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, list[torch.Tensor]]]" = (
    OrderedDict()
)
_WT_SLAB_STAT = {"hit": 0, "miss": 0, "evict": 0}


def _dump_wt_slab_stat() -> None:
    if not (_WT_SLAB_STAT["hit"] or _WT_SLAB_STAT["miss"]):
        return
    try:
        total = sum(
            s.numel() * s.element_size()
            for _, sl in _WT_SLAB_CACHE.values()
            for s in sl
        )
        budget = int(os.environ.get("VLLM_FL_EINSUM_SLAB_MB", "2048")) << 20
        print(
            f"[wt-slab-stat] hit={_WT_SLAB_STAT['hit']} "
            f"miss={_WT_SLAB_STAT['miss']} evict={_WT_SLAB_STAT['evict']} "
            f"entries={len(_WT_SLAB_CACHE)} bytes={total} budget={budget}",
            flush=True,
        )
    except Exception:  # noqa: BLE001 — best-effort debug dump
        pass


atexit.register(_dump_wt_slab_stat)

# Sub-4M weights MUST also use the cached transposed form. A per-call fresh
# weight.detach().T.contiguous() and the "mk,nk" form both crash the engine at
# prefill on glm-5 kv_b_proj [1792,512] (evError 2102 double_die_copy_ +
# firmware "WaitSync Overtime" trap) -- the per-call die-copy
# traffic tips the firmware watchdog. The cached w_t (created once, reused)
# stays under the limit (glm-5 PASS, tokens baseline-identical; traced
# weight_cached=True). MIN therefore defaults to 0: every weight <= MAX uses the
# cached transposed path. The knob stays for a real-DSV4-at-large-M fallback:
# raising it routes sub-4M weights to the UNSAFE fresh-copy branch below (glm-5
# crash).
_TRANSPOSE_MIN_WEIGHT_ELEMS = int(
    os.environ.get("VLLM_FL_EINSUM_TRANSPOSE_MIN_ELEMS", "0")  # 0 elems: cache all
)
# Weights above MAX are not rejected outright: admission is tied to the slab
# budget instead of a bare element count, so a huge-N weight takes the
# transposed path exactly when the memory knob says its copy fits. MAX alone
# dates from the 0.24 4-layer-MTP full-256 run, where the device sat at
# 39.55/43.83 GiB and the fresh [4096, 129280] bf16 w_t (1010 MiB) could not be
# satisfied; that copy is affordable under the slab budget, and taking it is
# what buys the measured decode-step win. Weights that do not fit keep falling
# to the chunked "mk,nk->mn" loop (row slices, no transposed copy). Qwen3-32B
# FFN (117M elems) keeps the ~2x fast path.
_TRANSPOSE_MAX_WEIGHT_ELEMS = int(
    os.environ.get("VLLM_FL_EINSUM_TRANSPOSE_MAX_ELEMS", str(256 << 20))  # 256M elems
)
# Upper bound on the resident transposed copy itself, independent of the budget.
# The ~2x admission model (resident + build transient) presumes the copy is
# materializable as a whole; past 1.5 GiB the LPU materializes the FULL parent
# storage on ANY view sub-read of it, so a budget raised past 3 GiB would admit
# a copy whose slabs each drag the whole thing in -- the OOM the budget exists
# to prevent.  Same threshold _LM_HEAD_SUBS_CACHE above is built around.
_TRANSPOSE_MAX_COPY_BYTES = 1536 << 20  # 1.5 GiB


def _transpose_admitted(weight: torch.Tensor, slab_mb: int) -> bool:
    """May ``weight`` take the transposed path?

    Below MAX it always may, unchanged. Above MAX the copy has to fit the slab
    budget: the resident transposed form equals the weight bytes, and the build
    transient peaks at ~2x those bytes, so both must sit inside ``slab_mb``.
    DSV4 lm_head [129280, 4096] bf16 is 1010 MiB resident / 2020 MiB transient,
    so it qualifies at the default 2048 MiB budget (slab cache 26 -> 27
    entries, evict=0). ``slab_mb`` of 0 disables the slab path entirely and
    restores the MAX-only rule.

    The above-MAX arm additionally requires the weight to be one the resident
    path will cache: under a raised VLLM_FL_EINSUM_TRANSPOSE_MIN_ELEMS it would
    otherwise reach the per-call fresh w_t branch (the glm-5 crash class), so
    those weights keep the chunked route they have today.

    The ~2x model holds only while the copy stays materializable as a whole
    (``_TRANSPOSE_MAX_COPY_BYTES``): past 1.5 GiB the LPU materializes the FULL
    parent storage on any view sub-read, so a budget raised enough to admit
    such a weight would spike far past 2x.  The default 2048 MiB budget cannot
    reach that -- the arm caps the copy at slab_mb/2 = 1024 MiB -- so this
    check only binds once the budget is raised past 3072 MiB.
    """
    if weight.numel() <= _TRANSPOSE_MAX_WEIGHT_ELEMS:
        return True
    if slab_mb <= 0:
        return False
    if weight.numel() < _TRANSPOSE_MIN_WEIGHT_ELEMS:
        return False
    weight_bytes = weight.numel() * weight.element_size()
    if weight_bytes > _TRANSPOSE_MAX_COPY_BYTES:
        return False
    return 2 * weight_bytes <= slab_mb << 20


def _safe_split_dim(t: torch.Tensor):
    """split_dim() that never raises (CPU / non-double-die tensors)."""
    try:
        return t.split_dim()
    except Exception:  # noqa: BLE001
        return None


def _trace_nk_operands(tag: str, input, weight, chunk: int, cached: bool):
    """Env-gated (VLLM_FL_EINSUM_TRACE=1) operand dump for einsum debugging.

    glm-5's device crash (evError 2102 double_die_copy_) was NOT triggered by
    the einsum *form*: both "mk,nk->mn" and a per-call fresh transposed w_t
    crashed at prefill (M=4) while the cached w_t passed -- the trigger was the
    per-call transpose die-copy tipping the firmware watchdog. This dumps what
    the einsum consumed at runtime (split_dim, view/shared-storage, chunk size)
    to pin any future trigger.
    """
    def dbg(t):
        base = getattr(t, "_base", None)
        shared = base is not None and base.data_ptr() == t.data_ptr()
        return (f"shape={tuple(t.shape)} numel={t.numel()} split_dim="
                f"{_safe_split_dim(t)} dtype={t.dtype} "
                f"view={base is not None} shared_storage={shared} "
                f"contig={t.is_contiguous()}")
    print(f"[gemm-trace] {tag}: chunk={chunk} weight_cached={cached}\n"
          f"  input  {dbg(input)}\n"
          f"  weight {dbg(weight)}", flush=True)


def _wt_slabs(weight: torch.Tensor, budget_mb: int) -> list[torch.Tensor]:
    """Resident contiguous ``[K, <=chunk]`` column slabs of ``weight.T``.

    Hit: LRU-touch and serve the cached slabs. Miss: build ``w_t`` once,
    clone its column chunks (each a contiguous copy, so einsum never pays
    the split_dim-0 view materialization per call), insert under the weight
    identity (strong ref on ``weight`` keeps data_ptr keys stable), then
    evict LRU entries until total slab bytes fit the budget (always keep
    >=1 entry, mirroring the fp8-linear cache). Memory: the build transient
    peaks at ~2x weight bytes (``w_t`` plus the slab copies accumulating
    while the local ``w_t`` is still alive) and lands on first touch
    (prefill/profile); after return only the slab copies are reachable, so
    steady-state memory is exactly the slab bytes (== one transposed copy).
    """
    key = (weight.data_ptr(), weight.shape, weight.stride())
    hit = _WT_SLAB_CACHE.get(key)
    if hit is not None:
        _WT_SLAB_CACHE.move_to_end(key)
        _WT_SLAB_STAT["hit"] += 1
        return hit[1]
    n = weight.shape[0]
    # Transient: w_t (1x weight bytes) stays alive while the slab copies
    # accumulate -> build peak ~2x weight bytes, freed once this returns.
    w_t = weight.T.contiguous()
    slabs = [
        w_t[:, i : i + _WT_SLAB_CHUNK].contiguous()
        for i in range(0, n, _WT_SLAB_CHUNK)
    ]
    _WT_SLAB_CACHE[key] = (weight, slabs)
    _WT_SLAB_CACHE.move_to_end(key)
    _WT_SLAB_STAT["miss"] += 1
    budget = budget_mb << 20
    while len(_WT_SLAB_CACHE) > 1:
        total = sum(
            s.numel() * s.element_size()
            for _, sl in _WT_SLAB_CACHE.values()
            for s in sl
        )
        if total <= budget:
            break
        _WT_SLAB_CACHE.popitem(last=False)
        _WT_SLAB_STAT["evict"] += 1
    return slabs


# --- flag_gems.mm as the 2D GEMM source of truth (lazy, fail-safe) -----------
# flag_gems' evas override for torch.mm (flag_gems.runtime.backend._evas.ops.mm)
# implements the same LPU-correct 2D einsum ("mk,kn->mn") that this file uses,
# verified bitwise-identical to _einsum_matmul on the Qwen3-32B shapes (probe:
# maxdiff=0 on all 4 shapes). Sinking our 2D einsum calls into
# flag_gems.mm makes flaggems the single source of truth: when upstream lands
# a faster mm kernel in _evas/ops/mm.py, this file picks it up with ZERO
# changes. Lazy-imported on first call (flag_gems' __init__ toggles double-die
# from ENABLE_DOUBLE_DIE, so it must not run at module import time — it runs
# once the env is final, exactly like the existing vit.py / dsv4_compressor
# lazy flag_gems imports). Any import/API failure falls back to the local
# einsum permanently (_FG_MM=False) — a flaggems change can never break LPU
# inference.
#
# Only the 2D paths go through flag_gems.mm: _einsum_matmul/_einsum_mm
# (B is already [K,N]) and _einsum_linear's transposed-weight branch (w_t is
# [K,N]). _einsum_bmm stays on local einsum — flag_gems.mm's non-2D fallback
# is bare torch.matmul, which is WRONG on LPU. The non-transposed
# _einsum_linear branch also stays local (its weight slice is [N,K], and
# transposing per call would cost more than it saves).
_FG_MM = None  # None=untried, False=unavailable/failed, else the callable


def _fg_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[M,K] @ [K,N] via flag_gems.mm when available, else local einsum."""
    global _FG_MM
    if _FG_MM is None:
        try:
            from flag_gems import mm as _fg  # evas override = einsum today
            _FG_MM = _fg
        except Exception as exc:  # noqa: BLE001 — fail-safe, never breaks infer
            print(f"[gemm] flag_gems.mm import failed, using einsum fallback: "
                  f"{exc}", flush=True)
            _FG_MM = False
    if _FG_MM is False:
        return torch.einsum("mk,kn->mn", a, b)
    try:
        return _FG_MM(a, b)
    except Exception as exc:  # noqa: BLE001 — fail-safe, never breaks infer
        print(f"[gemm] flag_gems.mm call failed ({exc}), using einsum fallback",
              flush=True)
        _FG_MM = False
        return torch.einsum("mk,kn->mn", a, b)


def _einsum_linear(input: torch.Tensor, weight: torch.Tensor, bias=None):
    """Drop-in replacement for torch.nn.functional.linear on LPU.

    Computes ``input @ weight.T (+ bias)`` via torch.einsum, which is a
    numerically-correct matmul path on LPU. Handles 2D, batched (N-D input +
    2D weight), and per-batch-weight (N-D input + N-D weight) cases so the
    fallback never lands on LPU's broken native mm/addmm (which returns NaN or
    raises ``addmm expected self and other matched shape``).

    The 2D path chunks the output dimension to avoid OOM on large weight
    matrices (e.g. lm_head: vocab=152064, K=5120 → 3.1 GiB intermediate).
    Einsum on LPU is numerically correct in all supported dtypes (bf16, fp16,
    fp32), so the intermediate is kept in the input dtype — no unnecessary
    fp32 casts that would double memory bandwidth for bf16 inference.

    When a TorchFunctionMode is active (e.g. io_dumper), delegate to the
    original F.linear so the mode can intercept it.
    """
    if torch._C._is_torch_function_mode_enabled():
        return _ORIG_LINEAR(input, weight, bias)
    if input.dim() == 2 and weight.dim() == 2:
        # [M, K] @ [N, K]^T → [M, N].  Chunk N to bound intermediate memory.
        M, K = input.shape
        N = weight.shape[0]
        # Transposed-weight form (VLLM_FL_EINSUM_TRANSPOSE_WT=1): einsum
        # "mk,kn->mn" on a weight.T.contiguous() [K,N] measures ~2x faster than
        # "mk,nk->mn" on the Qwen3-32B shapes (probe), numerically
        # identical. Chunking still applies to bound intermediate memory (slice
        # columns of the transposed weight). Admission is `_transpose_admitted`:
        # a weight enters when it is below MAX, or when its transposed copy both
        # fits the slab budget (VLLM_FL_EINSUM_SLAB_MB) and stays under the
        # materialization limit. Below MAX all weights (incl.
        # sub-4M) use a resident transposed form (contiguous slabs by default,
        # the legacy cached w_t when VLLM_FL_EINSUM_SLAB_MB=0; see the MIN
        # comment above for why sub-4M MUST cache).
        slab_mb = int(os.environ.get("VLLM_FL_EINSUM_SLAB_MB", "2048"))
        if _EINSUM_TRANSPOSE_WT and _transpose_admitted(weight, slab_mb):
            # Sub-4M weights use the CACHED transposed form (cacheable below):
            # both "mk,nk->mn" and a per-call FRESH w_t crash the engine at
            # prefill (glm-5 kv_b_proj [1792,512], evError 2102 double_die_copy_
            # + firmware "WaitSync Overtime" trap; traced for both forms). A
            # transposed form created ONCE and reused passes with
            # baseline-identical tokens -- the per-call transpose copy adds
            # enough die-copy traffic to tip the firmware watchdog. The slab
            # path (default) creates its slabs once per weight; the legacy
            # path's cached w_t does the same.
            cacheable = weight.numel() >= _TRANSPOSE_MIN_WEIGHT_ELEMS
            if cacheable and slab_mb > 0:
                # Resident contiguous slabs (rationale at _WT_SLAB_CACHE):
                # never feed einsum a live column view of w_t -- every column
                # slice of a split_dim=0 tensor is non-contiguous and would
                # be re-materialized inside each call (_EVmemcpyBounceD2H
                # host bounce).
                slabs = _wt_slabs(weight, slab_mb)
                if os.environ.get("VLLM_FL_EINSUM_TRACE", "0") == "1":
                    _trace_nk_operands(f"mk,kn slabs(n={len(slabs)})",
                                       input, slabs[0], _WT_SLAB_CHUNK, True)
                out_chunks = []
                if hasattr(torch, "lpu") and hasattr(torch.lpu, "synchronize"):
                    torch.lpu.synchronize()
                for j, w_chunk in enumerate(slabs):
                    out_chunk = _fg_mm(input, w_chunk)
                    out_chunks.append(out_chunk)
                    if j & 15 == 15 and hasattr(torch, "lpu") and hasattr(torch.lpu, "synchronize"):
                        torch.lpu.synchronize()
                out = torch.cat(out_chunks, dim=1)  # [M, N]
                if bias is not None:
                    out = out + bias
                return out
            # Legacy view path -- VLLM_FL_EINSUM_SLAB_MB=0 (A/B back to the
            # pre-slab behaviour) or a MIN-gated non-cacheable weight.
            # The live views w_t[:, i:i+chunk] are non-contiguous column
            # slices: every einsum re-materializes them (the cost the slab
            # path removes; local probe: o_proj-ish M=4 20.2 ms views vs
            # 10.5 ms pre-built slabs).
            if cacheable:
                key = (weight.data_ptr(), weight.shape, weight.stride())
                hit = _WT_T_CACHE.get(key)
                if hit is not None:
                    w_t = hit[1]
                else:
                    w_t = weight.T.contiguous()  # [K, N]
                    if len(_WT_T_CACHE) > 64:
                        _WT_T_CACHE.clear()
                    _WT_T_CACHE[key] = (weight, w_t)
            else:
                # UNSAFE (glm-5 crash): a per-call fresh w_t trips
                # the firmware watchdog at prefill. Reachable only via a high
                # VLLM_FL_EINSUM_TRANSPOSE_MIN_ELEMS; kept for A-B testing.
                w_t = weight.detach().T.contiguous()  # [K, N] fresh, no cache
            if os.environ.get("VLLM_FL_EINSUM_TRACE", "0") == "1":
                _trace_nk_operands("mk,kn transposed", input, w_t,
                                   max(512, 8_388_608 // (M * K)), cacheable)
            max_elements = 8_388_608  # 32 MiB at 4 bytes/elem
            chunk = max(512, max_elements // (M * K))
            out_chunks = []
            if hasattr(torch, "lpu") and hasattr(torch.lpu, "synchronize"):
                torch.lpu.synchronize()
            for i in range(0, N, chunk):
                # w_t[:, i:i+chunk] is [K, chunk] (column slice, no copy needed
                # for the K-major layout — but einsum reads it fine)
                w_chunk = w_t[:, i : i + chunk]
                out_chunk = _fg_mm(input, w_chunk)
                out_chunks.append(out_chunk)
                if (i // chunk) & 15 == 15 and hasattr(torch, "lpu") and hasattr(torch.lpu, "synchronize"):
                    torch.lpu.synchronize()
            out = torch.cat(out_chunks, dim=1)  # [M, N]
            if bias is not None:
                out = out + bias
            return out
        # Weights that are neither below MAX nor affordable under the slab
        # budget skip the transposed form: the w_t = weight.T.contiguous() copy
        # would OOM on the 0.24 4-layer-MTP full-256 device (18a16e5), which is
        # where the MAX-only rule came from. DSV4 lm_head (1010 MiB) now clears
        # the budget rule; anything larger still falls here, to the chunked
        # "mk,nk->mn" loop.  The row slices MUST be
        # pre-extracted contiguous sub-buffers, not live views: on LPU a view
        # ``weight[i:i+chunk]`` of a >1.5 GiB (double-die) weight materializes
        # the FULL parent storage as the einsum working copy (~+1 GiB transient,
        # OOM at decode on a fragmented heap even with 6-9 GiB free; seen on
        # DSV4 4-layer TP1; lm_head [129280, 4096] bf16 = 1010 MiB).  NOTE: this
        # form is UNSAFE for sub-4M double-die weights (the glm-5 crash above)
        # -- the transposed branch above (MIN defaults to 0) keeps them off this
        # path; do NOT raise _TRANSPOSE_MIN_WEIGHT_ELEMS to route them here.
        # Sub-buffer size is FIXED (512 rows) and keyed by weight identity only
        # (not by batch M): the lazy extraction still materializes the parent
        # once (~+1010 MiB transient), so it must land on the FIRST call -- the
        # prefill step, before the KV pool / workspace push decode's free memory
        # down.  A batch-dependent chunk would re-key on decode and re-extract
        # on the fragmented heap.  The einsum per 512-row chunk has a small
        # transient (~M*512*K*2 bytes; ~20 MiB at M=5), and the fixed chunk is
        # the pre-existing min from the firmware-watchdog floor.
        chunk = 512
        key = (weight.data_ptr(), weight.shape, weight.stride())
        entry = _LM_HEAD_SUBS_CACHE.get(key)
        if entry is None:
            # Pre-extract once: each explicit .clone() is chunk*K*2 bytes.  The
            # transient parent materialization is bounded to this one extraction.
            subs = [weight[i : i + chunk].clone() for i in range(0, N, chunk)]
            if len(_LM_HEAD_SUBS_CACHE) > 8:
                _LM_HEAD_SUBS_CACHE.clear()
            # Hold a strong ref on weight (mirrors _WT_SLAB_CACHE /
            # _WT_T_CACHE): the key is the data_ptr, so letting the weight be
            # freed would allow an ABA reuse of the same address for a
            # different tensor of identical shape/stride -> silent wrong chunks.
            _LM_HEAD_SUBS_CACHE[key] = (weight, subs)
        else:
            subs = entry[1]
        out_chunks = []
        # Pre-loop sync: reset the LPU firmware die-copy watchdog before
        # the chunked einsum loop. Without this, accumulated kernel time
        # across many chunks triggers "WaitSync Overtime" on the final cat.
        # Same pattern as flaggems evas mm.py auto-chunk.
        if hasattr(torch, "lpu") and hasattr(torch.lpu, "synchronize"):
            torch.lpu.synchronize()
        for j, w_chunk in enumerate(subs):
            out_chunk = torch.einsum("mk,nk->mn", input, w_chunk)
            out_chunks.append(out_chunk)
            # Periodic sync: reset the watchdog timer every 16 chunks so
            # accumulated kernel time doesn't exceed the firmware limit.
            if j & 15 == 15 and hasattr(torch, "lpu") and hasattr(torch.lpu, "synchronize"):
                torch.lpu.synchronize()
        out = torch.cat(out_chunks, dim=1)  # [M, N]
        if bias is not None:
            out = out + bias
        return out
    if input.dim() >= 3 and weight.dim() == 2:
        # Batched linear with a shared weight: [*, M, K] @ [N, K] -> [*, M, N].
        # Flatten leading dims to a single batch, broadcast the (K, N) weight
        # across it, and run one einsum.
        n = weight.shape[0]
        m, k = input.shape[-2], input.shape[-1]
        b = input.numel() // (m * k)
        inp = input.reshape(b, m, k)
        # bmk, nk -> bmn
        out = torch.einsum("bmk,nk->bmn", inp, weight)
        out = out.reshape(*input.shape[:-2], m, n)
        if bias is not None:
            out = out + bias
        return out
    if input.dim() == weight.dim() and input.dim() >= 3:
        # Per-batch weights: [B, M, K] @ [B, N, K] -> [B, M, N].
        orig_shape = input.shape
        m, k = orig_shape[-2], orig_shape[-1]
        b = input.numel() // (m * k)
        inp = input.reshape(b, m, k)
        wt = weight.reshape(b, weight.shape[-2], weight.shape[-1])
        # bmk, bnk -> bmn
        out = torch.einsum("bmk,bnk->bmn", inp, wt)
        out = out.reshape(*orig_shape[:-2], m, weight.shape[-2])
        if bias is not None:
            out = out + bias
        return out
    # Truly exotic shapes (e.g. mismatched batched dims) — preserve shape
    # semantics via the original implementation rather than guessing. This may
    # be wrong on LPU, but such cases do not occur in the model forward.
    return _ORIG_LINEAR(input, weight, bias)


def _einsum_matmul(*args, **kwargs):
    """Route torch.matmul through einsum for 2D x 2D on LPU.

    When a TorchFunctionMode is active (e.g. io_dumper), delegate to the
    original torch.matmul so the mode can intercept it.  The patched
    inline function would otherwise bypass the __torch_function__ protocol.
    """
    if torch._C._is_torch_function_mode_enabled():
        return _ORIG_MATMUL(*args, **kwargs)
    if (
        len(args) == 2
        and isinstance(args[0], torch.Tensor)
        and isinstance(args[1], torch.Tensor)
        and args[0].dim() == 2
        and args[1].dim() == 2
    ):
        # B is [K,N] — flag_gems.mm expects exactly that layout (einsum
        # "mk,kn->mn"), same as the local einsum it replaces.
        a, b = args
        return _fg_mm(a, b)
    return _ORIG_MATMUL(*args, **kwargs)


def _einsum_mm(input: torch.Tensor, mat2: torch.Tensor, *, out=None, out_dtype=None):
    """Route torch.mm through einsum on LPU.

    ``out_dtype`` (torch 2.11 mm API, e.g. the DSV4 compressor's
    ``torch.mm(..., out_dtype=torch.float32)``) is honored by casting the
    einsum result — the flaggems evas mm already accumulates fp32 internally
    for bf16 inputs, so the einsum-then-cast contract holds for the
    compression scores.
    """
    if torch._C._is_torch_function_mode_enabled():
        return _ORIG_MM(input, mat2, out=out, out_dtype=out_dtype)
    if input.dim() == 2 and mat2.dim() == 2:
        res = _fg_mm(input, mat2)  # mat2 is [K,N] (torch.mm layout)
        if out_dtype is not None:
            res = res.to(out_dtype)
        if out is not None:
            out.copy_(res)
            return out
        return res
    return _ORIG_MM(input, mat2, out=out, out_dtype=out_dtype)


def _einsum_bmm(input: torch.Tensor, mat2: torch.Tensor, *, out=None):
    """Route torch.bmm through einsum for 3D x 3D on LPU.

    torch.bmm is not called on the evas hot path (all GEMM goes through
    F.linear/matmul/mm which are already patched).  This patch exists as a
    safety net: any external code or future path that calls torch.bmm on LPU
    would hit the ~80x slow LPU native kernel or crash via the global FlagGems
    Triton bmm kernel (which fails compilation on evofc-opt with memref.store
    to MM).  Patching it to einsum ensures correctness.
    """
    if torch._C._is_torch_function_mode_enabled():
        return _ORIG_BMM(input, mat2, out=out)
    if input.dim() == 3 and mat2.dim() == 3:
        res = torch.einsum("bik,bkj->bij", input, mat2)
        if out is not None:
            out.copy_(res)
            return out
        return res
    return _ORIG_BMM(input, mat2, out=out)


# --- debug GEMM shape profiler (env-gated, off by default) -----------------
# Placed AFTER every _einsum_* definition (so it can rebind all four names) and
# BEFORE apply_lpu_gemm_patch() (so the patch reads the WRAPPED versions and
# F.linear/torch.matmul/torch.mm/torch.bmm all route through the profiler).
#
# VLLM_FL_GEMM_PROFILE=1 wraps the 4 einsum GEMM entries to accumulate
# per-shape (M,N,K) call count + wall time in THIS process, and dumps to
# VLLM_FL_GEMM_PROFILE_OUT/gemm-<pid>.json at exit. Env-gated so the wrap is
# installed at import time in every process that imports gemm.py — including
# the vLLM EngineCore/worker processes that actually run the model forward
# (which is why a probe-side wrap in the main process never sees the GEMMs).
# Qwen3's GEMMs reach _einsum_matmul (torch.matmul) as well as _einsum_linear
# (F.linear), so ALL four entries are wrapped. Stats are only collected under
# VLLM_FL_GEMM_PROFILE=1. Zero
# behavior change when off.
_GEMM_PROFILE = os.environ.get("VLLM_FL_GEMM_PROFILE", "0") == "1"
_gemm_stats: dict[tuple[int, int, int], list] = {}

if _GEMM_PROFILE:
    import atexit
    import json as _json
    import time as _time

    print(f"[gemm-profile] enabled pid={os.getpid()} "
          f"out={os.environ.get('VLLM_FL_GEMM_PROFILE_OUT', '/tmp')}",
          flush=True)

    def _gemm_profile_wrapped(fn, input, weight, *args, **kwargs):
        _t0 = _time.perf_counter()
        out = fn(input, weight, *args, **kwargs)
        _dt = _time.perf_counter() - _t0
        try:
            _m, _k = input.shape
            _n = weight.shape[0]
            _key = (_m, _n, _k)
            _st = _gemm_stats.get(_key)
            if _st is None:
                _st = [0, 0.0]
                _gemm_stats[_key] = _st
            _st[0] += 1
            _st[1] += _dt
        except Exception:  # noqa: BLE001 — profiling must never break forward
            pass
        return out

    def _gemm_profile_dump():
        try:
            _outdir = os.environ.get("VLLM_FL_GEMM_PROFILE_OUT", "/tmp")
            os.makedirs(_outdir, exist_ok=True)
            _path = os.path.join(_outdir, f"gemm-{os.getpid()}.json")
            _data = {
                "pid": os.getpid(),
                "shapes": [
                    {"M": _k[0], "N": _k[1], "K": _k[2],
                     "calls": _v[0], "total_s": round(_v[1], 4)}
                    for _k, _v in _gemm_stats.items()
                ],
            }
            with open(_path, "w") as _f:
                _json.dump(_data, _f, indent=2)
        except Exception:  # noqa: BLE001 — best-effort debug dump
            pass

    atexit.register(_gemm_profile_dump)

    # Rebinding AFTER the originals are defined: _gemm_*_orig capture the real
    # einsum implementations; the module-level _einsum_* names become wrapped
    # versions that apply_lpu_gemm_patch() will install into F.linear etc.
    _gemm_linear_orig = _einsum_linear
    _gemm_matmul_orig = _einsum_matmul
    _gemm_mm_orig = _einsum_mm
    _gemm_bmm_orig = _einsum_bmm

    def _einsum_linear(input, weight, bias=None):  # noqa: A001 — profiler wrap
        return _gemm_profile_wrapped(_gemm_linear_orig, input, weight, bias)

    def _einsum_matmul(*args, **kwargs):
        if (len(args) == 2 and isinstance(args[0], torch.Tensor)
                and isinstance(args[1], torch.Tensor)
                and args[0].dim() == 2 and args[1].dim() == 2):
            return _gemm_profile_wrapped(_gemm_matmul_orig, args[0], args[1])
        return _gemm_matmul_orig(*args, **kwargs)

    def _einsum_mm(input, mat2, *, out=None, out_dtype=None):
        # out_dtype must be forwarded: torch 2.11 mm callers pass it
        # (DSV4 compressor: torch.mm(..., out_dtype=torch.float32)); dropping it
        # made every VLLM_FL_GEMM_PROFILE=1 run TypeError on the first
        # compressor GEMM.
        return _gemm_profile_wrapped(
            _gemm_mm_orig, input, mat2, out=out, out_dtype=out_dtype
        )

    def _einsum_bmm(input, mat2, *, out=None):
        return _gemm_profile_wrapped(_gemm_bmm_orig, input, mat2, out=out)


def apply_lpu_gemm_patch() -> None:
    """Replace LPU's broken 2D GEMM with the einsum path.

    Patches F.linear, torch.matmul, torch.mm, and torch.bmm.  The first three
    are on the model hot path; torch.bmm is a safety net for external callers.
    """
    global _PATCHED, _ORIG_LINEAR, _ORIG_MATMUL, _ORIG_MM, _ORIG_BMM
    if _PATCHED:
        return
    _ORIG_LINEAR = _F.linear
    _ORIG_MATMUL = torch.matmul
    _ORIG_MM = torch.mm
    _ORIG_BMM = torch.bmm
    _F.linear = _einsum_linear
    torch.matmul = _einsum_matmul
    torch.mm = _einsum_mm
    torch.bmm = _einsum_bmm
    _PATCHED = True

# SPDX-License-Identifier: Apache-2.0
# 2026 - FlagOS S2 contest work on top of the MetaX Maca attention backend.
"""Own paged decode attention for the MetaX C500 in native MACA C++ (R6/N1 + R7/N4).

Serves the q_len-1 decode rows of the MetaX attention backend (../flash_attn.py)
through a MACA C++ kernel of our own design (csrc/n1_dec_attn.cuh) on the
16x16x16 bf16 MMA unit:

* K tile -> S^T = K . Q^T (8 MMAs per 16 tokens); the result lands in exactly the
  lanes and layout the P . V MMA needs as its P operand, and the V operand is a
  16-bit transpose of the lane's own four 128-bit V loads (no shared memory);
* fp32 online softmax (base 2), P rounded to bf16 once and summed from the same
  rounded values;
* two schedules, both rebuilt INSIDE the kernel from seq_lens in device memory:
  - SCHED 0 (R6/N1, cfg 13): a persistent grid claims (row, kv head, chunk) items
    with an atomic counter -> CTAs that start late take what is left, which is
    what the decode rows of an overlapped mixed step need (they run on the side
    stream next to the prefill kernel);
  - SCHED 1 (R7/N4, cfg 20): a static stream-K partition of tile space, equal to
    +-1 tile per CTA, no atomic claiming, <= 2 fp32 partials per CTA -> no claim
    granularity and no tail wave, which is what the decode-only steps need (they
    have the whole GPU and run inside the FULL CUDA graphs);
* split segments are merged by the CTA that arrives last on them (atomic counter,
  partials read in ascending tile order -> bit-identical output for the same
  seq_lens), no second launch and no host-side zeroing.

CUDA-graph replay safety (R7/N4; the decode-only steps are captured in FULL
graphs and replayed for thousands of steps while the contexts grow):

* the launch configuration is constant for a captured batch size: grid =
  CTAS_PER_AP * APs always, block = W * 64, no dynamic shared memory, workspace
  pointers from one pre-allocated Workspace per call site;
* every quantity that changes between replays is read from device memory by the
  kernel: the per-row context (p.seq = decode_seq_lens, a slice of the runner's
  persistent seq_lens buffer, padded rows zero), the page ids (block_table, also
  persistent), and from those the tile count, the partition, the number of
  partials per segment and which work does not exist in this replay;
* the only mutable device state is the counter array, zeroed once at setup and
  left zero by every launch (SCHED 1 only touches ctr[2 + segment], which the
  merging CTA resets); nothing is zeroed from the host between replays and
  nothing is allocated at replay;
* no .item() / .cpu() / .tolist() / nonzero / stream sync anywhere on the path:
  the launch only reads shapes, strides and data_ptr()s;
* the host-side selection uses only features that are constant for a captured
  size (row count, head counts, head size, page size, cache layout, AP count) --
  never the context length, which is unknown at capture time.

The library is built from csrc/ with mxcc (`python3 mx_dec_native.py --build`,
part of the plugin's MetaX build step; setup() builds it on first use when no
prebuilt copy for the current sources is found) and called through ctypes.

Where it serves:

* decode rows next to a prefill kernel (an overlapped mixed step, P1 side
  stream): SCHED 0 / cfg 13 (R6/N1 GPU test, work/R6/N1-mx-dec-attn-native:
  e.g. 36 rows @ 16k next to a 2012-token chunk at k 12072 / 16096: 1852-1913 /
  2444-2457 us per layer vs 2266-2290 / 2917-2946 for the vendor kernel on the
  side stream).  Unchanged by N4;
* decode-only steps (FULL decode graphs, eager decode-only, serial mixed steps):
  SCHED 1 / cfg 20, gated by DEC_MIN_ROWS (and, where the host knows it without
  a sync, the context bucket).  R6/N1 measured SCHED 0 at x0.95 (4k) / x1.00
  (16k) of the vendor kernel there, 78% / 87% of the paged-read floor; the
  stream-K schedule removed that scheduling slack where the rows and the context
  are large: measured in-graph against the vendor call, 64 rows x1.008 at a 4k
  context (84% of the paged-read floor, the vendor 84% as well -- at 4k the gap
  to the floor is the access pattern, not the schedule) and x1.056 at 16k (93%
  of the floor against the vendor's 88%).  Hence the row and context gates
  below: the native path is taken only where it is measured at or above the
  vendor kernel.

The vendor call (mx_prefill_attn.decode_attention = flash_attn fwd_kvcache)
stays the fallback: VLLM_FL_MX_DEC_NATIVE=0, shapes/layouts the library does
not cover (G != 8, head size != 128, rows > 512, non-bf16 cache, ...), a row
count or context outside the policy, or a failed build / start-up self-test
(which includes a capture + replay of the kernel in a CUDA graph).

Compliance: a new kernel of our own design inside vllm-plugin-FL (kernel
rewrite); selection keys on runtime features only (step composition, row count,
dtype / layout); serve arguments, cudagraph mode and cudagraph sizes are
untouched.  Nothing here touches the GPU at import time.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import subprocess
import sys
import time

import torch

# ---------------------------------------------------------------------------
# switches and launch policy (work/R6/N1-mx-dec-attn-native/bench_n1.py)
# ---------------------------------------------------------------------------
ENV_FLAG = "VLLM_FL_MX_DEC_NATIVE"
MODE = os.environ.get(ENV_FLAG, "1").strip().lower()
REQUESTED = MODE not in ("0", "off", "false", "no", "vendor")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# --- mixed-step site (R6/N1, unchanged): atomic-claimed items on the side stream ---
# best of the R6 GPU tuning (stage A+B, two seeds); claim lead 0-2 tie within 1 us.
CFG_MIXED = 13           # W=4 warps/CTA, 1 tile in flight/warp, coalesced K loads turned
                         # into MMA fragments by an identity MMA (KL=1), SCHED 0;
                         # 242 registers, no spill, 2 CTAs/AP
ITEMS_PER_CTA = 2        # target items per CTA (sets the chunk length on the device)
MIN_CHUNK_TILES = 16     # items cover >= 16 tiles (256 tokens) of a row
CLAIM_LEAD = 2           # warp 0 claims the next item 2 tiles before it runs out
MIXED_MIN_ROWS = 1
MIXED_MAX_ROWS = 512     # library limit (N1_MAXR)

# --- decode-only site (R7/N4): static stream-K, used inside the FULL graphs too ---
# cfg 20 = W 4 warps/CTA, no K prefetch (PF 0), identity-MMA K loads (KL 1),
# SCHED 1; 216 registers, no spill, 2 CTAs/AP.  Chosen over the staged cfg 18
# (246 reg) and over cfg 23 (W 2 / PF 1, 230 reg) on the C500 GPU test
# (work/R7/N4-mx-graph-dec-attn/gpu_results/n4_full_20261025_015235.json,
# in-graph, A/B/A/B vs the deployed vendor call, us per layer):
#     rows  4k ctx          16k ctx
#       64  241.1 x1.008    742.4 x1.056
#       48  187.3 x1.000    (32 rows: 394.9 x1.043)
#       32  136.5 x0.973
# cfg 23 is 5 us/layer better at 64 rows / 16k (737.0, x1.064) but x0.979 at
# 48 rows / 4k and x0.918 at 32 rows, so cfg 20 is the default and cfg 23 the
# opt-in (VLLM_FL_MX_DEC_CFG=23) for a 16k-heavy mix.
CFG_DEC = _env_int("VLLM_FL_MX_DEC_CFG", 20)   # same tile code, SCHED 1
# Row gate.  The GPU test put the crossover against the vendor kernel at ~48
# decode rows, not at the 8 the schedule analysis suggested: at 48+ rows cfg 20
# is >= x1.000 at a 4k context and >= x1.04 at 16k, while 32 rows is x0.973 at
# 4k (and a captured graph cannot be re-decided per context -- see
# wanted_decode -- so a bucket that loses at 4k has to stay with the vendor
# call).  Below 48 rows the vendor split-kv heuristic keeps the step; those
# steps are also cheap in absolute terms (32 rows read half the bytes of 64).
DEC_MIN_ROWS = _env_int("VLLM_FL_MX_DEC_MIN_ROWS", 48)
DEC_MAX_ROWS = 512
# Context gate, only applied where the host knows the context without a sync, i.e.
# NOT under CUDA-graph capture (at capture the contexts of the later replays do
# not exist yet).  Eager decode-only steps pay the launch cost that a replay does
# not, and measured eagerly cfg 20 is x0.991 at a 4k context and x0.982 at 8k but
# x1.047 at 16k (same json, "eager" rows), so the eager path is kept to long
# contexts only.  The win is a context effect (ctx sweep at 64 rows, in-graph:
# 512 x0.840, 1k x0.915, 2k x0.964, 8k x1.011, 16k x1.056).
DEC_MIN_CTX = _env_int("VLLM_FL_MX_DEC_MIN_CTX", 16384)

# --- R8/Q1: the same kernel over the SELF-DEVELOPED INT8 KV cache (Q8 = 1 configs) ---
# ops/mx_kv_int8.py owns the layout, the quantising write and the prefill gather; here
# only the launch.  Default off: the configs are reached only when that module enabled
# itself (which needs VLLM_FL_MX_INT8_KV=1 and its own self-test).
#
# The two configs below are the C500 GPU test's answer, not the staged guess: the
# designer's defaults (29 decode / 39 side) each carry a register-spill stack frame
# (32 B / 36 B, mxcc -resource-usage over all 43 instantiations) and lose 2.4% / 4.2%
# at 4k, which is the R7/N5 failure mode.  Spill-free int8 math configs are 28 (238
# reg, 2 CTAs/AP), 33 (246, 3) and 38 (236, 2); of those, measured interleaved
# A/B/A/B under graph replay against the deployed stack (us/layer, 42 layers):
#
#   site / shape          vendor bf16   native bf16   cfg 28   cfg 38   int8 floor
#   decode-only 64 @ 4k      242.5         243.9      209.2    223.8      111.7
#   decode-only 64 @ 16k     783.9         729.6      629.5    638.4      368.0
#
# so cfg 28 (SCHED 1 stream-K, W 4, PF 1, LDK 0) for the decode-only site, which runs
# alone inside the FULL graphs, and cfg 38 (SCHED 0 atomic-claimed, W 4, PF 1) for the
# side stream of an overlapped mixed step, where CTAs must take the APs as the prefill
# kernel frees them.  PF > 1 and W = 8 spill on every int8 instantiation and all are
# slower; W = 2 (cfg 33) is spill-free at 3 CTAs/AP but 15.7% / 18.3% slower.
# R9/R replaces the bf16-magic dequantisation of these configs with an fp16-magic one
# (Q8 = 2) and, optionally, the int8 MMA on QK^T (Q8 = 3) -- same cache, same loads, same
# read floor, 25 + 56 ALU ops per lane per tile instead of 161 + 160 (measured in the
# device IR, work/R9/R-q1-dequant/logs/).  cfg 42 mirrors 28 and cfg 47 mirrors 38, so
# they are drop-in A/B pairs, and VLLM_FL_MX_DEC_CFG_I8 / _SIDE select them.
#
# R9/G TIMED THEM on the C500 (us per layer, 42 layers in one graph, interleaved A/B,
# /data/exp/R9/G/logs/r9_full_20261025_075543.json; x = vs the vendor bf16 kernel on the
# bf16 cache, so > 1 is a win; the last column is the fraction of this kernel's OWN int8
# read floor, cfg 49):
#
#   variant             16 r/4k      48 r/4k      64 r/4k      48 r/16k     64 r/16k
#   vendor bf16          80.4         183.7        243.8        601.5        784.7
#   cfg 28 int8 (R8)     86.4 x0.931  164.2 x1.119 208.2 x1.171 497.5 x1.209 630.4 x1.245
#   cfg 38 int8 side    102.3 x0.786  182.2 x1.008 221.8 x1.099 519.0 x1.159 639.6 x1.227
#   cfg 42 fp16         76.4 x1.053   130.8 x1.405 162.6 x1.500 365.4 x1.646 460.5 x1.704
#   cfg 42 fp16 + fold  75.7 x1.063   125.9 x1.459 156.8 x1.555 349.4 x1.721 440.3 x1.782
#   cfg 45 fp16 PF0     75.2 x1.069   131.6 x1.396 165.0 x1.478 374.0 x1.608 472.8 x1.660
#   cfg 47 fp16 SCHED0  94.1 x0.855   150.8 x1.218 182.8 x1.334 388.1 x1.550 469.7 x1.670
#   cfg 50 int8-MMA     75.1 x1.071   125.8 x1.461 156.9 x1.554 350.0 x1.719 441.2 x1.779
#   read floor (c49)    35.7          84.9         111.5        287.0        368.8
#
# so THE DEFAULTS BELOW ARE NOW cfg 42 (with N1_Q1_PV_FOLD = 1, which q1_int8.cuh makes
# the default build) for the decode-only site and cfg 47 for the side stream of an
# overlapped mixed step -- the same SCHED 1 / SCHED 0 split as the R8 pair they replace,
# at 71% / 84% of the int8 read floor instead of 51% / 59%.  Fitting (time - load floor)
# against the ALU ops per lane per tile measured in the device IR (int8 418, fp16 141,
# fold 134, int8-MMA 120) gives R^2 = 0.992-0.998 at all five shapes, so the reader is
# VALU-bound exactly as the R8/Q1 audit said and the residual 4k gap (a fixed 24.5 us at
# 64 rows / 4k, 35.0 at 16 rows) is launch, tail and wave quantisation, not arithmetic.
# cfg 50 (int8 MMA on QK^T) is NOT shipped although it ties cfg 42 + fold on time: its
# err_contract is 9.0e-3 .. 2.3e-2 against the fold's 1.5-2.5e-3, i.e. 27-35% above the
# quantisation floor, and the exact N1_Q1_QPLANES = 2 variant that fixes the arithmetic
# is 4.3% slower than the fold at 64 rows / 4k.  cfg 45 (PF 0) is the better config at
# 16 rows only and 2-7% worse at 48 / 64, which the contest's concurrency 64 never sees.
# mxcc -resource-usage, all at 2 CTAs/AP (work/R9/R-q1-dequant/logs/r9_resource_table.txt):
#   42 fp16 W4 PF1 SCHED1  238 reg, 0 B   <- the A/B partner of cfg 28 (also 238, 0 B)
#   44 fp16 W2 PF2 SCHED1  246 reg, 0 B
#   45 fp16 W4 PF0 SCHED1  200 reg, 0 B
#   47 fp16 W4 PF1 SCHED0  236 reg, 0 B   <- the A/B partner of cfg 38 (also 236, 0 B)
#   49 fp16 MODE 1 read floor 78 reg, 0 B -- IDENTICAL to cfg 35, which is the proof that
#      the loads did not change and only the consume loop did
#   50 int8-MMA QK W4 PF1 SCHED1 224 reg, 51 PF0 192, 52 SCHED0 222, all 0 B
#      (+18..22 reg with N1_Q1_QPLANES=2, still 0 B)
# DO NOT TIME 43 / 46 / 48 (W4 PF2, W8 PF1, SCHED0 PF2): 40-64 B stack frame, the R7/N5
# failure mode.  The fp16 flavour carries the SAME register pressure as the bf16 one, so
# PF 2 and W 8 stay out of reach exactly as they are for cfgs 29 / 31 / 39.
CFG_DEC_I8 = _env_int("VLLM_FL_MX_DEC_CFG_I8", 42)    # SCHED 1, W 4, PF 1, fp16 dequant
CFG_SIDE_I8 = _env_int("VLLM_FL_MX_DEC_CFG_I8_SIDE", 47)   # SCHED 0, W 4, PF 1, fp16
CFG_DEC_I8_R8 = 28       # the R8 pair, kept as the documented A/B reference and rollback
CFG_SIDE_I8_R8 = 38
INT8_CFGS = (28, 29, 30, 31, 32, 33, 34, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52)
INT8_FLOOR_CFGS = (35, 36, 37, 41, 49)                # MODE 1: read-only floor
FP16_CFGS = (42, 44, 45, 47)                          # Q8 = 2 (R9/R), spill-free only
FP16_SPILL_CFGS = (43, 46, 48)                        # built, but never to be timed
I8MMA_CFGS = (50, 51, 52)                             # Q8 = 3 (R9/R)
# There is deliberately NO row / context gate here, because there is nothing to gate
# to: once the cache holds int8 the vendor kernel cannot read it, so this kernel is
# the only reader and a gate could only pick another int8 config.  With the R8 pair
# that was a known defect at few rows (16 rows @ 4k: cfg 28 86.4 us against the
# vendor's 80.4 = x0.931); WITH cfg 42 + fold IT IS FIXED -- x1.063 at 16 rows, the
# worst measured shape, and x1.459 / x1.555 / x1.721 / x1.782 at 48 and 64 rows over
# 4k and 16k contexts -- so the decode-only site needs no row gate at all now.  The
# side-stream site still loses at 16 rows (cfg 47 x0.855), which is why
# mx_kv_int8.SIDE_MIN_ROWS keeps a gate THERE: below it the step is not overlapped and
# the rows go out on the main stream with cfg 42, which wins at every measured shape.

CTAS_PER_AP = 2          # grid = APs x CTAS_PER_AP, constant for the whole process
# Workspaces, one per call site: False = main stream (decode-only, eager and
# in-graph), True = side stream of the overlapped mixed step.  Two sites never run
# at the same time (one step is either mixed or decode-only), but keeping them
# apart keeps N4 from touching anything N1 relies on.
SITES = (False, True)

# Start-up self-test: also capture the kernel in a CUDA graph and replay it with
# different seq_lens (0 = skip; the kernel then still runs, only unproven here).
GRAPH_SELFTEST = os.environ.get("VLLM_FL_MX_DEC_GRAPHTEST", "1") not in ("0", "off", "no")

N1_G = 8                 # q heads per kv head built into the library
SUPPORTED_D = (128,)
TILE = 16

INT8_OK = False          # the library has the Q8 = 1 configs (set by setup())

# state set by setup()
ENABLED = False
STATUS = "not set up"
LIB_PATH: str | None = None
_LIB = None
_SETUP_DONE = False
_WS: dict = {}           # (device index, side stream?) -> Workspace
_REJECTS: dict = {}      # library return code -> count (calls handed back to the vendor)


def rejects() -> dict:
    """Library return code -> count, for the caller's error and log messages."""
    return dict(_REJECTS)

# ---------------------------------------------------------------------------
# library: sources, build, load
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("n1_dec_attn.cuh", "q1_int8.cuh", "n1_capi.cu")
MACA = os.environ.get("MACA_PATH", "/opt/maca")
ARCH = os.environ.get("VLLM_FL_MX_NATIVE_ARCH", "xcore1000")


def src_dir() -> str:
    return os.environ.get("VLLM_FL_MX_NATIVE_SRC") or os.path.join(_HERE, "csrc")


def cache_dir() -> str:
    return os.environ.get("VLLM_FL_MX_NATIVE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "vllm_fl", "mx_native")


def src_hash(sdir: str | None = None) -> str:
    h = hashlib.sha1()
    sdir = sdir or src_dir()
    for name in SOURCES:
        with open(os.path.join(sdir, name), "rb") as f:
            h.update(f.read())
    h.update(ARCH.encode())
    h.update(os.environ.get("VLLM_FL_MX_NATIVE_FLAGS", "").encode())
    return h.hexdigest()[:12]


def lib_path(sdir: str | None = None, out_dir: str | None = None) -> str:
    return os.path.join(out_dir or cache_dir(), f"libn1_dec_attn_{src_hash(sdir)}.so")


def _mxcc_cmds(src: str, inc: str, out: str) -> list[list[str]]:
    mxcc = os.path.join(MACA, "mxgpu_llvm", "bin", "mxcc")
    base = [mxcc, "-x", "maca", "-O3", "-std=c++17", "-fPIC", "-shared", f"-I{inc}",
            f"--maca-path={MACA}"]
    rt = [f"-L{os.path.join(MACA, 'lib')}", "-lmcruntime"]
    extra = os.environ.get("VLLM_FL_MX_NATIVE_FLAGS", "").split()
    return [base + [f"--offload-arch={ARCH}"] + extra + [src, "-o", out] + rt,
            base + ["-offload-arch", ARCH] + extra + [src, "-o", out] + rt]


def build(sdir: str | None = None, out_dir: str | None = None, force: bool = False,
          timeout: int = 1200) -> tuple[str, str]:
    """Compile libn1_dec_attn (CPU only; no GPU context).  Returns (path, log tail).
    Concurrent builders serialise on a lock file; an existing library for the same
    source hash is reused unless force."""
    import fcntl  # noqa: PLC0415

    sdir = sdir or src_dir()
    out_dir = out_dir or cache_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = lib_path(sdir, out_dir)
    with open(os.path.join(out_dir, ".n1_build.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(path) and not force:
            return path, "cached"
        tmp = f"{path}.{os.getpid()}.tmp"
        logs = []
        src = os.path.join(sdir, "n1_capi.cu")
        # default build first; the conservative N1_SAFE build only if that fails
        cmds = _mxcc_cmds(src, sdir, tmp)
        cmds.append([cmds[0][0], "-DN1_SAFE=1"] + cmds[0][1:])
        for cmd in cmds:
            t0 = time.time()
            try:
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=timeout,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
                rc, txt = p.returncode, p.stdout
            except (OSError, subprocess.TimeoutExpired) as exc:
                rc, txt = -1, repr(exc)
            logs.append(f"$ {' '.join(cmd)}\nrc={rc} {time.time() - t0:.1f}s\n{txt[-3000:]}")
            if rc == 0 and os.path.exists(tmp):
                os.replace(tmp, path)
                return path, logs[-1][-2000:]
        raise RuntimeError("mxcc build of libn1_dec_attn failed:\n" + "\n".join(logs)[-6000:])


VP, I32, I64, F32 = C.c_void_p, C.c_int, C.c_longlong, C.c_float
PI32 = C.POINTER(C.c_int)
_SIGS = {
    "n1_version": ([], I32),
    "n1_features": ([], I32),
    "n1_num_cfgs": ([], I32),
    "n1_cfg_info": ([I32, PI32], I32),
    "n1_attr": ([I32, PI32], I32),
    "n1_dec_attn": ([I32, VP, I64, I64, VP, VP, I64, I64, I64, VP, I64, VP, I32, VP, I64, I64,
                     VP, VP, VP, I32, I32, I32, I32, F32, I32, I32, I32, I32, VP, VP, I32,
                     I32], I32),
    "n1_probe_mma": ([VP, VP], I32),
}


def load(path: str):
    lib = C.CDLL(path)
    for name, (args, res) in _SIGS.items():
        f = getattr(lib, name)
        f.argtypes = args
        f.restype = res
    if lib.n1_version() % 100 != N1_G:
        raise RuntimeError(f"library G={lib.n1_version() % 100}, expected {N1_G}")
    if lib.n1_version() // 100 < 2:
        raise RuntimeError(f"library version {lib.n1_version() // 100} < 2 (no SCHED 1)")
    return lib


def cfg_info(lib, cfg: int) -> dict:
    a = (C.c_int * 8)()
    if lib.n1_cfg_info(cfg, a) != 0:
        raise ValueError(f"no config {cfg}")
    return {"w": a[0], "pf": a[1], "ldk": a[2], "mode": a[3], "kl": a[4], "sched": a[5],
            "q8": a[6]}


def attr(lib, cfg: int) -> dict:
    a = (C.c_int * 5)()
    rc = lib.n1_attr(cfg, a)
    return {"rc": rc, "regs": a[0], "spill": a[1], "max_threads": a[2], "smem": a[3],
            "ctas_per_ap": a[4]}


# ---------------------------------------------------------------------------
# workspace and launch
# ---------------------------------------------------------------------------
class Workspace:
    """Partials + counters of one call site (main stream / side stream).

    Allocated ONCE per site, for the worst case over every captured batch size and
    every context (the slot count does not depend on the contexts: SCHED 0 needs
    target_items + hkv * max_rows slots because its chunk length doubles until the
    item count fits, SCHED 1 needs exactly 2 per CTA).  So one workspace serves all
    captured graphs, nothing is allocated at replay, and the addresses baked into a
    captured launch stay valid for the life of the process.

    The counters must be zero before the first launch; every launch leaves them
    zero again (the merging CTA resets its segment counter; SCHED 0's last CTA out
    resets the claim / exit counters).  There is therefore no host-side zeroing and
    no memset node between replays."""

    def __init__(self, device, max_rows: int, hkv: int, grid: int, target_items: int,
                 g: int = N1_G, d: int = 128):
        self.grid = int(grid)
        self.target = max(1, int(target_items))
        self.max_rows = int(max_rows)
        # rows * hkv (one slot per segment) + 2 * grid (SCHED 1) + target (SCHED 0)
        self.max_items = self.target + hkv * self.max_rows + 2 * self.grid
        self.o = torch.empty(self.max_items * g * d, dtype=torch.float32, device=device)
        self.ml = torch.empty(self.max_items * 32, dtype=torch.float32, device=device)
        self.ctr = torch.zeros(2 + hkv * self.max_rows, dtype=torch.int32, device=device)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.o, self.ml, self.ctr))


def launch(lib, ws: Workspace, q3, kc, vc, bt, seq, out3, scale: float, *, cfg: int,
           grid: int, target: int, min_ct: int, claim_lead: int, stream, sink=None,
           q8: int = 0, q8_sc: int = 0) -> int:
    """Raw call.  q3/out3: [rows, HQ, 128]; kc/vc: [blocks, BS, HKV, 128]; bt/seq int32.
    stream: an int/c_void_p stream handle (None in the CPU emulator).  Returns rc.
    q8 = 1 (R8/Q1): kc / vc are the SAME bf16 tensors, reinterpreted as the INT8 layout
    of ops/mx_kv_int8.py; q8_sc is the byte offset of the scale block inside a half page
    (view.sc_off).  The cfg must be one of INT8_CFGS -- the library checks it."""
    rows = int(seq.shape[0])
    return lib.n1_dec_attn(
        int(cfg), q3.data_ptr(), q3.stride(0), q3.stride(1), kc.data_ptr(), vc.data_ptr(),
        kc.stride(0), kc.stride(1), kc.stride(2), bt.data_ptr(), bt.stride(0), seq.data_ptr(),
        rows, out3.data_ptr(), out3.stride(0), out3.stride(1), ws.o.data_ptr(),
        ws.ml.data_ptr(), ws.ctr.data_ptr(), ws.max_items, int(q3.shape[1]), int(kc.shape[2]),
        int(kc.shape[1]) // TILE, float(scale), int(grid), int(target), int(min_ct),
        int(claim_lead), None if sink is None else sink.data_ptr(), stream,
        int(q8), int(q8_sc))


def layout_ok(key_cache: torch.Tensor, value_cache: torch.Tensor) -> bool:
    """Cache layout the library addresses: [blocks, BS, HKV, 128] bf16, unit inner
    stride, block = run of BS tokens, K and V with identical strides, 16 B aligned."""
    kc, vc = key_cache, value_cache
    try:
        return (kc.dtype == torch.bfloat16 and vc.dtype == kc.dtype and kc.dim() == 4
                and tuple(kc.shape) == tuple(vc.shape) and kc.stride() == vc.stride()
                and kc.shape[3] in SUPPORTED_D and kc.stride(3) == 1
                and kc.shape[1] % TILE == 0 and kc.stride(0) == kc.shape[1] * kc.stride(1)
                and kc.stride(1) % 8 == 0 and kc.stride(2) % 8 == 0
                and kc.data_ptr() % 16 == 0 and vc.data_ptr() % 16 == 0)
    except Exception:  # noqa: BLE001
        return False


def supports(num_heads_q: int, num_heads_kv: int, head_size: int) -> bool:
    return (num_heads_kv >= 1 and num_heads_q == N1_G * num_heads_kv
            and head_size in SUPPORTED_D)


def wanted(num_rows: int) -> bool:
    """Policy for the decode rows of an OVERLAPPED MIXED step (R6/N1, side stream,
    atomic-claimed items): the decode-row count.  Mixed steps are never captured
    (steps over 512 tokens have no graph and PIECEWISE graphs run attention
    eagerly), and the caller checks is_current_stream_capturing() as well."""
    if not ENABLED or num_rows < MIXED_MIN_ROWS or num_rows > MIXED_MAX_ROWS:
        return False
    return not torch.cuda.is_current_stream_capturing()


def wanted_decode(num_rows: int, max_ctx: int = 0) -> bool:
    """Policy for a DECODE-ONLY step (R7/N4, stream-K), the path that is captured in
    the FULL decode graphs.

    Only features that are constant for a captured batch size may be used here,
    because this runs once at capture and the decision is then frozen for every
    replay: the row count (= the captured size), the head counts / head size / page
    size / cache layout (checked by the caller and by the library) and the AP count
    (baked into ws.grid).  The context length is NOT one of them -- it grows between
    replays -- so the context gate is applied only when we are not capturing, where
    max_ctx is the metadata's decode_max_seq_len (a host int the builder already
    read from the pinned CPU copy, no sync)."""
    if not ENABLED or num_rows < DEC_MIN_ROWS or num_rows > DEC_MAX_ROWS:
        return False
    if torch.cuda.is_current_stream_capturing():
        return True
    return max_ctx <= 0 or max_ctx >= DEC_MIN_CTX


def decode_attention(q, key_cache, value_cache, out, block_table, seq_lens, scale,
                     side: bool = False) -> bool:
    """Decode rows: q (n, HQ, D) or (n, 1, HQ, D); out (n, HQ, D) or any view of n x HQ x D
    elements.  Same contract as mx_prefill_attn.decode_attention (graph-safe: it only
    enqueues one kernel, reads nothing back and allocates nothing).  side=True picks the
    mixed-step site (claimed items), side=False the decode-only site (stream-K).
    Returns False (nothing enqueued) when the library rejects the arguments (stride /
    alignment / size checks) or no workspace exists for this call site -- the caller
    then issues the vendor call; raises on a launch error."""
    n = int(seq_lens.shape[0])
    side = bool(side)
    ws = _WS.get((q.device.index, side))
    if ws is None or n > ws.max_rows:
        return False
    q3 = q if q.dim() == 3 else q.reshape(n, q.shape[-2], q.shape[-1])
    out3 = out if out.dim() == 3 else out.view(n, q3.shape[1], q3.shape[2])
    rc = launch(_LIB, ws, q3, key_cache, value_cache, block_table, seq_lens, out3, scale,
                cfg=CFG_MIXED if side else CFG_DEC, grid=ws.grid, target=ws.target,
                min_ct=MIN_CHUNK_TILES, claim_lead=CLAIM_LEAD,
                stream=torch.cuda.current_stream(q.device).cuda_stream)
    if rc == 0:
        return True
    if rc >= 1000:
        raise RuntimeError(f"n1_dec_attn launch failed rc={rc} (rows {n}, q {tuple(q3.shape)} "
                           f"strides {q3.stride()}, cache {tuple(key_cache.shape)})")
    _REJECTS[rc] = _REJECTS.get(rc, 0) + 1
    return False


def decode_attention_int8(q, kv, view, block_table, seq_lens, scale, *, cfg=None,
                          side: bool = False, out=None):
    """R8/Q1: the same decode kernel over the INT8 cache.  kv = the layer's bf16 KV
    tensor (2, NB, PAGE, HKV, D); view = ops/mx_kv_int8.view_of(kv).  Returns the output
    tensor (allocated when out is None) or None when the library rejected the call --
    with INT8 KV there is no vendor fallback, so the caller must treat None as a fatal
    configuration error (the start-up self-test exercises every shape the server will
    see, so this cannot appear only at run time).  Graph-safe: one kernel, no allocation
    when out is given, nothing read back."""
    n = int(seq_lens.shape[0])
    side = bool(side)
    ws = _WS.get((q.device.index, side))
    if ws is None or n > ws.max_rows or not INT8_OK:
        return None
    q3 = q if q.dim() == 3 else q.reshape(n, q.shape[-2], q.shape[-1])
    if out is None:
        out = torch.empty(n, q3.shape[1], q3.shape[2], dtype=q3.dtype, device=q3.device)
    out3 = out if out.dim() == 3 else out.view(n, q3.shape[1], q3.shape[2])
    kc, vc = kv[0], kv[1]
    q8, q8_sc = view.native_args()
    rc = launch(_LIB, ws, q3, kc, vc, block_table, seq_lens, out3, scale,
                cfg=(cfg if cfg is not None else (CFG_SIDE_I8 if side else CFG_DEC_I8)),
                grid=ws.grid, target=ws.target, min_ct=MIN_CHUNK_TILES,
                claim_lead=CLAIM_LEAD,
                stream=torch.cuda.current_stream(q.device).cuda_stream,
                q8=q8, q8_sc=q8_sc)
    if rc == 0:
        return out
    if rc >= 1000:
        raise RuntimeError(f"n1_dec_attn (int8) launch failed rc={rc} (rows {n})")
    _REJECTS[rc] = _REJECTS.get(rc, 0) + 1
    return None


# ---------------------------------------------------------------------------
# start-up self-test
# ---------------------------------------------------------------------------
def reference(q3, kc, vc, bt, seq, scale):
    """fp32 torch reference (gathers every row's pages)."""
    n, hq, d = q3.shape
    hkv, bs = kc.shape[2], kc.shape[1]
    g = hq // hkv
    out = torch.zeros(n, hq, d, dtype=torch.float32, device=q3.device)
    for r in range(n):
        k = int(seq[r])
        if k <= 0:
            continue
        pos = torch.arange(k, device=q3.device)
        blk = bt[r][pos // bs].long()
        K = kc[blk, pos % bs].float()            # [k, hkv, d]
        V = vc[blk, pos % bs].float()
        Q = q3[r].float().view(hkv, g, d)
        S = torch.einsum("hgd,khd->hgk", Q, K) * scale
        P = torch.softmax(S, dim=-1)
        out[r] = torch.einsum("hgk,khd->hgd", P, V).reshape(hq, d)
    return out


SELFTEST_CTX = (0, 1, 15, 16, 17, 333, 1500, 4100)
# second context set for the graph replay test: same batch, different lengths (one row
# that was alive becomes padding and one that was padding becomes alive)
SELFTEST_CTX2 = (2048, 0, 16, 1, 4099, 48, 0, 700)


def _self_test(device, hq, hkv, d, block_size, ref_kvcache):
    gen = torch.Generator().manual_seed(0)
    ctx = list(SELFTEST_CTX)
    n = len(ctx)
    cmax = max(list(SELFTEST_CTX) + list(SELFTEST_CTX2))
    width = max(1, -(-cmax // block_size)) + 3
    nb = width * n + 7

    def rnd(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16).to(device).contiguous()

    kv = rnd(2, nb, block_size, hkv, d)
    kc, vc = kv.unbind(0)
    perm = torch.randperm(nb, generator=gen).to(torch.int32)
    bt_cpu = torch.zeros(n, width, dtype=torch.int32)
    for r in range(n):
        bt_cpu[r, :width] = perm[r * width:(r + 1) * width]
    bt = bt_cpu.to(device)
    seq = torch.tensor(ctx, dtype=torch.int32, device=device)
    q = (rnd(n, hq, d).float() * 2.0).to(torch.bfloat16)   # peakier softmax
    scale = d ** -0.5
    report = []
    tol = 2e-2

    # The stream handle MUST be read at launch time, not hoisted: inside
    # torch.cuda.graph() the current stream is the capture stream, and a launch
    # on the default stream is NOT captured (MACA accepts it and runs it eagerly,
    # so the graph comes out empty and every replay leaves the output untouched --
    # the 'graph self-test A: replay != eager (max nan)' of 2026-10-25 03:08).
    def run(cfg, min_ct, side, out=None, seq_t=None):
        ws = _WS[(torch.device(device).index, side)]
        if out is None:
            out = torch.full((n, hq, d), float("nan"), dtype=torch.bfloat16, device=device)
        rc = launch(_LIB, ws, q, kc, vc, bt, seq_t if seq_t is not None else seq, out, scale,
                    cfg=cfg, grid=ws.grid, target=ws.target, min_ct=min_ct,
                    claim_lead=CLAIM_LEAD,
                    stream=torch.cuda.current_stream(device).cuda_stream)
        if rc != 0:
            raise RuntimeError(f"self-test launch rc={rc} (cfg {cfg}, side {side})")
        return out

    def checked(name, cfg, min_ct, side, ref):
        ws = _WS[(torch.device(device).index, side)]
        out = run(cfg, min_ct, side)
        torch.cuda.synchronize(device)
        if int(ws.ctr.abs().sum()) != 0:
            raise RuntimeError(f"self-test {name}: counters not reset after the launch")
        if not torch.isfinite(out.float()).all():
            raise RuntimeError(f"self-test {name}: non-finite output")
        err = float((out.float() - ref).abs().max())
        report.append(f"{name}={err:.1e}")
        if err > tol:
            raise RuntimeError(f"self-test {name}: max abs err {err:.3e} > {tol}")
        again = run(cfg, min_ct, side)
        torch.cuda.synchronize(device)
        if not torch.equal(out, again):
            raise RuntimeError(f"self-test {name}: two launches differ")
        return out

    ref = reference(q, kc, vc, bt, seq, scale)
    # mixed-step site (claimed items) and decode-only site (stream-K), incl. a forced
    # fine split of every row
    checked("mixed", CFG_MIXED, MIN_CHUNK_TILES, True, ref)
    checked("mixed_split1", CFG_MIXED, 1, True, ref)
    out_dec = checked("dec", CFG_DEC, MIN_CHUNK_TILES, False, ref)

    # CUDA-graph capture + replay of the decode-only path: capture once, then replay
    # twice with DIFFERENT contexts written into the same device buffer, and check both
    # against the eager result of those contexts and against each other (bit-stable).
    if GRAPH_SELFTEST:
        seq_g = seq.clone()
        out_g = torch.full((n, hq, d), float("nan"), dtype=torch.bfloat16, device=device)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            run(CFG_DEC, MIN_CHUNK_TILES, False, out=out_g, seq_t=seq_g)
        for tag, cx in (("A", list(SELFTEST_CTX)), ("B", list(SELFTEST_CTX2)),
                        ("A2", list(SELFTEST_CTX)), ("B2", list(SELFTEST_CTX2))):
            seq_g.copy_(torch.tensor(cx, dtype=torch.int32, device=device))
            out_g.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            eager = run(CFG_DEC, MIN_CHUNK_TILES, False, seq_t=seq_g)
            torch.cuda.synchronize(device)
            ws = _WS[(torch.device(device).index, False)]
            if int(ws.ctr.abs().sum()) != 0:
                raise RuntimeError(f"graph self-test {tag}: counters not reset")
            if not torch.equal(out_g, eager):
                bad = float((out_g.float() - eager.float()).abs().max())
                raise RuntimeError(f"graph self-test {tag}: replay != eager (max {bad:.3e})")
            if tag == "A2":
                report.append("graph=bitstable")
        del graph
        torch.cuda.synchronize(device)

    if ref_kvcache is not None:
        live = [r for r, c in enumerate(ctx) if c > 0]
        idx = torch.tensor(live, device=device)
        vend = ref_kvcache(q=q[idx].view(len(live), 1, hq, d), k_cache=kc, v_cache=vc,
                           block_table=bt[idx], cache_seqlens=seq[idx], softmax_scale=scale,
                           causal=True, window_size=(-1, -1), alibi_slopes=None, softcap=0,
                           s_aux=None)
        err = float((out_dec[idx].float() - vend.view(len(live), hq, d).float()).abs().max())
        report.append(f"vendor={err:.1e}")
        if err > tol:
            raise RuntimeError(f"self-test vs vendor: max abs err {err:.3e} > {tol}")
    return " ".join(report)


def setup(device, num_heads_q, num_heads_kv, head_size, block_size, dtype, max_rows,
          ref_kvcache=None) -> str:
    """Once per process: library (prebuilt for the current sources or built now),
    workspaces for the call sites in SITES, self-test.  Sets ENABLED."""
    global ENABLED, STATUS, LIB_PATH, _LIB, _SETUP_DONE, INT8_OK
    if _SETUP_DONE:
        return STATUS
    _SETUP_DONE = True
    ENABLED = False
    try:
        if not REQUESTED:
            raise RuntimeError(f"{ENV_FLAG}={MODE}")
        if not supports(num_heads_q, num_heads_kv, head_size):
            raise RuntimeError(f"heads {num_heads_q}/{num_heads_kv} x {head_size} not built")
        if dtype != torch.bfloat16:
            raise RuntimeError(f"kv cache dtype {dtype}")
        if block_size % TILE:
            raise RuntimeError(f"block size {block_size}")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # workspaces are keyed by q.device.index, which is always set on the GPU
            device = torch.device("cuda", torch.cuda.current_device())
        path = os.environ.get("VLLM_FL_MX_NATIVE_LIB") or lib_path()
        how = "prebuilt"
        if not os.path.exists(path):
            path, _ = build()
            how = "built"
        _LIB = load(path)
        LIB_PATH = path
        aps = torch.cuda.get_device_properties(device).multi_processor_count
        grid = CTAS_PER_AP * aps
        rows = min(int(max_rows), DEC_MAX_ROWS)
        for side in SITES:
            _WS[(device.index, side)] = Workspace(device, rows, num_heads_kv, grid,
                                                  ITEMS_PER_CTA * grid)
        # R8/Q1: does this library carry INT8 configs for the two sites?  ops/mx_kv_int8.py
        # checks INT8_OK before it enables the INT8 cache (and runs its own self-test).
        # q8 >= 1 means "reads the int8 cache"; the value says HOW (1 bf16-magic dequant,
        # 2 R9/R fp16-magic, 3 + int8 MMA on QK), and 2 / 3 need library version >= 4.
        _q8s = [cfg_info(_LIB, c).get("q8") for c in (CFG_DEC_I8, CFG_SIDE_I8)]
        _vmin = 4 if any((q or 0) >= 2 for q in _q8s) else 3
        INT8_OK = bool(_LIB.n1_version() // 100 >= _vmin
                       and all((q or 0) >= 1 for q in _q8s))
        ENABLED = True       # the self-test calls decode paths through wanted_*()
        report = _self_test(device, num_heads_q, num_heads_kv, head_size, block_size,
                            ref_kvcache)
    except Exception as exc:  # noqa: BLE001
        ENABLED = False
        INT8_OK = False   # a failed self-test must also keep the INT8 cache out
        STATUS = f"off ({type(exc).__name__}: {str(exc)[:300]})"
        return STATUS
    ENABLED = True
    ws0 = _WS[(device.index, SITES[0])]
    STATUS = (f"on: decode-only steps (incl. FULL graphs) cfg {CFG_DEC} stream-K, rows "
              f">= {DEC_MIN_ROWS}, eager ctx >= {DEC_MIN_CTX}; overlapped mixed steps "
              f"cfg {CFG_MIXED} claimed items, rows >= {MIXED_MIN_ROWS} (grid {grid} = "
              f"{aps} APs x {CTAS_PER_AP}, items/CTA {ITEMS_PER_CTA}, min chunk "
              f"{MIN_CHUNK_TILES} tiles, claim lead {CLAIM_LEAD}, workspace "
              f"{len(SITES)} x {ws0.nbytes() / 2**20:.1f} MiB for <= {rows} rows, lib "
              f"{how} {os.path.basename(path)}; int8 cfgs "
              f"{'yes' if INT8_OK else 'no'}; self-test {report})")
    return STATUS


# ---------------------------------------------------------------------------
# build entry point (plugin build step):  python3 mx_dec_native.py --build
# ---------------------------------------------------------------------------
def _main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description="build libn1_dec_attn (MetaX native decode attention)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--src-dir", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.build:
        ap.print_help()
        return 2
    path, log = build(a.src_dir, a.out_dir, a.force)
    print(log)
    print(f"N1_LIB {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

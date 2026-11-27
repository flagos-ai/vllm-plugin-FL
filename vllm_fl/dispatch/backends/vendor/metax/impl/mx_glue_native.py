# SPDX-License-Identifier: Apache-2.0
# 2026 - FlagOS S2 contest work on top of the MetaX Maca backend.
"""MetaX: own native glue kernels of a Llama-style decoder layer (R7/N6).

Three kernels of our own design (csrc/n6_glue.cuh, built with mxcc, called through
ctypes), used by impl/mx_glue.py for the steps where they were measured to win:

* ``add_rms_norm``  residual add + RMSNorm in one pass, the row in registers
                    (fp32 accumulation, one rounding per stored tensor);
* ``rope_kv``       neox RoPE of q and of k + the paged KV write in ONE launch;
* ``silu_mul``      SiLU(gate) * up of the gate_up GEMM output.

Why native and not the FlagGems/Triton glue of R6/G1: on the mixed (chunked-prefill)
steps, which are NOT graph-captured, a Triton launch of these shapes costs 61-116 us of
HOST time against 22-30 us of GPU time at M=2048, so G1's kernels moved the mixed step's
cost from the GPU to the host (measured: 4k mean TTFT 1904 -> 2055 ms, E2E +0.27% only).
One ctypes call into this library is ~4 us of host time, so the mixed-step glue actually
shrinks.  On the graph-captured decode steps the host cost is irrelevant and the choice
between this library and the FlagGems kernels is pure GPU time -- impl/mx_glue.py keys
that choice on the step's token count (never on the case).

Compliance: new kernels of our own design inside vllm-plugin-FL, reached through the
plugin's own op-selection path (the vLLM IR provider for fused_add_rms_norm and the MetaX
attention impl), with the original implementation kept as the fallback; selection keys on
runtime features only (token count, hidden size, head counts, dtype, strides, alignment);
serve arguments are untouched.  Nothing here touches the GPU at import time.

Switches: VLLM_FL_MX_GLUE_NATIVE=0 disables the library (impl/mx_glue.py then leaves the
whole glue on the deployed path), VLLM_FL_MX_NATIVE_ARCH / _FLAGS / _DIR / _SRC as in
attention/ops/mx_dec_native.py.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import logging
import os
import subprocess
import time

import torch

logger = logging.getLogger(__name__)

ENV_FLAG = "VLLM_FL_MX_GLUE_NATIVE"
REQUESTED = os.environ.get(ENV_FLAG, "1").strip().lower() not in ("0", "off", "false", "no")

# Pinned launch configs (bench_n6.py --emit rewrites the table between the markers).
# norm: hidden size -> library config (each config covers exactly N = threads * VPT);
# rope: max token count -> config (threads per CTA) and q heads per CTA group;
# silu: max rows -> config.
#
# From the C500 GPU run of 2026-10-24 (logs/gpu/n6_full_20261024_225056.log), with ONE
# correction: the config sweep of that run does not check the launch return code, so a
# config the kernel REJECTS is recorded as a 0.55-0.73 us "win".  Its norm pick (cfg 8 =
# 4 warps x 32 values = N 8192) does not cover the served hidden size 2048 and would be
# rejected with -11, i.e. the norm fusion would never run.  The entries below are the
# configs whose N matches the hidden size, so 2048 -> cfg 6 (4 warps x 8 values, 31.7 us
# at M=2048, the config every timing of this round was measured with).  _check_tables()
# re-checks this against the library at set-up, so a bad table entry can only cost the
# fusion, never correctness.
# --- pinned native glue tables: begin ---
NORM_CFG = {512: 0, 1024: 3, 2048: 6, 4096: 9, 8192: 11}
ROPE_CFG = [(256, (0, 1)), (1 << 30, (2, 2))]
SILU_CFG = [(1 << 30, 0)]
# --- pinned native glue tables: end ---

# kernel ids of n6_attr / n6_cfg_info
K_NORM, K_ROPE, K_SILU = 0, 1, 2

ENABLED = False          # library built, loaded and self-tested
NORM_OK = ROPE_OK = SILU_OK = False
# R10/F: the RoPE-only entry point (rotation without the cache write).  Separate
# flag because only the INT8 KV path uses it, and because it is the one RoPE that
# MUST be available for that path: the torch fallback is illegal under capture.
ROPE_ONLY_OK = False
STATUS = "not set up"
LIB_PATH: str | None = None
_LIB = None
_BUILD_DONE = False      # mxcc build + ctypes load tried (CPU only)
_SETUP_DONE = False      # ... and the device self-test ran
_REJECTS: dict = {}

_HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("n6_glue.cuh", "n6_capi.cu")
MACA = os.environ.get("MACA_PATH", "/opt/maca")
ARCH = os.environ.get("VLLM_FL_MX_NATIVE_ARCH", "xcore1000")


# --------------------------------------------------------------------------- #
# library: sources, build, load
# --------------------------------------------------------------------------- #
def src_dir() -> str:
    return os.environ.get("VLLM_FL_MX_GLUE_SRC") or os.path.join(_HERE, "csrc")


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
    return os.path.join(out_dir or cache_dir(), f"libn6_glue_{src_hash(sdir)}.so")


def _mxcc_cmds(src: str, inc: str, out: str) -> list[list[str]]:
    mxcc = os.path.join(MACA, "mxgpu_llvm", "bin", "mxcc")
    # -ffp-contract=off: x1*cos - x2*sin must NOT become an fma, or the rotated q and the
    # cache contents stop being bitwise equal to the deployed Inductor RoPE.
    base = [mxcc, "-x", "maca", "-O3", "-std=c++17", "-ffp-contract=off", "-fPIC", "-shared",
            f"-I{inc}", f"--maca-path={MACA}"]
    rt = [f"-L{os.path.join(MACA, 'lib')}", "-lmcruntime"]
    extra = os.environ.get("VLLM_FL_MX_NATIVE_FLAGS", "").split()
    return [base + [f"--offload-arch={ARCH}"] + extra + [src, "-o", out] + rt,
            base + ["-offload-arch", ARCH] + extra + [src, "-o", out] + rt]


def build(sdir: str | None = None, out_dir: str | None = None, force: bool = False,
          timeout: int = 900) -> tuple[str, str]:
    """Compile libn6_glue (CPU only; no GPU context).  Returns (path, log tail).
    Concurrent builders serialise on a lock file; an existing library for the same
    source hash is reused unless force."""
    import fcntl  # noqa: PLC0415

    sdir = sdir or src_dir()
    out_dir = out_dir or cache_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = lib_path(sdir, out_dir)
    with open(os.path.join(out_dir, ".n6_build.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(path) and not force:
            return path, "cached"
        tmp = f"{path}.{os.getpid()}.tmp"
        logs = []
        src = os.path.join(sdir, "n6_capi.cu")
        cmds = _mxcc_cmds(src, sdir, tmp)   # the second entry is the older flag spelling
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
        raise RuntimeError("mxcc build of libn6_glue failed:\n" + "\n".join(logs)[-6000:])


VP, I32, I64, F32 = C.c_void_p, C.c_int, C.c_longlong, C.c_float
PI32 = C.POINTER(C.c_int)
_SIGS = {
    "n6_version": ([], I32),
    "n6_num_cfgs": ([I32], I32),
    "n6_cfg_info": ([I32, I32, PI32], I32),
    "n6_attr": ([I32, I32, PI32], I32),
    "n6_add_rms_norm": ([I32, VP, I64, VP, I64, VP, VP, I64, VP, I64, I32, I32, F32, VP], I32),
    "n6_rope_kv": ([I32, VP, I64, I64, VP, I64, I64, VP, I64, I64, VP, I32, I32, VP, I64,
                    I32, VP, I64, I64, VP, VP, I64, I64, I64, VP, I32, I32, I32, I32, I32,
                    I32, I32, I32, I32, VP], I32),
    "n6_rope_only": ([I32, VP, I64, I64, VP, I64, I64, VP, I32, I32, VP, I64, I32, VP,
                      I64, I64, VP, I64, I64, I32, I32, I32, I32, I32, VP], I32),
    "n6_silu_mul": ([I32, VP, I64, VP, I64, I32, I32, VP], I32),
}


def load(path: str):
    lib = C.CDLL(path)
    for name, (args, res) in _SIGS.items():
        f = getattr(lib, name)
        f.argtypes = args
        f.restype = res
    if lib.n6_version() // 100 != 2:   # 2: N6RopeParams carries ko / rope_only (R10/F)
        raise RuntimeError(f"libn6_glue version {lib.n6_version()}")
    return lib


def attr(lib, kernel: int, cfg: int) -> dict:
    a = (C.c_int * 5)()
    rc = lib.n6_attr(kernel, cfg, a)
    return {"rc": rc, "regs": a[0], "spill": a[1], "max_threads": a[2], "smem": a[3],
            "ctas_per_ap": a[4]}


def cfg_info(lib, kernel: int, cfg: int) -> dict:
    a = (C.c_int * 3)()
    if lib.n6_cfg_info(kernel, cfg, a) != 0:
        raise ValueError(f"no config {kernel}/{cfg}")
    return {"threads": a[0], "vpt": a[1], "n": a[2]}


def _check_tables(lib) -> str:
    """Check the pinned tables against the library and drop what it cannot run.

    A norm config covers exactly N = threads * VPT, so a table entry for another hidden
    size is rejected at launch (rc -11).  That is safe -- the caller falls back -- but it
    would silently cost the fusion for that size, and the start-up self-test only covers
    one hidden size, so every entry is checked here (one call per entry, CPU only).
    Dropping the entry also takes that hidden size out of the IR provider's
    supports_args, which is consulted before the self-test runs."""
    bad = []
    for n, cfg in list(NORM_CFG.items()):
        try:
            got = cfg_info(lib, K_NORM, cfg)["n"]
        except ValueError:
            got = None
        if got != n:
            del NORM_CFG[n]
            bad.append(f"norm N={n}: config {cfg} covers N={got}, entry dropped")
    for name, kernel, table in (("rope", K_ROPE, ROPE_CFG), ("silu", K_SILU, SILU_CFG)):
        num = lib.n6_num_cfgs(kernel)
        for max_n, cfg in table:
            c = cfg[0] if isinstance(cfg, tuple) else cfg
            if not 0 <= c < num:
                bad.append(f"{name} <= {max_n}: config {c} not in 0..{num - 1}")
    if bad:
        logger.warning("MetaX native glue: pinned table rejected: %s", "; ".join(bad))
    return "; ".join(bad)


# --------------------------------------------------------------------------- #
# launch (hot path: as few Python / torch calls as the arguments allow)
# --------------------------------------------------------------------------- #
_CUR = torch.cuda.current_stream


def _stream(t: torch.Tensor):
    return _CUR(t.device).cuda_stream if t.is_cuda else None


def _pick(table, n):
    for max_n, cfg in table:
        if n <= max_n:
            return cfg
    return table[-1][1]


def add_rms_norm(x, residual, weight, eps, out, residual_out, cfg=None, stream=-1) -> bool:
    """out = RMSNorm(x + residual) * weight, residual_out = x + residual (both bf16).

    2-D [M, N] tensors with unit stride in N; returns False when the library rejects
    the arguments (the caller then uses its fallback).  Nothing is allocated here.
    """
    n = x.shape[1]
    if cfg is None:
        cfg = NORM_CFG.get(n, -1)
    rc = _LIB.n6_add_rms_norm(
        cfg, x.data_ptr(), x.stride(0), residual.data_ptr(), residual.stride(0),
        weight.data_ptr(), out.data_ptr(), out.stride(0), residual_out.data_ptr(),
        residual_out.stride(0), x.shape[0], n, eps,
        _stream(x) if stream == -1 else stream)
    if rc == 0:
        return True
    return _reject("norm", rc)


def rope_kv(q, k, v, positions, cos_sin, q_out, key_cache, value_cache, slot_mapping,
            cfg=None, hpg=None, stream=-1) -> bool:
    """Rotate q (-> q_out) and k, write rotated k and raw v into the paged cache.

    q [T, HQ, D], k / v [T, HK, D] (any row / head stride, unit stride in D),
    positions [n] int32/int64, cos_sin [max_pos, D] bf16 (cos | sin halves),
    key_cache / value_cache [blocks, block_size, HK, D], slot_mapping [n] int32/int64.
    """
    t, hq, d = q.shape
    half = d >> 1
    if cfg is None:
        cfg, hpg = _pick(ROPE_CFG, t)
    bs = key_cache.shape[1]
    rc = _LIB.n6_rope_kv(
        cfg, q.data_ptr(), q.stride(0), q.stride(1), k.data_ptr(), k.stride(0), k.stride(1),
        v.data_ptr(), v.stride(0), v.stride(1), positions.data_ptr(), positions.shape[0],
        positions.dtype == torch.int64, cos_sin.data_ptr(), cos_sin.stride(0),
        cos_sin.shape[0], q_out.data_ptr(), q_out.stride(0), q_out.stride(1),
        key_cache.data_ptr(), value_cache.data_ptr(), key_cache.stride(0),
        key_cache.stride(1), key_cache.stride(2), slot_mapping.data_ptr(),
        min(slot_mapping.shape[0], t), slot_mapping.dtype == torch.int64, bs,
        bs.bit_length() - 1 if bs and not (bs & (bs - 1)) else -1, t, hq, k.shape[1], half,
        hpg if hpg else hq, _stream(q) if stream == -1 else stream)
    if rc == 0:
        return True
    return _reject("rope", rc)


def rope_only(q, k, positions, cos_sin, q_out, k_out, cfg=None, hpg=None,
              stream=-1) -> bool:
    """Rotate q (-> q_out) and k (-> k_out); write no cache.  q_out may be q and k_out
    may be k (every thread reads the two halves of its own 8 dims before writing them).

    q [T, HQ, D], k [T, HK, D] (any row / head stride, unit stride in D), positions [n]
    int32/int64, cos_sin [max_pos, D] bf16 (cos | sin halves).  Rows >= n rotate at
    position 0, like the fused kernel, so one captured graph serves every padded step.
    """
    t, hq, d = q.shape
    half = d >> 1
    if cfg is None:
        cfg, hpg = _pick(ROPE_CFG, t)
    rc = _LIB.n6_rope_only(
        cfg, q.data_ptr(), q.stride(0), q.stride(1), k.data_ptr(), k.stride(0), k.stride(1),
        positions.data_ptr(), positions.shape[0], positions.dtype == torch.int64,
        cos_sin.data_ptr(), cos_sin.stride(0), cos_sin.shape[0], q_out.data_ptr(),
        q_out.stride(0), q_out.stride(1), k_out.data_ptr(), k_out.stride(0),
        k_out.stride(1), t, hq, k.shape[1], half, hpg if hpg else hq,
        _stream(q) if stream == -1 else stream)
    if rc == 0:
        return True
    return _reject("rope_only", rc)


def silu_mul(x, out, cfg=None, stream=-1) -> bool:
    """out[m, i] = silu(x[m, i]) * x[m, I + i] for x [M, 2I], out [M, I]."""
    i = x.shape[1] >> 1
    if cfg is None:
        cfg = _pick(SILU_CFG, x.shape[0])
    rc = _LIB.n6_silu_mul(cfg, x.data_ptr(), x.stride(0), out.data_ptr(), out.stride(0),
                          x.shape[0], i, _stream(x) if stream == -1 else stream)
    if rc == 0:
        return True
    return _reject("silu", rc)


def _reject(what, rc):
    if rc >= 1000:
        raise RuntimeError(f"libn6_glue {what} launch failed rc={rc}")
    _REJECTS[(what, rc)] = _REJECTS.get((what, rc), 0) + 1
    return False


# --------------------------------------------------------------------------- #
# feature gates (host side; impl/mx_glue.py adds the step-type policy)
# --------------------------------------------------------------------------- #
def norm_supported(x, residual, weight) -> bool:
    return (NORM_OK and x.dtype == torch.bfloat16 and residual.dtype == x.dtype
            and weight.dtype == x.dtype and x.dim() == 2 and x.shape[1] in NORM_CFG
            and x.stride(1) == 1 and residual.stride(1) == 1 and weight.stride(0) == 1
            and x.shape[0] > 0)


def rope_supported(q, k, v, cos_sin, key_cache) -> bool:
    d = q.shape[-1]
    return (ROPE_OK and q.dtype == torch.bfloat16 and k.dtype == q.dtype
            and v.dtype == q.dtype and cos_sin.dtype == q.dtype
            and key_cache.dtype == q.dtype and q.dim() == 3 and k.dim() == 3 and v.dim() == 3
            and d == cos_sin.shape[-1] and d % 16 == 0 and (d >> 1) % 8 == 0
            and ((d >> 1) & ((d >> 1) - 1)) == 0
            and q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1
            and key_cache.dim() == 4 and key_cache.shape[2] == k.shape[1]
            and key_cache.shape[3] == d and key_cache.stride(3) == 1)


def rope_only_supported(q, k, cos_sin) -> bool:
    """The RoPE-only form: no v, no cache, so only q / k / cos_sin have to fit."""
    d = q.shape[-1]
    return (ROPE_ONLY_OK and q.dtype == torch.bfloat16 and k.dtype == q.dtype
            and cos_sin.dtype == q.dtype and q.dim() == 3 and k.dim() == 3
            and q.shape[0] == k.shape[0] and q.shape[0] > 0
            and d == k.shape[-1] and d == cos_sin.shape[-1] and d % 16 == 0
            and (d >> 1) % 8 == 0 and ((d >> 1) & ((d >> 1) - 1)) == 0
            and q.stride(2) == 1 and k.stride(2) == 1 and cos_sin.stride(1) == 1
            and q.shape[1] % k.shape[1] == 0)


def silu_supported(x) -> bool:
    return (SILU_OK and x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
            and x.shape[1] % 16 == 0)


# --------------------------------------------------------------------------- #
# set-up and self-test
# --------------------------------------------------------------------------- #
def ref_norm(x, r, w, eps):
    s = x.float() + r.float()
    y = s * torch.rsqrt((s * s).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype), s.to(x.dtype)


def ref_rope(x, pos, cs):
    half = x.shape[-1] // 2
    c = cs[pos.long()]
    cos, sin = c[:, None, :half].float(), c[:, None, half:].float()
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).to(x.dtype)


def self_test(device=None, n: int = 2048, rows=(1, 3, 64, 520)) -> tuple[bool, bool, bool, str]:
    """Run every pinned config once against a torch fp32 reference.  Returns
    (norm ok, rope ok, silu ok, message); also latches the module's ROPE_ONLY_OK."""
    global ROPE_ONLY_OK
    ROPE_ONLY_OK = False
    ro_ok = True
    dev = device or torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(7)
    msg = []
    ok = [True, True, True]
    eps = 1e-6
    # norm
    w = (torch.rand(n, device=dev, generator=g) + 0.5).to(torch.bfloat16)
    for m in rows:
        x = (torch.randn(m, n, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        r = (torch.randn(m, n, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        y = torch.empty_like(x)
        ro = torch.empty_like(x)
        if not add_rms_norm(x, r, w, eps, y, ro):
            ok[0] = False
            msg.append(f"norm rejected m={m}")
            break
        ry, rr = ref_norm(x, r, w, eps)
        if not torch.equal(ro, rr):
            ok[0] = False
            msg.append(f"norm residual differs m={m}")
        d = (y.float() - ry.float()).abs().max().item()
        scale = ry.float().abs().max().item() or 1.0
        if d > 8.0 * scale * 2 ** -8:
            ok[0] = False
            msg.append(f"norm y max err {d:.3g} m={m}")
    # rope + kv write
    hq, hk, dh, page = 16, 2, 128, 16
    maxpos = 4096
    inv = 1.0 / (5e6 ** (torch.arange(0, dh, 2, device=dev, dtype=torch.float32) / dh))
    f = torch.outer(torch.arange(maxpos, device=dev, dtype=torch.float32), inv)
    cs = torch.cat((f.cos(), f.sin()), -1).to(torch.bfloat16)
    for m in (1, 3, 64, 520):
        qkv = (torch.randn(m, (hq + 2 * hk) * dh, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        q = qkv[:, : hq * dh].view(m, hq, dh)
        k = qkv[:, hq * dh: (hq + hk) * dh].view(m, hk, dh)
        v = qkv[:, (hq + hk) * dh:].view(m, hk, dh)
        pos = torch.randint(0, maxpos, (m,), device=dev, generator=g, dtype=torch.int64)
        nb = (m + page - 1) // page + 2
        kc = torch.zeros(nb, page, hk, dh, device=dev, dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        slots = torch.randperm(nb * page, device=dev, generator=g)[:m].to(torch.int64)
        if m > 2:
            slots[1] = -1                                    # skipped token
        qo = torch.empty(m, hq, dh, device=dev, dtype=torch.bfloat16)
        if not rope_kv(q, k, v, pos, cs, qo, kc, vc, slots):
            ok[1] = False
            msg.append(f"rope rejected m={m}")
            break
        if not torch.equal(qo, ref_rope(q, pos, cs)):
            ok[1] = False
            msg.append(f"rope q differs m={m}")
        kr = ref_rope(k, pos, cs)
        keep = slots >= 0
        kb, ko = slots[keep] // page, slots[keep] % page
        if not torch.equal(kc[kb, ko], kr[keep]) or not torch.equal(vc[kb, ko], v[keep]):
            ok[1] = False
            msg.append(f"kv cache differs m={m}")
        if m > 2 and (kc.sum(dtype=torch.float32) == 0):
            ok[1] = False
            msg.append("kv cache not written")
        # R10/F: the RoPE-only form -- same rotation, no cache write -- out of place and
        # then aliasing its own (strided) inputs, which is how the INT8 path calls it.
        qo2 = torch.empty(m, hq, dh, device=dev, dtype=torch.bfloat16)
        ko2 = torch.empty(m, hk, dh, device=dev, dtype=torch.bfloat16)
        if not rope_only(q, k, pos, cs, qo2, ko2):
            ro_ok = False
            msg.append(f"rope_only rejected m={m}")
        else:
            if not torch.equal(qo2, ref_rope(q, pos, cs)):
                ro_ok = False
                msg.append(f"rope_only q differs m={m}")
            if not torch.equal(ko2, ref_rope(k, pos, cs)):
                ro_ok = False
                msg.append(f"rope_only k differs m={m}")
            q2 = qkv.clone()
            qa = q2[:, : hq * dh].view(m, hq, dh)
            ka = q2[:, hq * dh: (hq + hk) * dh].view(m, hk, dh)
            va = q2[:, (hq + hk) * dh:].view(m, hk, dh)
            if not rope_only(qa, ka, pos, cs, qa, ka):
                ro_ok = False
                msg.append(f"rope_only in-place rejected m={m}")
            elif not (torch.equal(qa, qo2) and torch.equal(ka, ko2)
                      and torch.equal(va, v)):
                ro_ok = False
                msg.append(f"rope_only in place differs m={m}")
    # silu
    for m in (1, 64, 520):
        x = (torch.randn(m, 2 * 6144, device=dev, generator=g)).to(torch.bfloat16)
        out = torch.empty(m, 6144, device=dev, dtype=torch.bfloat16)
        if not silu_mul(x, out):
            ok[2] = False
            msg.append(f"silu rejected m={m}")
            break
        gg = x[:, :6144].float()
        ref = (gg * torch.sigmoid(gg) * x[:, 6144:].float()).to(torch.bfloat16)
        d = (out.float() - ref.float()).abs().max().item()
        if d > 4.0 * (ref.float().abs().max().item() or 1.0) * 2 ** -8:
            ok[2] = False
            msg.append(f"silu max err {d:.3g} m={m}")
    ROPE_ONLY_OK = bool(ok[1] and ro_ok)
    return ok[0], ok[1], ok[2], "; ".join(msg) or "all ok"


def setup(build_only: bool = False, device=None) -> bool:
    """Build / load the library and (unless build_only) self-test it on the device.

    build_only is the start-up path (mxcc is a CPU compiler, so it can run before any GPU
    context exists, e.g. while the platform builds its IR op priorities); it leaves the
    self-test to the later full setup() in the worker, so it must NOT mark the set-up
    done.  Idempotent; never raises."""
    global _LIB, _BUILD_DONE, _SETUP_DONE, ENABLED, NORM_OK, ROPE_OK, SILU_OK, STATUS, LIB_PATH
    global ROPE_ONLY_OK
    if _SETUP_DONE:
        return ENABLED
    if not REQUESTED:
        STATUS = f"off ({ENV_FLAG}=0)"
        _BUILD_DONE = _SETUP_DONE = True
        return False
    if not _BUILD_DONE:
        _BUILD_DONE = True
        try:
            t0 = time.time()
            LIB_PATH, blog = build()
            _LIB = load(LIB_PATH)
            bad = _check_tables(_LIB)
            STATUS = (f"loaded {os.path.basename(LIB_PATH)} "
                      f"({blog.splitlines()[0][:40]}) in {time.time() - t0:.1f}s"
                      + (f"; table rejected: {bad}" if bad else ""))
        except Exception as e:  # noqa: BLE001
            STATUS = f"unavailable: {e}"
            _SETUP_DONE = True
            logger.warning("MetaX native glue: %s", STATUS)
            return False
    if _LIB is None:
        _SETUP_DONE = True
        return False
    if build_only:
        return True
    NORM_OK = ROPE_OK = SILU_OK = True
    try:
        NORM_OK, ROPE_OK, SILU_OK, msg = self_test(device)
        STATUS += f"; self-test {msg}"
    except Exception as e:  # noqa: BLE001
        NORM_OK = ROPE_OK = SILU_OK = ROPE_ONLY_OK = False
        STATUS += f"; self-test failed: {e}"
    ENABLED = NORM_OK or ROPE_OK or SILU_OK
    _SETUP_DONE = True
    logger.info("MetaX native glue (R7/N6): %s (norm=%s rope=%s rope_only=%s silu=%s)",
                STATUS, NORM_OK, ROPE_OK, ROPE_ONLY_OK, SILU_OK)
    return ENABLED


def summary() -> dict:
    return {"status": STATUS, "lib": LIB_PATH, "norm": NORM_OK, "rope": ROPE_OK,
            "rope_only": ROPE_ONLY_OK, "silu": SILU_OK, "rejects": dict(_REJECTS)}


if __name__ == "__main__":  # python3 mx_glue_native.py [--build]
    import sys

    path, log = build(force="--force" in sys.argv)
    print(path)
    print(log[-1500:])

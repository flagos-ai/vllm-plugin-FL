"""Self-developed INT8 KV cache for the opt-in GQA hybrid backend.

Enabled by the worker model profile or explicit ``VLLM_FL_KV_INT8=1``.
Explicit ``VLLM_FL_KV_INT8=0`` overrides the profile. The serve-level
``--kv-cache-dtype`` stays ``auto``; the plugin changes only its own KV spec.

Page layout (one 16-token block of one layer, 8320 bytes = 65 x 128):

* bytes [0, 8192): int8 data viewed as ``(2 kv, 16 slot, 2 head, 128 dim)``,
  i.e. the standard ``(num_blocks, 2, 16, 2, 128)`` cache shape;
* bytes [8192, 8320): fp16 scales ``(2 kv, 2 head, 16 slot)``. One KV head's
  16 K (or V) scales are one contiguous 32-byte sector.

Quantization is symmetric absmax per (token, KV head): ``s = fp16(max(absmax /
127, 1e-6))`` and ``q = clamp(round_half_away(x / s), -127, 127)``, where the
division uses the stored fp16 scale so that ``q * s`` is the exact dequantized
value. The fast decode path factors K scales into scores and V scales into P,
then uses a transposed PV dot. The fallback reconstructs BF16 K/V and performs
QK and PV in FP32.
"""

import functools
import os
from contextlib import contextmanager

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

# Under the configured "vllm" namespace so INFO lines reach the server log.
logger = init_logger("vllm.vllm_fl.kv_int8")

PAGE_TOKENS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
DATA_BYTES = 2 * PAGE_TOKENS * NUM_KV_HEADS * HEAD_DIM  # 8192
SCALE_BYTES = 2 * NUM_KV_HEADS * PAGE_TOKENS * 2  # 128
PAGE_BYTES = DATA_BYTES + SCALE_BYTES  # 8320
SPARSE_METADATA_BYTES = 1024
SPARSE_PAGE_BYTES = PAGE_BYTES + SPARSE_METADATA_BYTES  # 9344
SCALE_EPS = 1.0e-6
# fp16 element offsets inside one page.
PAGE_HALFS = PAGE_BYTES // 2  # 4160
SCALE_BASE_HALFS = DATA_BYTES // 2  # 4096


def kv_int8_enabled() -> bool:
    return os.environ.get("VLLM_FL_KV_INT8", "0").lower() in ("1", "true", "yes")


def is_int8_layout(kv_cache: torch.Tensor) -> bool:
    if (
        kv_cache.dtype != torch.int8
        or kv_cache.dim() != 5
        or tuple(kv_cache.shape[1:]) != (2, PAGE_TOKENS, NUM_KV_HEADS, HEAD_DIM)
        or kv_cache.stride()[1:] != (DATA_BYTES // 2, 256, 128, 1)
    ):
        return False
    return kv_cache.stride(0) in (PAGE_BYTES, SPARSE_PAGE_BYTES)


def page_stride_bytes(kv_cache: torch.Tensor) -> int:
    """Return the validated physical stride of an INT8 physical page.

    The prefill dequantizer receives the 4D K view of a full cache, while the
    write and decode paths receive its 5D K/V view.
    """
    full_cache = is_int8_layout(kv_cache)
    key_view = (
        kv_cache.dtype == torch.int8
        and kv_cache.dim() == 4
        and tuple(kv_cache.shape[1:]) == (PAGE_TOKENS, NUM_KV_HEADS, HEAD_DIM)
        and kv_cache.stride()[1:] == (256, 128, 1)
        and kv_cache.stride(0) in (PAGE_BYTES, SPARSE_PAGE_BYTES)
    )
    if not (full_cache or key_view):
        raise ValueError("expected the INT8 page layout with 8320- or 9344-byte pages")
    return int(kv_cache.stride(0))


def scale_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """fp16 view ``(num_blocks, 2 kv, 2 head, 16 slot)`` of the page scales."""
    if not is_int8_layout(kv_cache):
        raise ValueError("expected the INT8 page layout with 8320- or 9344-byte pages")
    offset = kv_cache.storage_offset()
    if offset % 2:
        raise ValueError("INT8 KV page base must be 2-byte aligned")
    base = torch.empty(0, dtype=torch.float16, device=kv_cache.device).set_(
        kv_cache.untyped_storage()
    )
    return torch.as_strided(
        base,
        size=(kv_cache.shape[0], 2, NUM_KV_HEADS, PAGE_TOKENS),
        stride=(
            page_stride_bytes(kv_cache) // 2,
            NUM_KV_HEADS * PAGE_TOKENS,
            PAGE_TOKENS,
            1,
        ),
        storage_offset=offset // 2 + SCALE_BASE_HALFS,
    )


# --------------------------------------------------------------------------
# Write path
# --------------------------------------------------------------------------
@triton.jit
def _quantize_row(x, eps: tl.constexpr):
    # x: (H, 128) fp32. Returns int8 values and fp16 scales (H,).
    absmax = tl.max(tl.abs(x), 1)
    s16 = tl.maximum(absmax / 127.0, eps).to(tl.float16)
    s = s16.to(tl.float32)
    y = tl.math.div_rn(x, s[:, None])
    y = tl.where(y >= 0, tl.math.floor(y + 0.5), -tl.math.floor(0.5 - y))
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    return y.to(tl.int8), s16


@triton.jit
def _kv_int8_write(
    Key,
    Value,
    Cache,
    Scales,
    Slots,
    ks0,
    ks1,
    vs0,
    vs1,
    PAGE_STRIDE: tl.constexpr,
    PAGE_HALF_STRIDE: tl.constexpr,
    SCALE_BASE: tl.constexpr,
    V_OFFSET: tl.constexpr,
    PAGE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    tok = tl.program_id(0)
    slot = tl.load(Slots + tok).to(tl.int64)
    if slot < 0:
        return
    page = slot // PAGE
    off = slot % PAGE
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    tok64 = tok.to(tl.int64)
    k = tl.load(Key + tok64 * ks0 + h[:, None] * ks1 + d[None, :]).to(tl.float32)
    v = tl.load(Value + tok64 * vs0 + h[:, None] * vs1 + d[None, :]).to(tl.float32)
    kq, kscale = _quantize_row(k, EPS)
    vq, vscale = _quantize_row(v, EPS)
    dst = Cache + page * PAGE_STRIDE + off * (H * D) + h[:, None] * D + d[None, :]
    tl.store(dst, kq)
    tl.store(dst + V_OFFSET, vq)
    sdst = Scales + page * PAGE_HALF_STRIDE + SCALE_BASE + h * PAGE + off
    tl.store(sdst, kscale)
    tl.store(sdst + H * PAGE, vscale)


def kv_int8_write(key, value, kv_cache, slot_mapping):
    """Quantize ``key``/``value`` (T, 2, 128) into the INT8 paged cache."""
    if not is_int8_layout(kv_cache):
        raise ValueError("expected the INT8 page layout")
    if key.shape[1:] != (NUM_KV_HEADS, HEAD_DIM) or value.shape != key.shape:
        raise ValueError("INT8 KV write supports 2 KV heads x 128 only")
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("last dimension must be contiguous")
    n = min(key.shape[0], slot_mapping.shape[0])
    if n == 0:
        return
    if kv_cache.storage_offset() % 2:
        raise ValueError("INT8 KV page base must be 2-byte aligned")
    page_stride = page_stride_bytes(kv_cache)
    half = half_base(kv_cache)
    _kv_int8_write[(n,)](
        key,
        value,
        kv_cache,
        half,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        page_stride,
        page_stride // 2,
        SCALE_BASE_HALFS,
        DATA_BYTES // 2,
        PAGE_TOKENS,
        NUM_KV_HEADS,
        HEAD_DIM,
        SCALE_EPS,
        num_warps=1,
    )


# --------------------------------------------------------------------------
# Decode read path
# --------------------------------------------------------------------------
@triton.jit
def _int8_to_bf16(x):
    # XOR builds 2^23 + (x + 128); subtracting 2^23 + 128 recovers exact x.
    u = x.to(tl.uint8, bitcast=True).to(tl.uint32) ^ 0x4B000080
    return (u.to(tl.float32, bitcast=True) - 8388736.0).to(tl.bfloat16)


@triton.jit
def _int8_decode(
    Q,
    Cache,
    Scales,
    Out,
    Cu,
    Lens,
    BT,
    Partial,
    Stats,
    qk_scale,
    q0,
    q1,
    o0,
    o1,
    bts,
    PAGE_STRIDE: tl.constexpr,
    PAGE_HALF_STRIDE: tl.constexpr,
    SCALE_BASE: tl.constexpr,
    V_OFFSET: tl.constexpr,
    PAGE: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    VARIANT: tl.constexpr,
):
    # Variant 3 is the precision fallback; variant 8 is the factored fast path.
    seq = tl.program_id(0)
    head = tl.program_id(1)
    seg = tl.program_id(2)
    start = tl.load(Cu + seq)
    qlen = tl.load(Cu + seq + 1) - start
    length = tl.load(Lens + seq)
    if qlen != 1:
        length = 0
    tiles = tl.cdiv(length, N)
    per = tl.cdiv(tiles, S)
    t0 = seg * per
    t1 = tl.minimum(t0 + per, tiles)
    R: tl.constexpr = 8 if VARIANT == 3 else 16
    r = tl.arange(0, R)
    d = tl.arange(0, 128)
    n = tl.arange(0, N)
    pg = tl.arange(0, N // PAGE)
    sl = tl.arange(0, PAGE)
    rv = r < 8
    q = tl.load(
        Q + start.to(tl.int64) * q0 + (head * 8 + r[:, None]) * q1 + d[None, :],
        rv[:, None] & (length > 0),
        0.0,
    )
    if VARIANT == 3:
        q = q.to(tl.float32)
    mx = tl.full((R,), -float("inf"), tl.float32)
    den = tl.zeros((R,), tl.float32)
    if VARIANT == 8:
        acc = tl.zeros((128, R), tl.float32)
    else:
        acc = tl.zeros((R, 128), tl.float32)
    bt_row = BT + seq.to(tl.int64) * bts
    for tile in range(t0, t1):
        pos = tile * N + n
        inside = pos < length
        # One page-table lookup per physical page; slots of a page are
        # contiguous in data (256 B per slot) and scales (16 halves per head).
        logical = tile * (N // PAGE) + pg
        pages = tl.load(bt_row + logical, logical * PAGE < length, 0).to(tl.int64)
        spos = tile * N + (pg[:, None] * PAGE + sl[None, :])
        sin = spos < length
        sp = (
            Scales
            + pages[:, None] * PAGE_HALF_STRIDE
            + (SCALE_BASE + head * PAGE)
            + sl[None, :]
        )
        ks = tl.reshape(tl.load(sp, sin, 0.0), (N,)).to(tl.float32)
        vs = tl.reshape(tl.load(sp + 2 * PAGE, sin, 0.0), (N,)).to(tl.float32)
        page = tl.reshape(tl.broadcast_to(pages[:, None], (N // PAGE, PAGE)), (N,))
        row = (
            Cache
            + page[:, None] * PAGE_STRIDE
            + (n % PAGE)[:, None] * 256
            + head * 128
            + d[None, :]
        )
        k_bf16 = _int8_to_bf16(tl.load(row, inside[:, None], 0))
        if VARIANT == 3:
            kreal = (k_bf16.to(tl.float32) * ks[:, None]).to(tl.bfloat16)
            kf = kreal.to(tl.float32)
            scores = tl.zeros((R, N), tl.float32)
            for i in tl.static_range(8):
                qi = tl.sum(tl.where(r[:, None] == i, q, 0.0), 0)
                si = tl.sum(kf * qi[None, :], 1)
                scores = tl.where(r[:, None] == i, si[None, :], scores)
            scores = scores * qk_scale
        else:
            scores = tl.dot(q, tl.trans(k_bf16))
            scores = scores * (ks * qk_scale)[None, :]
        scores = tl.where(inside[None, :], scores, -float("inf"))
        nm = tl.maximum(mx, tl.max(scores, 1))
        alpha = tl.math.exp2(mx - nm)
        p = tl.math.exp2(scores - nm[:, None])
        den = den * alpha + tl.sum(p, 1)
        v_bf16 = _int8_to_bf16(tl.load(row + V_OFFSET, inside[:, None], 0))
        if VARIANT == 8:
            pvs = (p * vs[None, :]).to(tl.bfloat16)
            pv = tl.dot(tl.trans(v_bf16), tl.trans(pvs))
            acc = acc * alpha[None, :] + pv
        else:
            vreal = (v_bf16.to(tl.float32) * vs[:, None]).to(tl.bfloat16)
            vf = vreal.to(tl.float32)
            pv = tl.zeros((R, 128), tl.float32)
            for i in tl.static_range(8):
                pi = tl.sum(tl.where(r[:, None] == i, p, 0.0), 0)
                oi = tl.sum(vf * pi[:, None], 0)
                pv = tl.where(r[:, None] == i, oi[None, :], pv)
            acc = acc * alpha[:, None] + pv
        mx = nm
    if VARIANT == 8:
        acc = tl.trans(acc)
    if S == 1:
        if length > 0:
            res = acc / tl.maximum(den[:, None], 1.0e-30)
            tl.store(
                Out
                + start.to(tl.int64) * o0
                + (head * 8 + r[:, None]) * o1
                + d[None, :],
                res,
                rv[:, None],
            )
    else:
        orow = ((seq * 2 + head) * S + seg).to(tl.int64) * 8 + r
        tl.store(Partial + orow[:, None] * 128 + d[None, :], acc, rv[:, None])
        tl.store(Stats + orow * 2, mx, rv)
        tl.store(Stats + orow * 2 + 1, den, rv)


@triton.jit
def _int8_decode_reduce(
    Out,
    Cu,
    Lens,
    Partial,
    Stats,
    o0,
    o1,
    S: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    start = tl.load(Cu + seq)
    qlen = tl.load(Cu + seq + 1) - start
    length = tl.load(Lens + seq)
    if qlen != 1 or length <= 0:
        return
    r = tl.arange(0, 8)
    d = tl.arange(0, 128)
    mx = tl.full((8,), -float("inf"), tl.float32)
    den = tl.zeros((8,), tl.float32)
    acc = tl.zeros((8, 128), tl.float32)
    base = ((seq * 2 + head) * S).to(tl.int64) * 8
    for s in range(S):
        row = base + s * 8 + r
        m = tl.load(Stats + row * 2)
        l = tl.load(Stats + row * 2 + 1)
        a = tl.load(Partial + row[:, None] * 128 + d[None, :])
        nm = tl.maximum(mx, m)
        safe = tl.where(nm == -float("inf"), 0.0, nm)
        w0 = tl.math.exp2(mx - safe)
        w1 = tl.math.exp2(m - safe)
        acc = acc * w0[:, None] + a * w1[:, None]
        den = den * w0 + l * w1
        mx = nm
    res = acc / tl.maximum(den[:, None], 1.0e-30)
    tl.store(
        Out + start.to(tl.int64) * o0 + (head * 8 + r[:, None]) * o1 + d[None, :], res
    )


LOG2E = 1.4426950408889634


@functools.cache
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def decode_segments(num_seqs: int, num_sms=None) -> int:
    """Device-aware fallback; tuned tables remain static under CUDA Graphs."""
    if num_sms is None:
        num_sms = _sm_count(torch.cuda.current_device())
    target = 4 * max(1, num_sms)
    splits = 1
    while num_seqs * NUM_KV_HEADS * splits < target and splits < 64:
        splits *= 2
    return splits


class DecodeWorkspace:
    """Fixed buffers for split decode; allocated before graph capture."""

    def __init__(self, device, max_units: int = 512):
        # Units are (request, segment) pairs; decode_segments keeps
        # num_seqs * S below 512 for every batch size (about 4 MB in total).
        self.max_units = max_units
        rows = max_units * NUM_KV_HEADS * 8
        self.partial = torch.empty((rows, 128), dtype=torch.float32, device=device)
        self.stats = torch.empty((rows, 2), dtype=torch.float32, device=device)


def kv_int8_decode(
    q,
    kv_cache,
    scales,
    out,
    cu_seqlens_q,
    seq_lens,
    block_table,
    softmax_scale,
    workspace,
    *,
    segments=None,
    tile=32,
    num_warps=4,
    num_stages=2,
    variant=None,
    config=None,
):
    """Single-token decode attention over the INT8 cache (16Q:2KV:d128)."""
    num_seqs = seq_lens.shape[0]
    if num_seqs == 0:
        return out
    page_stride = page_stride_bytes(kv_cache)
    if config is not None:
        tile, num_warps, num_stages = config.tile, config.num_warps, config.num_stages
        variant = config.variant if variant is None else variant
        if segments is None:
            segments = config.lookup(num_seqs)
    variant = 8 if variant is None else variant
    if variant not in (3, 8):
        raise ValueError(f"unknown INT8 decode variant {variant}")
    s = (
        decode_segments(num_seqs, _sm_count(q.device.index or 0))
        if segments is None
        else segments
    )
    if s < 1 or s & (s - 1) or tile < 16 or tile & (tile - 1):
        raise ValueError("INT8 decode splits/tile must be powers of two (tile >= 16)")
    if s > 1 and num_seqs * s > workspace.max_units:
        s = max(1, workspace.max_units // num_seqs)
        s = 1 << (s.bit_length() - 1)
    half = scales
    if half.data_ptr() != kv_cache.data_ptr() or half.dtype != torch.float16:
        raise ValueError("INT8 decode needs the fp16 scale base of this cache")
    grid = (num_seqs, NUM_KV_HEADS, s)
    _int8_decode[grid](
        q,
        kv_cache,
        half,
        out,
        cu_seqlens_q,
        seq_lens,
        block_table,
        workspace.partial,
        workspace.stats,
        softmax_scale * LOG2E,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        block_table.stride(0),
        page_stride,
        page_stride // 2,
        SCALE_BASE_HALFS,
        DATA_BYTES // 2,
        PAGE_TOKENS,
        tile,
        s,
        variant,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if s > 1:
        _int8_decode_reduce[(num_seqs, NUM_KV_HEADS)](
            out,
            cu_seqlens_q,
            seq_lens,
            workspace.partial,
            workspace.stats,
            out.stride(0),
            out.stride(1),
            s,
            num_warps=4,
        )
    return out


def half_base(kv_cache: torch.Tensor) -> torch.Tensor:
    """fp16 pointer base aliasing the INT8 cache storage at the cache origin."""
    storage = kv_cache.untyped_storage()
    offset = kv_cache.storage_offset() // 2
    # A non-empty view: zero-size tensors report a null data pointer.
    return torch.empty(0, dtype=torch.float16, device=kv_cache.device).set_(
        storage, offset, (storage.nbytes() // 2 - offset,), (1,)
    )


# --------------------------------------------------------------------------
# Prefill support: dequantize prefill requests' pages once per layer
# --------------------------------------------------------------------------
@triton.jit
def _kv_int8_dequant_pages(
    Cache,
    Scales,
    BT,
    Work,
    Dst,
    bts,
    PAGE_STRIDE: tl.constexpr,
    PAGE_HALF_STRIDE: tl.constexpr,
    SCALE_BASE: tl.constexpr,
    PAGE: tl.constexpr,
):
    # One program per scratch page; Work rows are (request, logical page).
    j = tl.program_id(0)
    seq = tl.load(Work + j * 2)
    lp = tl.load(Work + j * 2 + 1)
    page = tl.load(BT + seq.to(tl.int64) * bts + lp).to(tl.int64)
    rows = tl.arange(0, 4 * PAGE)  # (kv, slot, head) in storage order
    d = tl.arange(0, 128)
    x = tl.load(Cache + page * PAGE_STRIDE + rows[:, None] * 128 + d[None, :])
    kv = rows // (2 * PAGE)
    slot = (rows // 2) % PAGE
    head = rows % 2
    s = tl.load(
        Scales
        + page * PAGE_HALF_STRIDE
        + SCALE_BASE
        + kv * (2 * PAGE)
        + head * PAGE
        + slot
    )
    y = x.to(tl.float32) * s.to(tl.float32)[:, None]
    tl.store(
        Dst + j.to(tl.int64) * (4 * PAGE * 128) + rows[:, None] * 128 + d[None, :],
        y.to(Dst.dtype.element_ty),
    )


def kv_int8_dequant_pages(kv_cache, scales_half, block_table, work, dst):
    """dst: (P, 2, 16, 2, 128) BF16 scratch; work: (P, 2) int32 (request, logical page)."""
    n = work.shape[0]
    if n == 0:
        return
    page_stride = page_stride_bytes(kv_cache)
    _kv_int8_dequant_pages[(n,)](
        kv_cache,
        scales_half,
        block_table,
        work,
        dst,
        block_table.stride(0),
        page_stride,
        page_stride // 2,
        SCALE_BASE_HALFS,
        PAGE_TOKENS,
        num_warps=4,
    )


SCRATCH_PAGE_SHAPE = (2, PAGE_TOKENS, NUM_KV_HEADS, HEAD_DIM)  # BF16 dequantized page
_prefill_scratch: dict[torch.device, torch.Tensor] = {}


@contextmanager
def temporary_prefill_scratch(device):
    """Keep startup probes from reserving scratch before memory profiling."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    previous = _prefill_scratch.get(device)
    try:
        yield
    finally:
        if previous is None:
            _prefill_scratch.pop(device, None)
        else:
            _prefill_scratch[device] = previous


def prefill_scratch(device, cap_tokens: int) -> torch.Tensor:
    """Shared BF16 prefill scratch of ``cap_tokens`` tokens, allocated once per device.

    The attention impl reserves it during the profiling run, so the KV cache
    budget already excludes it and the first real prefill cannot OOM on it.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    pages = cap_tokens // PAGE_TOKENS
    buf = _prefill_scratch.get(device)
    if buf is None or buf.shape[0] < pages:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("INT8 KV prefill scratch growth during capture")
        buf = torch.empty(
            (pages, *SCRATCH_PAGE_SHAPE), dtype=torch.bfloat16, device=device
        )
        _prefill_scratch[device] = buf
    return buf


def prefill_scratch_cap_tokens() -> int:
    """Per-layer BF16 scratch cap (tokens); 0 disables the scratch path."""
    return int(os.environ.get("VLLM_FL_KV_INT8_PREFILL_SCRATCH_TOKENS", "65536"))

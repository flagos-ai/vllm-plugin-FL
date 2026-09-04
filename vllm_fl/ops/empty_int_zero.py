# Copyright (c) 2026 BAAI. All rights reserved.
"""aten::empty override that zero-fills only integer/bool tensors.

Background: FlagGems used to override aten::empty.memory_format with a
triton kernel that also zero-filled the allocation (~3000 launches and 6%
of decode GPU time on DeepSeek-V4/PPU). Upstream PR #5438 unregistered it,
which exposes consumers in the DeepSeek-V4 path that read partially-written
buffers and silently relied on that zero-fill — the garbage reaches topk
indices and faults the sparse-attention gather.

Zeroing only integer/bool dtypes keeps index/offset buffers safe (they are
the ones that turn garbage into out-of-bounds accesses) while leaving the
much larger float activation buffers truly uninitialized, so most of the
kernel launches are still avoided.

Set VLLM_FL_EMPTY_INT_ZERO=0 to disable.
"""

import os

import torch

_ALL_INT_DTYPES = {
    torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool,
}

# Only these dtypes are used for indices/offsets, i.e. the ones that turn
# uninitialised memory into an out-of-bounds access. The DeepSeek-V4 consumer
# that actually faults without this override is the KV compressor triton kernel
# (deepseek_v4_compress_cache.py), which reads int32 block_table, int64
# slot_mapping and token_to_req_indices. int8/uint8 buffers here are quantised
# activations: garbage in them is a numerical wrong answer at worst, never an
# IMA, and they are the bulk of the allocations.
#
# FlagGems overrides aten::zero_ with a triton kernel rather than a memset, so
# every zero-filled buffer costs a full kernel launch: profiling decode at batch
# 64 showed 1219 zeros_kernel launches/step = 1.19 ms = 3.4% of GPU time.
_INDEX_DTYPES = {torch.int32, torch.int64}

_DTYPE_SETS = {"all": _ALL_INT_DTYPES, "index": _INDEX_DTYPES}

# "index" (default) zeroes only index/offset dtypes; "all" restores the original
# behaviour of zeroing every integer/bool allocation.
# Default "all" keeps the original conservative behaviour. "index" narrows to
# index/offset dtypes only, which removed 365 of 1219 zeros_kernel launches per
# decode step (1.19 -> 0.78 ms of GPU kernel time) but measured no throughput
# change (+0.13%, within noise): those tiny kernels sit inside existing idle
# bubbles rather than on the critical path. Kept as a knob, not the default,
# because the safety argument for int8 buffers rests on only light validation.
_INT_DTYPES = _DTYPE_SETS.get(
    os.environ.get("VLLM_FL_EMPTY_ZERO_DTYPES", "all").lower(), _ALL_INT_DTYPES
)


def _contiguous_strides(shape):
    """Row-major strides for `shape`, matching torch's contiguous_format.

    Avoids allocating a meta tensor just to read .stride(). Verified identical to
    the meta path for 1-D, N-D, size-1 dims, and empty shapes. Only valid for
    contiguous_format, so callers must fall back for other memory formats.
    """
    n = len(shape)
    strides = [1] * n
    acc = 1
    for i in range(n - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


def _empty_int_zero(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False

    shape = tuple(size)
    # This override sits on aten::empty for the whole process, so every
    # torch.empty pays its cost. Profiling DeepSeek-V4 decode showed that
    # mattered: aten::empty at 1.07 ms/step of host time vs 0.10 on a stack
    # without the override. Computing contiguous strides directly instead of
    # round-tripping through a meta tensor cuts ~25-42% of the wrapper cost
    # (int32 9.98 -> 7.41 us, bf16 5.27 -> 3.05 us measured in isolation).
    if memory_format is None or memory_format is torch.contiguous_format:
        strides = _contiguous_strides(shape)
    else:
        # channels_last et al: let torch work the strides out.
        strides = torch.empty(
            shape, dtype=dtype, device="meta", memory_format=memory_format
        ).stride()

    out = torch.empty_strided(
        shape,
        strides,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if dtype in _INT_DTYPES and out.numel() > 0:
        if _AUDIT:
            _record_site(shape, dtype)
        out.zero_()
    return out


_AUDIT = os.environ.get("VLLM_FL_EMPTY_AUDIT") == "1"
_sites: dict = {}


def _record_site(shape, dtype) -> None:
    """Record the first non-torch caller for each integer empty() allocation."""
    import traceback

    frames = traceback.extract_stack()
    site = None
    for fr in reversed(frames[:-2]):
        if "/site-packages/torch/" in fr.filename or fr.filename.endswith(
            "empty_int_zero.py"
        ):
            continue
        site = f"{fr.filename}:{fr.lineno} in {fr.name}"
        break
    key = (site, tuple(shape), str(dtype))
    is_new = key not in _sites
    _sites[key] = _sites.get(key, 0) + 1
    if is_new:
        # Flush immediately: unique sites are few, and signal-based teardown
        # does not reliably run atexit handlers.
        try:
            dump_audit()
        except Exception:
            pass


def dump_audit(path: str | None = None) -> None:
    if path is None:
        path = f"/tmp/fl_empty_audit_{os.getpid()}.txt"
    if not _sites:
        return
    with open(path, "w") as f:
        for (site, shape, dt), n in sorted(_sites.items(), key=lambda kv: -kv[1]):
            f.write(f"{n:8d}  {dt:<14} {str(shape):<24} {site}\n")


if _AUDIT:
    import atexit

    atexit.register(dump_audit)


_lib = None


def register() -> bool:
    """Register the override on the CUDA dispatch key. Returns True if done."""
    global _lib
    if _lib is not None:
        return False
    if os.environ.get("VLLM_FL_EMPTY_INT_ZERO", "1") != "1":
        return False
    _lib = torch.library.Library("aten", "IMPL")
    _lib.impl("empty.memory_format", _empty_int_zero, "CUDA")
    print(
        "[vllm_fl] empty() int zero-fill active, dtypes="
        + ",".join(sorted(str(d).replace("torch.", "") for d in _INT_DTYPES)),
        flush=True,
    )
    return True

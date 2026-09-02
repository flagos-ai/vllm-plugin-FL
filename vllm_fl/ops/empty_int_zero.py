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

_INT_DTYPES = {
    torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool,
}


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
    if memory_format is None:
        memory_format = torch.contiguous_format

    shape = tuple(size)
    meta = torch.empty(shape, dtype=dtype, device="meta", memory_format=memory_format)
    out = torch.empty_strided(
        shape,
        meta.stride(),
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
    return True

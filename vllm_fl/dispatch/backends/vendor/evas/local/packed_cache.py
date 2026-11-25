# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for KV caches that live inside a shared backing allocation.

Plugin-side copy of FlagGems ``_evas/fused/_packed_cache.py`` (see
``local/__init__.py``); only the symbols the plugin consumes are carried —
``scatter_rows_into_host`` and ``gather_pages_to_device`` stay behind, uncalled
here. A layer's cache over a packed pool is a window on the whole backing:
read with ``host_view_of_storage``, write by mutating that host copy and
pushing it back with ``write_back_from_host``, judge size by
``cache_footprint_bytes``.

"""

import logging
import os

import torch

# Above this, a cache is served from host memory on the LPU backend.  Override
# with VLLM_FL_KV_CPU_DEQUANT_THRESHOLD (bytes).
VIEW_READ_THRESHOLD = int(
    os.environ.get("VLLM_FL_KV_CPU_DEQUANT_THRESHOLD", str(3 << 29))  # 1.5 GiB
)

logger = logging.getLogger(__name__)


def cache_footprint_bytes(t: torch.Tensor) -> int:
    """Bytes a read of ``t`` would be charged on the LPU device.

    Sizes by the BASE storage, not the window: LPU stages the backing, so a
    per-layer slice of a packed pool costs the pool.  ``untyped_storage()``
    reports the base for LPU views too (probe 2026-09-30) and raises on exotic
    impls -- fall back to the view size there (pre-0.25 behaviour).
    """
    view_bytes = t.numel() * t.element_size()
    try:
        return max(view_bytes, t.untyped_storage().nbytes())
    except Exception:  # pragma: no cover -- exotic impls without storage()
        return view_bytes


def is_oversized(t: torch.Tensor, threshold: int = None) -> bool:
    """True when ``t`` must not be read on the device (see module docstring)."""
    if threshold is None:
        threshold = VIEW_READ_THRESHOLD
    return cache_footprint_bytes(t) > threshold


def storage_flat(t: torch.Tensor) -> torch.Tensor:
    """1-D tensor over the WHOLE backing storage of ``t``, in ``t``'s OWN dtype.

    Reading or writing it moves the entire allocation and nothing else -- the
    only transient-free route on the LPU backend (see module docstring).

    The view is built in ``t.dtype`` on purpose.  A dtype-reinterpreted ``set_``
    view (uint8 over a bf16 buffer, say) is a plain alias on CPU, but on LPU it
    does not carry the layout descriptor of a >=2-D tensor: a D2H through it
    reads correctly while an H2D through it writes NOTHING and raises nothing
    (probe 2026-10-26: a bf16 ``[8, 2048]`` written back through a uint8 flat
    view read back as 24 byte diffs, i.e. the whole payload missing; the same
    write in bf16 landed byte-exactly, as did a 1-D uint8 target).  Callers that
    want bytes can ``.view(torch.uint8)`` the result -- a dtype-changing *view*
    is fine, only a dtype-changing ``set_`` is not.
    """
    storage = t.untyped_storage()
    n = storage.nbytes() // t.element_size()
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(
        storage, 0, (n,), (1,)
    )


def host_view_of_storage(t: torch.Tensor) -> torch.Tensor:
    """Host copy of a window over a large backing, read as ONE whole D2H.

    Returns the same logical view (shape, dtype, offset, strides) on the host.
    Blueprint: build a 1-D uint8 tensor over the backing, ``.cpu()`` it (zero
    device transient), then rebuild the window on the host.  The rebuild is
    mandatory -- in a packed pool the layers are interleaved page by page, so a
    flat slice from the byte offset is a DIFFERENT tensor (see
    ``tests/test_evas_dequantize_and_gather_k_cache.py``).

    A failed rebuild **raises**.  It used to fall back to ``t.cpu()``, which is
    the one thing this function exists to avoid: callers only get here because
    ``t`` is a window over a big backing, and a plain D2H of that window stages
    the whole backing -- so the fallback replaced a precise rebuild error with an
    OOM of the pool size, raised from a line far from the cause (observed on a
    0.25 carrier: a swallowed exception inside the rebuild surfaced as a 1.74 GiB
    allocation failure).
    """
    try:
        # 1-D view over the whole backing, already in t's dtype (see
        # ``storage_flat``), so the window is rebuilt by striding in ELEMENTS.
        base = storage_flat(t).cpu()
        return torch.as_strided(base, t.shape, tuple(t.stride()),
                                t.storage_offset())
    except Exception as exc:
        logger.warning(
            "[packed-cache] host view rebuild failed (shape=%s stride=%s "
            "dtype=%s): %s: %s -- raising instead of falling back to .cpu(), "
            "which would stage the whole backing",
            tuple(t.shape), tuple(t.stride()), t.dtype,
            type(exc).__name__, exc,
        )
        raise


def write_back_from_host(t: torch.Tensor, host_view: torch.Tensor) -> None:
    """Push a host window (from ``host_view_of_storage``) back as ONE whole H2D.

    ``host_view`` must be the object ``host_view_of_storage(t)`` returned so the
    two byte ranges are the same size.  Any device-side write of a sub-interval
    of ``t`` stages the whole backing, so callers mutate the host window and
    call this instead.  Writes into the window that belong to OTHER views of the
    same allocation are carried along unchanged -- the copy is a plain
    read-modify-write of the identical bytes.

    Both ends are 1-D views in ``t.dtype`` (see ``storage_flat``): an H2D into a
    dtype-reinterpreted view of a >=2-D tensor is dropped silently on LPU, and
    the whole point of this route is that the write must actually land.
    """
    dst = storage_flat(t)
    src = storage_flat(host_view)
    if src.numel() != dst.numel():
        raise RuntimeError(
            f"host window spans {src.numel()} elements ({src.numel() * src.element_size()} B) "
            f"but the backing is {dst.numel()} ({dst.numel() * dst.element_size()} B); "
            f"pass the tensor returned by host_view_of_storage"
        )
    dst.copy_(src)

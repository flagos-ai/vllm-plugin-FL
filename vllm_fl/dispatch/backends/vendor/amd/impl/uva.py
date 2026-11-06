# Copyright (c) 2026 BAAI. All rights reserved.

"""Zero-copy host-to-device tensor views for AMD ROCm.

vLLM's UVA input buffers (``vllm/v1/worker/gpu/buffer_utils.py``) ask the
platform for an accelerator-side view of a pinned CPU tensor via
``torch.ops._C.get_cuda_view_from_cpu_tensor``. That operator lives in the
native CUDA extension, which an empty-mode install does not ship, so AMD must
supply it.

ROCm maps page-locked host allocations into the device address space under
unified virtual addressing: ``hipHostGetDevicePointer`` returns the *same*
virtual address as the host pointer on gfx1100. The view is therefore built by
retyping the mapped pointer as CUDA(HIP) storage — no copy, and device writes
are visible to the host, matching the CUDA operator's contract.
"""

from __future__ import annotations

import ctypes
import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

_HIP_SONAMES = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")


@lru_cache(maxsize=1)
def _load_hip_runtime() -> ctypes.CDLL | None:
    """Load and cache the HIP runtime shared library."""
    for soname in _HIP_SONAMES:
        try:
            return ctypes.CDLL(soname)
        except OSError:
            continue
    logger.warning(
        "Could not load the HIP runtime (%s); zero-copy host views are "
        "unavailable on this AMD build.",
        ", ".join(_HIP_SONAMES),
    )
    return None


def _mapped_device_pointer(host_ptr: int) -> int:
    """Return the device-visible address for a pinned host address.

    Takes a raw address rather than a tensor: the caller maps the *storage
    base*, not ``tensor.data_ptr()``. ``data_ptr()`` already includes the
    tensor's storage offset, and the offset is applied again when the view is
    built, so mapping it here would shift the view twice.
    """
    hip = _load_hip_runtime()
    if hip is None:
        # Under UVA the host address is already device-addressable; use it
        # directly rather than failing outright.
        return host_ptr

    device_ptr = ctypes.c_void_p()
    status = hip.hipHostGetDevicePointer(
        ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
    )
    if status != 0:
        raise RuntimeError(
            "hipHostGetDevicePointer failed with status "
            f"{status} for a pinned CPU tensor; cannot build a zero-copy "
            "device view on this AMD device."
        )
    return device_ptr.value or host_ptr


# Storage built by ``_construct_storage_from_data_pointer`` wraps a raw pointer
# with no deleter, so it does not retain the host allocation the way upstream's
# ``from_blob`` deleter does. Callers such as vLLM's ``UVAOffloader`` assign the
# view to ``p.data`` and then drop their own handle on the pinned tensor; the
# allocation would be recycled while the parameter still points into it, which
# silently corrupts the data. Ownership cannot be attached to the returned
# storage from Python — assigning to ``p.data`` keeps the StorageImpl, not the
# Python tensor or any attribute on it — so the source storage is retained here,
# keyed by its host address. Keys repeat because the pinned-memory caching
# allocator reuses addresses, so this is bounded by the distinct pinned
# allocations actually mapped, and retaining them is the intended lifetime: the
# view *is* that memory.
_mapped_sources: dict[int, torch.UntypedStorage] = {}


def get_cuda_view_from_cpu_tensor_amd(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy accelerator view of a pinned CPU tensor.

    Args:
        cpu_tensor: A page-locked (pinned) CPU tensor.

    Returns:
        A CUDA(HIP)-device tensor sharing storage with ``cpu_tensor``.

    Raises:
        ValueError: If the tensor is not a pinned CPU tensor.
        RuntimeError: If the HIP mapping or storage construction fails.
    """
    if cpu_tensor.device.type != "cpu":
        raise ValueError(
            "get_cuda_view_from_cpu_tensor expects a CPU tensor, got "
            f"{cpu_tensor.device}."
        )
    if not cpu_tensor.is_pinned():
        raise ValueError(
            "A zero-copy device view requires page-locked (pinned) host "
            "memory; got an unpinned CPU tensor."
        )

    construct_storage = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if construct_storage is None:
        raise RuntimeError(
            "torch._C._construct_storage_from_data_pointer is unavailable, so "
            "a zero-copy host view cannot be constructed on this torch build."
        )

    # Map the storage base, so that ``storage_offset`` below is applied exactly
    # once; mapping ``data_ptr()`` would apply it twice and shift the view.
    source = cpu_tensor.untyped_storage()
    base_ptr = source.data_ptr()
    device_ptr = _mapped_device_pointer(base_ptr)
    device = torch.device("cuda", torch.cuda.current_device())

    # Keep the pinned allocation alive for as long as any view of it can exist.
    _mapped_sources.setdefault(base_ptr, source)

    storage = construct_storage(device_ptr, device, source.nbytes())
    view = torch.empty(0, dtype=cpu_tensor.dtype, device=device)
    view.set_(
        storage,
        storage_offset=cpu_tensor.storage_offset(),
        size=cpu_tensor.size(),
        stride=cpu_tensor.stride(),
    )
    return view

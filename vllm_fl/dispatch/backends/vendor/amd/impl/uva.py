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

import torch

logger = logging.getLogger(__name__)

_HIP_SONAMES = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")

_hip_runtime: ctypes.CDLL | None = None
_hip_runtime_loaded = False


def _load_hip_runtime() -> ctypes.CDLL | None:
    """Load and cache the HIP runtime shared library."""
    global _hip_runtime, _hip_runtime_loaded
    if _hip_runtime_loaded:
        return _hip_runtime
    _hip_runtime_loaded = True
    for soname in _HIP_SONAMES:
        try:
            _hip_runtime = ctypes.CDLL(soname)
            return _hip_runtime
        except OSError:
            continue
    logger.warning(
        "Could not load the HIP runtime (%s); zero-copy host views are "
        "unavailable on this AMD build.",
        ", ".join(_HIP_SONAMES),
    )
    return None


def _mapped_device_pointer(cpu_tensor: torch.Tensor) -> int:
    """Return the device-visible address of a pinned CPU tensor's storage."""
    hip = _load_hip_runtime()
    host_ptr = cpu_tensor.data_ptr()
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

    device_ptr = _mapped_device_pointer(cpu_tensor)
    nbytes = cpu_tensor.untyped_storage().nbytes()
    device = torch.device("cuda", torch.cuda.current_device())

    storage = construct_storage(device_ptr, device, nbytes)
    view = torch.empty(0, dtype=cpu_tensor.dtype, device=device)
    view.set_(
        storage,
        storage_offset=cpu_tensor.storage_offset(),
        size=cpu_tensor.size(),
        stride=cpu_tensor.stride(),
    )
    return view

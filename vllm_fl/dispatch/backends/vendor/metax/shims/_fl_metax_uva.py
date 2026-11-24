"""UVA view of pinned CPU tensors for MetaX with the empty vLLM wheel.

``torch.ops._C.get_cuda_view_from_cpu_tensor`` lives in ``vllm._C``, which the
empty wheel does not build.  On a UVA system (MACA qualifies: a pinned host
allocation is directly device-accessible at the same address) we can produce
the alias with a DLPack capsule that points the CUDA tensor at the pinned host
pointer.  No copy, no extra GPU memory.

Kill switch: ``FL_METAX_UVA=0``.
"""

import ctypes
import os

import torch
from torch.utils.dlpack import from_dlpack

from ._fl_metax_gate import metax_active

_SWITCH = "FL_METAX_UVA"


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_DTYPE = {
    torch.float32: (2, 32),
    torch.bfloat16: (4, 16),  # kDLBfloat
    torch.float16: (2, 16),
    torch.int64: (0, 64),
    torch.int32: (0, 32),
    torch.int16: (0, 16),
    torch.int8: (0, 8),
    torch.uint8: (1, 8),
    torch.bool: (6, 8),
}

# Keep the ctypes structs/arrays alive for the lifetime of the alias.
_KEEP: dict[int, tuple] = {}
_LIB = None


def _uva_view(cpu_tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(cpu_tensor, torch.Tensor):
        raise TypeError("get_cuda_view_from_cpu_tensor expects a tensor")
    if cpu_tensor.device.type != "cpu":
        raise ValueError(
            f"get_cuda_view_from_cpu_tensor expects a CPU tensor, got {cpu_tensor.device}"
        )
    if not cpu_tensor.is_pinned():
        raise ValueError("get_cuda_view_from_cpu_tensor requires pinned host memory")
    fp8 = cpu_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if not fp8 and cpu_tensor.dtype not in _DTYPE:
        raise TypeError(f"unsupported dtype for UVA alias: {cpu_tensor.dtype}")

    shape = (ctypes.c_int64 * cpu_tensor.dim())(*cpu_tensor.shape)
    strides = (ctypes.c_int64 * cpu_tensor.dim())(*cpu_tensor.stride())
    # DLPack has no FP8 float code; expose it as uint8 and view back below.
    code, bits = (1, 8) if fp8 else _DTYPE[cpu_tensor.dtype]

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(cpu_tensor.data_ptr())
    # CPU tensors carry no device index; use the worker's current device so
    # every TP rank gets an alias on its own cuda:N (not always cuda:0).
    managed.dl_tensor.device = _DLDevice(2, torch.cuda.current_device())  # kDLCUDA
    managed.dl_tensor.ndim = cpu_tensor.dim()
    managed.dl_tensor.dtype = _DLDataType(code, bits, 1)
    managed.dl_tensor.shape = shape
    managed.dl_tensor.strides = strides
    managed.dl_tensor.byte_offset = 0
    # The host tensor owns the memory; this capsule must never free it.
    deleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _p: None)
    managed.deleter = ctypes.cast(deleter, ctypes.c_void_p)

    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.byref(managed), b"dltensor", None
    )
    view = from_dlpack(capsule)
    if fp8:
        view = view.view(cpu_tensor.dtype)
    _KEEP[view.data_ptr()] = (managed, shape, strides, deleter, cpu_tensor)
    return view


def patch() -> bool:
    if not metax_active():
        return False
    value = os.environ.get(_SWITCH, "1").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False

    global _LIB
    _LIB = torch.library.Library("_C", "FRAGMENT")
    if not hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor"):
        _LIB.define("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor")
    _LIB.impl("get_cuda_view_from_cpu_tensor", _uva_view, "CPU")
    print(
        "[vllm_fl][metax] registered _C.get_cuda_view_from_cpu_tensor (DLPack UVA)",
        hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor"),
    )
    return True


patch()

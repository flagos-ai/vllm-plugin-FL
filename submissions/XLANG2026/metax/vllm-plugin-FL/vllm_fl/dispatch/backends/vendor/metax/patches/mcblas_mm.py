# SPDX-License-Identifier: Apache-2.0
"""MetaX: direct mcBLAS GEMM binding (ctypes -> mcblasGemmEx).

Why this exists: FlagGems registers its Triton kernels as the CUDA-dispatch
implementation of aten::mm / aten::linear process-wide, so once FlagGems is
enabled there is NO python-level call form that reaches the chip vendor's
mcBLAS GEMM.  Measured on C500 (MiniCPM5-2B prefill shapes, M=2048, bf16):

    shape (K,N)     vendor    mm_kernel_nn (best of expanded sweep)
    qkv  2048,2560   131us      178us   (1.36x slower)
    o    2048,2048   113us      177us   (1.56x)
    gate 2048,12288  522us      707us   (1.35x)
    down 6144,2048   294us      492us   (1.67x)

The vendor library wins by a wide margin at prefill M; the hand-tuned
Triton kernels keep their edge at skinny decode M (see mm_kernel_nn_db).
This module binds mcblasGemmEx directly so the linear dispatch can pick
the faster implementation per shape.  The binding is bit-exact vs
aten::mm (fp32 compute, bf16 io) and adds ~2us of host overhead.

Row-major C[M,N] = A[M,K] @ B[K,N] is issued as the column-major identity
C^T = B^T A^T: gemmEx(OP_N, OP_N, m=N, n=M, k=K, A=B lda=N, B=A ldb=K,
C=C ldc=N).

Registered as a custom op (vllm_fl_metax::mcblas_mm) so dynamo/inductor
treat it as an opaque extern call inside compiled regions.
"""

import ctypes
import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_MCBLAS_OP_N = 0
_MACA_R_16BF = 14
_MCBLAS_COMPUTE_32F = 68
_MCBLAS_GEMM_DEFAULT = -1

# VLLM_FL_DISABLE_MCBLAS_MM=1: never take the vendor path (A/B switch)
_DISABLED = os.getenv("VLLM_FL_DISABLE_MCBLAS_MM", "0") == "1"

_lib = None
_handle = None
_alpha = None
_beta = None


def _init() -> bool:
    global _lib, _handle, _alpha, _beta
    if _lib is not None:
        return True
    try:
        lib = ctypes.CDLL("/opt/maca/lib/libmcblas.so")
        lib.mcblasCreate.restype = ctypes.c_int
        lib.mcblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.mcblasSetStream.restype = ctypes.c_int
        lib.mcblasSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.mcblasGemmEx.restype = ctypes.c_int
        lib.mcblasGemmEx.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        handle = ctypes.c_void_p()
        if lib.mcblasCreate(ctypes.byref(handle)) != 0:
            return False
        _lib = lib
        _handle = handle
        _alpha = ctypes.c_float(1.0)
        _beta = ctypes.c_float(0.0)
        return True
    except Exception as e:
        logger.warning("fl mcblas_mm: binding unavailable: %s", e)
        return False


def _mm_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    st = _lib.mcblasSetStream(
        _handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    )
    assert st == 0, f"mcblasSetStream failed: {st}"
    st = _lib.mcblasGemmEx(
        _handle, _MCBLAS_OP_N, _MCBLAS_OP_N,
        N, M, K,
        ctypes.byref(_alpha),
        ctypes.c_void_p(b.data_ptr()), _MACA_R_16BF, N,
        ctypes.c_void_p(a.data_ptr()), _MACA_R_16BF, K,
        ctypes.byref(_beta),
        ctypes.c_void_p(out.data_ptr()), _MACA_R_16BF, N,
        _MCBLAS_COMPUTE_32F, _MCBLAS_GEMM_DEFAULT,
    )
    assert st == 0, f"mcblasGemmEx failed: {st}"
    return out


def _mm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty(a.shape[0], b.shape[1])


AVAILABLE = not _DISABLED and _init()
if AVAILABLE:
    torch.library.custom_op(
        "vllm_fl_metax::mcblas_mm", mutates_args=()
    )(_mm_impl)
    torch.library.register_fake("vllm_fl_metax::mcblas_mm")(_mm_fake)
    logger.info("fl mcblas_mm: vendor mcBLAS GEMM binding registered")


def mcblas_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B via vendor mcBLAS. a [M,K] bf16 contig, b [K,N] bf16 contig."""
    return torch.ops.vllm_fl_metax.mcblas_mm(a, b)

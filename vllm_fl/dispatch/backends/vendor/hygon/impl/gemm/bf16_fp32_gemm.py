# Copyright (c) 2026 BAAI. All rights reserved.

"""BF16xBF16 -> FP32 GEMM on hipBLASLt for DeepSeek-V4 compressor/indexer.

vLLM computes the compressor score projections with
``torch.mm(hidden, W.T, out_dtype=torch.float32)``
(``vllm/models/deepseek_v4/attention.py::attn_gemm_parallel_execute``).  On
this ROCm stack that routes to a rocBLAS bf16->fp32 kernel whose heuristics
collapse on skinny decode batches: measured 1268us for a 64x1024x4096 GEMM
(0.08% of peak) — ~2.4ms per decode iteration across the 6 projections.

The Hygon-native vLLM ships the same fix this module ports: a JIT hipBLASLt
call for ``x @ W.T`` with bf16 inputs and fp32 output (identical numerics:
fp32 accumulation).  On top of the native version this module adds

- a heuristic variant (``hipblasLtMatmulAlgoGetHeuristic``) with per-shape
  descriptor/algo caching, and
- shape routing, because hipBLASLt regresses badly on narrow-N x large-M
  (N<=512, M>=4096: 3336us vs rocBLAS 948us):

  ============  =====================  =======================================
  N >= 1024     any M                  hipBLASLt default algo (2.4~38.9x)
  N <  1024     M <= 1024              hipBLASLt heuristic (2.9~11.1x)
  N <  1024     M >  1024              torch.mm(out_dtype=fp32) (upstream)
  ============  =====================  =======================================

Measured on BW1000 (``performance_opt/dv4_model/gemm``); all three paths
match the fp32 reference within 2e-2 relative.

M1..256/K4096/N512,1024,2048 use deterministic Triton split-K with FP32
partials and output. Set ``VLLM_FL_HYGON_DECODE_SPLITK_GEMM=0`` to restore
the hipBLASLt routing below. This covers the full ordinary decode token
range for max_num_seqs=256, including padded graph batches and tail batches.
Larger token counts retain the existing paths.

Select with ``VLLM_FL_BF16_FP32_GEMM`` = ``blaslt`` (default) or ``torch``;
any build/call failure falls back to upstream for that call.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_HIPBLASLT_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>
#include <mutex>

static hipblasLtHandle_t get_handle() {
    static hipblasLtHandle_t handle = nullptr;
    if (handle == nullptr) hipblasLtCreate(&handle);
    return handle;
}

struct Plan {
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t la, lb, lc;
    hipblasLtMatmulAlgo_t algo;
};

static std::map<std::tuple<int,int,int>, Plan> g_plans;
static std::mutex g_mu;

static const Plan& get_plan(int batch, int in_f, int out_f) {
    auto key = std::make_tuple(batch, in_f, out_f);
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_plans.find(key);
    if (it != g_plans.end()) return it->second;

    Plan p;
    hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    int transA = HIPBLAS_OP_T, transB = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
    hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16BF, in_f, out_f, in_f);
    hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16BF, in_f, batch, in_f);
    hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_32F, out_f, batch, out_f);

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    size_t ws = 32 << 20;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    hipblasLtMatmulHeuristicResult_t res[4];
    int n = 0;
    hipblasLtMatmulAlgoGetHeuristic(get_handle(), p.desc, p.la, p.lb, p.lc, p.lc, pref, 4, res, &n);
    hipblasLtMatmulPreferenceDestroy(pref);
    TORCH_CHECK(n > 0, "hipblasLt: no algo for M=", batch, " N=", out_f, " K=", in_f);
    p.algo = res[0].algo;

    auto [ins, ok] = g_plans.emplace(key, p);
    return ins->second;
}

static torch::Tensor run(torch::Tensor X, torch::Tensor W, bool heuristic) {
    int batch = X.size(0), in_f = X.size(1), out_f = W.size(0);
    auto Y = torch::empty({batch, out_f},
        torch::dtype(torch::kFloat32).device(X.device()));
    float alpha = 1.0f, beta = 0.0f;
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (heuristic) {
        const Plan& p = get_plan(batch, in_f, out_f);
        static void* ws = nullptr;
        static size_t ws_size = 32 << 20;
        if (!ws) hipMalloc(&ws, ws_size);
        hipblasLtMatmul(get_handle(), p.desc, &alpha,
            W.data_ptr(), p.la, X.data_ptr(), p.lb, &beta,
            Y.data_ptr(), p.lc, Y.data_ptr(), p.lc,
            &p.algo, ws, ws_size, stream);
    } else {
        hipblasLtMatmulDesc_t desc;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        int transA = HIPBLAS_OP_T, transB = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
        hipblasLtMatrixLayout_t la, lb, lc;
        hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, in_f, out_f, in_f);
        hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, in_f, batch, in_f);
        hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, out_f, batch, out_f);
        hipblasLtMatmul(get_handle(), desc, &alpha,
            W.data_ptr(), la, X.data_ptr(), lb, &beta,
            Y.data_ptr(), lc, Y.data_ptr(), lc,
            nullptr, nullptr, 0, stream);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatrixLayoutDestroy(la);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
    }
    return Y;
}

torch::Tensor linear_bf16_fp32(torch::Tensor X, torch::Tensor W) { return run(X, W, false); }
torch::Tensor linear_bf16_fp32_heuristic(torch::Tensor X, torch::Tensor W) { return run(X, W, true); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_bf16_fp32", &linear_bf16_fp32);
    m.def("linear_bf16_fp32_heuristic", &linear_bf16_fp32_heuristic);
}
"""


@functools.lru_cache(maxsize=1)
def _jit_hipblaslt():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="vllm_fl_linear_bf16_fp32_hipblaslt",
        cpp_sources="",
        cuda_sources=_HIPBLASLT_SOURCE,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _blaslt_available() -> bool:
    if os.environ.get("VLLM_FL_BF16_FP32_GEMM", "blaslt").lower() in ("torch", "0", "off"):
        return False
    if torch.version.hip is None:
        return False
    try:
        _jit_hipblaslt()
        return True
    except Exception as exc:
        logger.warning(
            "hipBLASLt bf16->fp32 GEMM unavailable (%s: %s); compressor "
            "projections stay on torch.mm",
            type(exc).__name__,
            exc,
        )
        return False


def _torch_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.mm(x, w.T, out_dtype=torch.float32)


def linear_bf16_fp32_hygon(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w.T`` with bf16 inputs and fp32 output (hybrid routing)."""
    if x.dim() != 2 or w.dim() != 2 or w.shape[1] != x.shape[1]:
        return _torch_reference(x, w)
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return _torch_reference(x, w)
    # hipBLASLt 布局假设行主序连续; torch.mm 接受任意 stride —— 非连续输入
    # (如 MHC 输出的视图) 必须回落, 否则读到错位内存
    if not (x.is_contiguous() and w.is_contiguous()):
        return _torch_reference(x, w)
    m, k = x.shape
    n = w.shape[0]
    if (
        1 <= m <= 256 and k == 4096 and n in (512, 1024, 2048)
        and x.is_cuda and w.device == x.device and torch.version.hip is not None
        and os.getenv("VLLM_FL_BF16_FP32_GEMM", "blaslt").lower()
        not in ("torch", "0", "off")
        and os.getenv("VLLM_FL_HYGON_DECODE_SPLITK_GEMM", "1").lower()
        not in ("0", "false", "no", "off")
    ):
        from .decode_projection import decode_projection_bf16_fp32

        return decode_projection_bf16_fp32(x, w)
    if not _blaslt_available():
        return _torch_reference(x, w)

    try:
        ext = _jit_hipblaslt()
        if n >= 1024:
            return ext.linear_bf16_fp32(x, w)
        if m <= 1024:
            return ext.linear_bf16_fp32_heuristic(x, w)
    except Exception as exc:  # noqa: BLE001 - fall back per call
        logger.warning_once(
            "hipBLASLt bf16->fp32 GEMM failed (%s: %s); using torch.mm",
            type(exc).__name__,
            exc,
        )
    return _torch_reference(x, w)

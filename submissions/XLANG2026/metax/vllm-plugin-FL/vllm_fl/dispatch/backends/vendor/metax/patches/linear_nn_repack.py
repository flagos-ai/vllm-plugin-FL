# SPDX-License-Identifier: Apache-2.0
# 2026 - MiniCPM-FlagServe: MetaX single nn-layout weights + skinny-M nn_db.

# FlagGems' metax mm_kernel_nn ([K,N] row-major weight) is ~1.35-2x faster
# than the nt path at prefill shapes (M >= ~256), because the MACA Triton
# async-load pipeline works for nn but not nt.  At decode shapes (M <= 128)
# the hand-written double-buffered mm_kernel_nn_db matches or beats nt_db
# (measured layer total ratio 0.982), so decode can use the same nn copy.
# This patch therefore repacks linear weights ONCE into nn layout, routes all
# M through torch.mm(x, W_nn), and releases the original (N,K) storage —
# no duplicate weight copy (~4GB returned to the KV pool on MiniCPM5-2B).
# Math and dtypes are unchanged.

import os

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.layers.utils as layer_utils
import vllm.model_executor.layers.vocab_parallel_embedding as vpe_mod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.logger import init_logger

try:
    from . import mcblas_mm as _mcblas
except Exception:  # binding must never break the stock path
    _mcblas = None

logger = init_logger(__name__)

_FL_NN_MAX_N = 16384  # layer GEMMs above this N keep their existing path
_FL_NN_MIN_N = 512
# Single nn layout (default): each repacked linear keeps only the [K, N]
# copy, so every M -- decode included -- goes through the mcBLAS nn GEMM.
# Releasing the nt storage returns ~4 GB to the KV pool: 1,111,744 ->
# 1,203,904 tokens (+8.3%), i.e. max concurrency at 131,072 tokens goes
# 8.48x -> 9.19x.  The two official cases are unchanged within noise
# (4k 12451.84 vs 12450.37, 16k 11097.29 vs 11095.14), and at 64-way
# concurrency the pool is not the limiter -- the win is headroom for
# long-context or higher-concurrency serving.
# The dual layout stays available: NN_MIN_M=192 + FREE_NT_WEIGHTS=0.
_FL_FREE_NT = os.getenv("VLLM_FL_FREE_NT_WEIGHTS", "1") == "1"
_FL_NN_MIN_M = int(os.getenv("VLLM_FL_NN_MIN_M", "0"))
# VLLM_FL_DISABLE_LMHEAD_NN=1: lm_head keeps the stock vendor nt path (A/B switch)
_FL_DISABLE_LMHEAD_NN = os.getenv("VLLM_FL_DISABLE_LMHEAD_NN", "0") == "1"

_orig_pw = UnquantizedLinearMethod.process_weights_after_loading


def _process_weights_after_loading_metax(self, layer):
    _orig_pw(self, layer)
    # dynamo traces dispatch_unquantized_gemm: every attribute it reads via
    # getattr-with-default must actually EXIST on the module, so set the
    # flags unconditionally here instead of only when repacking.
    layer._fl_always_nn = False
    w = getattr(layer, "weight", None)
    if (
        w is not None
        and w.dim() == 2
        # bias layers fall back to F.linear, which needs the original weight
        and getattr(layer, "bias", None) is None
        and w.dtype in (torch.bfloat16, torch.float16)
        and w.is_cuda
        and w.numel() > 0
        and w.stride(1) == 1
        and w.stride(0) == w.shape[1]
        and _FL_NN_MIN_N <= w.shape[0] <= _FL_NN_MAX_N
        and w.shape[1] % 128 == 0
    ):
        layer._fl_w_nn = w.data.t().contiguous()
        if _FL_FREE_NT:
            # decode no longer needs the (N,K) copy; keep the Parameter shell
            # (shape metadata) but release the storage
            layer.weight.data = torch.empty(
                0, dtype=w.dtype, device=w.device
            )


# lm_head (N = vocab = 130560) is excluded from the generic repack above
# (_FL_NN_MAX_N) on the assumption that huge-N GEMMs are served better
# elsewhere.  Measured on C500 (bench_lm_head*.py): the vendor nn GEMM beats
# the vendor nt (F.linear) path ~1.75x at every decode batch size
# (378us vs 659us at M=64), while the Triton double-buffer kernels lose badly
# at this N (best 736us).  lm_head only ever sees skinny M here (vLLM gathers
# last-token hidden states, M <= num_seqs), so always route it to the nn copy.
# ParallelLMHead goes through UnquantizedEmbeddingMethod, not
# UnquantizedLinearMethod, so it needs its own process_weights hook.
# Cost: one extra [K, N] bf16 copy (+534MB); the nt copy stays (fallback).

_orig_pw_embed = UnquantizedEmbeddingMethod.process_weights_after_loading


def _process_weights_after_loading_embed_metax(self, layer):
    _orig_pw_embed(self, layer)
    layer._fl_always_nn = False
    if _FL_DISABLE_LMHEAD_NN or type(layer).__name__ != "ParallelLMHead":
        return
    w = getattr(layer, "weight", None)
    if (
        w is not None
        and w.dim() == 2
        # bias layers fall back to F.linear, which needs the original weight
        and getattr(layer, "bias", None) is None
        and w.dtype in (torch.bfloat16, torch.float16)
        and w.is_cuda
        and w.numel() > 0
        and w.stride(1) == 1
        and w.stride(0) == w.shape[1]
        and w.shape[1] % 128 == 0
    ):
        layer._fl_w_nn = w.data.t().contiguous()
        layer._fl_always_nn = True
        logger.info(
            "fl opt7: lm_head repacked to nn layout (%s), always-nn routing",
            tuple(layer._fl_w_nn.shape),
        )


def _metax_unquantized_gemm(layer, x, weight, bias):
    wn = getattr(layer, "_fl_w_nn", None)
    if wn is not None and bias is None and x.dim() >= 2:
        m = x.numel() // x.shape[-1]
        if (
            getattr(layer, "_fl_always_nn", False)
            or m > _FL_NN_MIN_M
            or weight.numel() == 0
        ):
            x2 = x.reshape(-1, x.shape[-1])
            # Vendor mcBLAS beats the Triton nn kernel 1.35-1.67x at prefill
            # M (sweep_mm_prefill.py); FlagGems' process-wide aten override
            # makes the vendor library unreachable through torch.mm, so the
            # ctypes-bound custom op is the only way to reach it.
            if (
                _mcblas is not None
                and _mcblas.AVAILABLE
                and wn.dtype == torch.bfloat16
                and x2.is_contiguous()
            ):
                out = _mcblas.mcblas_mm(x2, wn)
            else:
                out = torch.mm(x2, wn)
            return out.view(*x.shape[:-1], wn.shape[1])
    return torch.nn.functional.linear(x, weight, bias)


def _dispatch_unquantized_gemm_metax():
    return _metax_unquantized_gemm


UnquantizedLinearMethod.process_weights_after_loading = (
    _process_weights_after_loading_metax
)
UnquantizedEmbeddingMethod.process_weights_after_loading = (
    _process_weights_after_loading_embed_metax
)
layer_utils.dispatch_unquantized_gemm = _dispatch_unquantized_gemm_metax
linear_mod.dispatch_unquantized_gemm = _dispatch_unquantized_gemm_metax
# vocab_parallel_embedding did a from-import of dispatch_unquantized_gemm at
# module import time, so its local binding still points to the stock factory —
# lm_head (ParallelLMHead) would never see our dispatch without this.
vpe_mod.dispatch_unquantized_gemm = _dispatch_unquantized_gemm_metax

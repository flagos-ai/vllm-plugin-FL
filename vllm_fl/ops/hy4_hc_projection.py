# SPDX-License-Identifier: Apache-2.0
"""Skinny-N FP32 projection used by HY4 hyper-connections.

HY4's read gate is an unusually small linear layer: after flattening the four
streams, the input is ``[M, 24576]`` and ``hc_fn`` is ``[8, 24576]``.  The
regular GEMM path is a poor fit for the decode values of ``M`` (one to a few
dozen tokens).  This module keeps the FP32 reduction and projection in one
Triton launch for that tail.  For larger batches it computes only the per-row
inverse RMS in Triton before the regular GEMM.  That path avoids materializing
the ``[M, 24576]`` FP32 ``square()`` temporary from the literal expression.
It deliberately has conservative metadata-only dispatch guards and always has
the reference PyTorch path available.

The literal contract follows the NVIDIA HY4 reference in
``Hy4-p vLLM/vllm/models/hy_v4/nvidia/hc.py``.  That source notes that fused
HPC kernels are not bundled, so this implementation does not change the
surrounding gate/sigmoid or residual-reduction semantics.

The kernel is graph-safe for a fixed shape: it derives all addresses from
``program_id`` and tensor metadata, does not read device data on the host, and
does not build runtime metadata tensors.  Triton autotuning (if enabled by a
future change) must be warmed up before graph capture; this implementation
uses a fixed config for that reason.  The validated HY4 path is enabled by
default.  Set ``VLLM_HY4_HC_N8_PROJECTION=0`` for an exact reference-path
rollback.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # Keep importing the model usable on non-CUDA/non-Triton installations.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):  # pragma: no cover - CPU-only environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


HC_N8_K = 24576
HC_N8_N = 8
HC_N8_MAX_M = 64
HC_N8_ENABLE_ENV = "VLLM_HY4_HC_N8_PROJECTION"


def _enabled() -> bool:
    """Return the default-on HY4 switch without doing any device work."""

    value = os.getenv(HC_N8_ENABLE_ENV)
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "on", "yes"}


if _TRITON_AVAILABLE:

    @triton.jit
    def _hc_n8_gemv_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        M,
        K,
        eps,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_K: tl.constexpr,
    ):
        """One warp per output channel for the tiny-M GEMV tail.

        Reducing all eight channels in one wide program leaves most lanes idle
        at the final reductions for this skinny shape.  This GEMV layout gives
        the scheduler independent programs and relies on the read-only cache
        for repeated input/norm loads.
        """

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        norm_acc = tl.zeros((), dtype=tl.float32)
        dot_acc = tl.zeros((), dtype=tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                w_ptr + pid_n * stride_wn + offs_k * stride_wk,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            norm_acc += tl.sum(x * x, axis=0)
            dot_acc += tl.sum(x * w, axis=0)
        inv_norm = tl.rsqrt(norm_acc / K + eps)
        tl.store(
            out_ptr + pid_m * stride_om + pid_n * stride_on,
            dot_acc * inv_norm,
        )

    @triton.jit
    def _hc_row_inv_rms_kernel(
        x_ptr,
        inv_rms_ptr,
        K,
        eps,
        stride_xm,
        stride_xk,
        stride_rm,
        BLOCK_K: tl.constexpr,
    ):
        """Reduce one flattened HC row without a full-size square tensor."""

        pid_m = tl.program_id(0)
        norm_acc = tl.zeros((), dtype=tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            norm_acc += tl.sum(x * x, axis=0)
        tl.store(inv_rms_ptr + pid_m * stride_rm, tl.rsqrt(norm_acc / K + eps))

else:
    _hc_n8_gemv_kernel = None
    _hc_row_inv_rms_kernel = None


def _is_candidate(flat: torch.Tensor, weight: torch.Tensor) -> bool:
    """Metadata-only eligibility check for the specialized kernel."""

    if not _TRITON_AVAILABLE or not _enabled():
        return False
    if flat.device.type != "cuda" or weight.device != flat.device:
        return False
    if flat.dtype is not torch.float32 or weight.dtype is not torch.float32:
        return False
    if flat.ndim != 2 or weight.ndim != 2:
        return False
    if flat.shape[1] != HC_N8_K or weight.shape != (HC_N8_N, HC_N8_K):
        return False
    # A single-token GEMV is already handled efficiently by the native
    # backend; keep M=1 on the reference path until a dedicated one-row
    # promotion proves a stable win across eager and graph timings.
    if not (2 <= flat.shape[0] <= HC_N8_MAX_M):
        return False
    # The address arithmetic below intentionally assumes these layouts.  The
    # reference path handles transposed/strided parameters without changing
    # semantics.
    if not flat.is_contiguous() or not weight.is_contiguous():
        return False
    # FP32 arithmetic is supported on every CUDA architecture, but this
    # specialization is only validated on SM80+ where Triton has the stable
    # CUDA codegen used by the production path.
    major, _minor = torch.cuda.get_device_capability(flat.device)
    return major >= 8


def _is_large_m_rms_candidate(flat: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether the memory-safe large-M inverse-RMS path is usable."""

    if not _TRITON_AVAILABLE or not _enabled():
        return False
    if flat.device.type != "cuda" or weight.device != flat.device:
        return False
    if flat.dtype is not torch.float32 or weight.dtype is not torch.float32:
        return False
    if flat.ndim != 2 or weight.ndim != 2:
        return False
    if flat.shape[1] != HC_N8_K or weight.shape[1] != HC_N8_K:
        return False
    # The two production callers have eight gate rows in every HC pre-layer
    # and four gate rows in the final HC head.
    if weight.shape[0] not in (4, HC_N8_N) or flat.shape[0] <= HC_N8_MAX_M:
        return False
    if not flat.is_contiguous():
        return False
    major, _minor = torch.cuda.get_device_capability(flat.device)
    return major >= 8


def _large_m_inv_rms(flat: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute ``rsqrt(mean(flat**2) + eps)`` as an ``[M, 1]`` tensor."""

    inv_rms = torch.empty(
        (flat.shape[0], 1), device=flat.device, dtype=torch.float32
    )
    _hc_row_inv_rms_kernel[(flat.shape[0],)](
        flat,
        inv_rms,
        HC_N8_K,
        float(eps),
        flat.stride(0),
        flat.stride(1),
        inv_rms.stride(0),
        BLOCK_K=2048,
        num_warps=4,
        num_stages=2,
    )
    return inv_rms


def _hc_n8_projection_impl(
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return the normalized skinny projection, or the exact reference path.

    The public contract mirrors the expression in ``HYV4HyperConnection``::

        F.linear(flat.float(), weight) * rsqrt(mean(flat.float() ** 2) + eps)

    Callers normally pass already-flattened FP32 tensors.  Keeping the fallback
    here makes the dispatch safe for CPU, unsupported dtypes/layouts, disabled
    deployments, and future HY4 dimensions.
    """

    if not _is_candidate(flat, weight):
        flat_fp32 = flat.float()
        if _is_large_m_rms_candidate(flat_fp32, weight):
            # At the production scheduler ceiling (M=8192), the literal
            # square() temporary is 768 MiB.  This allocation is only M*4
            # bytes and is safe to replay in a fixed-shape CUDA graph.
            norm = _large_m_inv_rms(flat_fp32, eps)
        else:
            norm = torch.rsqrt(
                flat_fp32.square().mean(dim=-1, keepdim=True) + eps
            )
        return F.linear(flat_fp32, weight) * norm

    m = flat.shape[0]
    out = torch.empty((m, HC_N8_N), device=flat.device, dtype=torch.float32)
    # No host synchronization or data-dependent metadata is used here.  The
    # output allocation and launch are replayable by CUDA graph capture after
    # one eager warmup at each captured shape.
    _hc_n8_gemv_kernel[(m, HC_N8_N)](
        flat,
        weight,
        out,
        m,
        HC_N8_K,
        float(eps),
        flat.stride(0),
        flat.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_K=2048,
        num_warps=1,
        num_stages=2,
    )
    return out


_HC_N8_OP_REGISTERED = False


def _hc_n8_projection_op(
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _hc_n8_projection_impl(flat, weight, eps)


def _hc_n8_projection_fake(
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del eps
    return torch.empty(
        (flat.shape[0], weight.shape[0]),
        device=flat.device,
        dtype=torch.float32,
    )


try:  # vLLM's direct custom-op wrapper keeps Dynamo from graph-breaking here.
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="hy4_hc_n8_projection",
        op_func=_hc_n8_projection_op,
        mutates_args=[],
        fake_impl=_hc_n8_projection_fake,
    )
    _HC_N8_OP_REGISTERED = True
except (ImportError, AttributeError, RuntimeError):
    # Unit tests and CPU-only tools can load this module without vLLM.  The
    # direct Python function remains fully usable in those environments.
    pass


def hc_n8_projection(
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Graph-friendly wrapper around the specialized projection."""

    if _HC_N8_OP_REGISTERED:
        return torch.ops.vllm.hy4_hc_n8_projection(flat, weight, float(eps))
    return _hc_n8_projection_impl(flat, weight, eps)


__all__ = [
    "HC_N8_ENABLE_ENV",
    "HC_N8_K",
    "HC_N8_MAX_M",
    "HC_N8_N",
    "_is_candidate",
    "_is_large_m_rms_candidate",
    "hc_n8_projection",
]

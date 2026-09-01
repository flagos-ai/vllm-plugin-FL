# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU INT8 activation quantization for vLLM W8A8 path."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INT8_MIN = -128
_INT8_MAX = 127
# Symmetric int8 uses 127 as the scale denominator so +/-amax maps to +/-127.
_SCALE_DENOM = 127.0


@triton.jit
def _dynamic_scaled_int8_quant_single_kernel(
    y_ptr,
    q_ptr,
    s_ptr,
    K,
    y_row_stride,
    q_row_stride,
    eps,
    scale_denom,
    int8_min,
    int8_max,
    BLOCK: tl.constexpr,
):
    """Single-load path for K <= BLOCK: amax + quantize without a second HBM read."""
    row = tl.program_id(0)
    y_row = y_ptr + row * y_row_stride
    q_row = q_ptr + row * q_row_stride

    cols = tl.arange(0, BLOCK)
    mask = cols < K
    y = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(y))
    scale = tl.maximum(amax, eps) / scale_denom
    tl.store(s_ptr + row, scale)

    inv_scale = 1.0 / scale
    x = y * inv_scale
    r = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
    r = tl.minimum(tl.maximum(r, int8_min), int8_max)
    tl.store(q_row + cols, r.to(q_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dynamic_scaled_int8_quant_twopass_kernel(
    y_ptr,
    q_ptr,
    s_ptr,
    K,
    y_row_stride,
    q_row_stride,
    eps,
    scale_denom,
    int8_min,
    int8_max,
    BLOCK: tl.constexpr,
):
    """Two-pass path for K > BLOCK (chunked amax, then chunked quantize)."""
    row = tl.program_id(0)
    y_row = y_ptr + row * y_row_stride
    q_row = q_ptr + row * q_row_stride

    amax = 0.0
    for off in range(0, K, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < K
        y = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(y)))

    scale = tl.maximum(amax, eps) / scale_denom
    tl.store(s_ptr + row, scale)

    inv_scale = 1.0 / scale
    for off in range(0, K, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < K
        y = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
        x = y * inv_scale
        r = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
        r = tl.minimum(tl.maximum(r, int8_min), int8_max)
        tl.store(q_row + cols, r.to(q_ptr.dtype.element_ty), mask=mask)


def _dynamic_triton(x2d: torch.Tensor, eps: float):
    M, K = x2d.shape
    q = torch.empty_like(x2d, dtype=torch.int8)
    s = torch.empty((M, 1), device=x2d.device, dtype=torch.float32)
    BLOCK = min(triton.next_power_of_2(K), 8192)
    num_warps = min(max(BLOCK // 256, 1), 8)
    # Common Linear/MoE K (e.g. 2048/4096) fits in one BLOCK: use single-load
    # kernel to avoid a second full-row HBM read per token.
    kernel = (
        _dynamic_scaled_int8_quant_single_kernel
        if K <= BLOCK
        else _dynamic_scaled_int8_quant_twopass_kernel
    )
    kernel[(M,)](
        x2d,
        q,
        s,
        K,
        x2d.stride(0),
        q.stride(0),
        eps,
        _SCALE_DENOM,
        _INT8_MIN,
        _INT8_MAX,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return q, s


def _dynamic_torch(x2d: torch.Tensor, eps: float):
    amax = x2d.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
    scale = amax / _SCALE_DENOM
    q = (
        (x2d.to(torch.float32) / scale)
        .round()
        .clamp(_INT8_MIN, _INT8_MAX)
        .to(torch.int8)
    )
    return q, scale


def _dynamic_torch_asymmetric(x2d, input, eps):
    xmin = x2d.amin(dim=-1, keepdim=True).to(torch.float32)
    xmax = x2d.amax(dim=-1, keepdim=True).to(torch.float32)
    scale = ((xmax - xmin) / (_INT8_MAX - _INT8_MIN)).clamp(min=eps)
    azp = (_INT8_MIN - (xmin / scale).round()).to(torch.int32)
    q = (
        (x2d.to(torch.float32) / scale).round() + azp
    ).clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
    return q.reshape(input.shape), scale, azp


def _static_quant(input, scale, azp, symmetric):
    """Static (per-tensor) INT8 quantization to match vLLM's static path."""
    scale_f = scale.to(torch.float32)
    x = input.to(torch.float32) / scale_f
    if not symmetric and azp is not None:
        x = x + azp.to(torch.float32)
    q = x.round().clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
    return q, scale, azp


def dynamic_scaled_int8_quant(
    input: torch.Tensor,
    symmetric: bool = True,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Dynamic per-token INT8 quantization (Triton, torch fallback)."""
    assert input.shape[-1] >= 1
    x2d = input.reshape(-1, input.shape[-1]).contiguous()

    if not symmetric:
        return _dynamic_torch_asymmetric(x2d, input, eps)

    backend = os.environ.get("FLAGGEMS_VLLM_INT8_QUANT_BACKEND", "triton").lower()
    if backend == "torch":
        q, s = _dynamic_torch(x2d, eps)
    else:
        try:
            q, s = _dynamic_triton(x2d, eps)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "dynamic_scaled_int8_quant: Triton path failed (%s); "
                "falling back to torch.",
                e,
            )
            q, s = _dynamic_torch(x2d, eps)

    return q.reshape(input.shape), s, None


def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Drop-in replacement for ``vllm._custom_ops.scaled_int8_quant``.

    * ``scale is None``  -> dynamic per-token quantization (Triton).
    * ``scale`` provided -> static per-tensor quantization (torch).
    """
    if scale is not None:
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        return _static_quant(input, scale, azp, symmetric)
    return dynamic_scaled_int8_quant(input, symmetric=symmetric)

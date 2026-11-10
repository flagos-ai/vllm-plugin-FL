# Copyright (c) 2026 BAAI. All rights reserved.

"""BW1000 small-M dense INT8 tiles; reuse vLLM's unchanged scaled-MM kernel."""

import logging
import os

import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
    scaled_mm_kernel,
    triton_scaled_mm as _original_scaled_mm,
)

# (N, K): (block M, block N, block K, waves). Validated across M=1..256.
_CONFIGS = {
    (1536, 4096): (32, 32, 256, 2),
    (512, 4096): (16, 16, 256, 2),
    (4096, 1024): (64, 32, 128, 4),
}


def _valid_scale(scale, size, device):
    return (
        scale.device == device and scale.dtype == torch.float32
        and scale.is_contiguous() and scale.ndim <= 2
        and (scale.ndim < 2 or scale.shape[1] == 1)
        and scale.numel() in (1, size)
    )


def hygon_triton_scaled_mm(
    input, weight, scale_a, scale_b, out_dtype, bias=None,
    block_size_m=32, block_size_n=32, block_size_k=32, use_heuristic=True,
):
    """Optimize validated small-M shapes, preserving the original fallback API."""
    eligible = (
        os.getenv("VLLM_FL_HYGON_DENSE_INT8_TUNING", "1").lower()
        not in {"0", "off", "false", "native"}
        and use_heuristic
        and (block_size_m, block_size_n, block_size_k) == (32, 32, 32)
        and input.is_cuda and input.ndim == weight.ndim == 2
        and input.dtype == weight.dtype == torch.int8
        and out_dtype == torch.bfloat16 and weight.device == input.device
        and 1 <= input.shape[0] <= 256 and input.is_contiguous()
        and weight.shape[0] == input.shape[1]
        and weight.stride() == (1, weight.shape[0])
        and (weight.shape[1], input.shape[1]) in _CONFIGS
        and _valid_scale(scale_a, input.shape[0], input.device)
        and _valid_scale(scale_b, weight.shape[1], input.device)
        and (bias is None or (
            bias.device == input.device and bias.dtype == torch.bfloat16
            and bias.shape == (weight.shape[1],) and bias.is_contiguous()
        ))
    )
    if not eligible:
        return _original_scaled_mm(
            input, weight, scale_a, scale_b, out_dtype, bias,
            block_size_m, block_size_n, block_size_k, use_heuristic,
        )

    m, k = input.shape
    n = weight.shape[1]
    bm, bn, bk, waves = _CONFIGS[n, k]
    result = torch.empty((m, n), dtype=out_dtype, device=input.device)
    scaled_mm_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        input, weight, scale_a, scale_b, result, bias, m, n, k,
        *input.stride(), *weight.stride(), *result.stride(), tl.int32,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
        BLOCK_SIZE_SCALE_A=1 if scale_a.numel() == 1 else bm,
        BLOCK_SIZE_SCALE_B=1 if scale_b.numel() == 1 else bn,
        num_warps=waves,
    )
    return result


def patch_dense_int8_tiles():
    """Patch only the INT8 linear caller inside the Hygon backend setup."""
    from vllm.model_executor.kernels.linear.scaled_mm import triton as linear

    if linear.triton_scaled_mm is hygon_triton_scaled_mm:
        return
    if linear.triton_scaled_mm is not _original_scaled_mm:
        logging.getLogger(__name__).warning(
            "Hygon dense INT8 tuning skipped: scaled-MM caller already replaced"
        )
        return
    linear.triton_scaled_mm = hygon_triton_scaled_mm
    logging.getLogger(__name__).info("Installed Hygon small-M dense INT8 tile tuning")

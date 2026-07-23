# Copyright (c) 2026 BAAI. All rights reserved.
"""Direct binding for the plugin-owned fused WNA16 GEMM."""

from __future__ import annotations

import torch


def _resolve_wna16_gemm():
    """Resolve the fixed torch operator exported by this plugin's extension."""
    try:
        return torch.ops.vllm_fl.wna16_gemm.default
    except AttributeError:
        return None


def is_wna16_gemm_available() -> bool:
    return _resolve_wna16_gemm() is not None


def wna16_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused dequantization GEMM without an intermediate FP tensor."""
    kernel = _resolve_wna16_gemm()
    if kernel is None:
        raise RuntimeError(
            "vllm_fl::wna16_gemm is not built; implement it under "
            "vllm_fl/quantization/wna16/kernels"
        )
    return kernel(x, weight_packed, weight_scale, group_size, bias)


__all__ = ["is_wna16_gemm_available", "wna16_gemm"]

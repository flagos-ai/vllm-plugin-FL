# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Direct binding for the plugin-owned fused WNA16 GEMM."""

from __future__ import annotations

import torch


def _resolve_wna16_gemm():
    """Resolve the fixed torch operator exported by this plugin's extension."""
    try:
        return torch.ops.vllm_fl.wna16_gemm.default
    except AttributeError:
        return None


def _resolve_w8a16_gemm():
    """Resolve an optional operator that explicitly supports uint8b128."""
    try:
        return torch.ops.vllm_fl.w8a16_gemm.default
    except AttributeError:
        return None


def is_wna16_gemm_available() -> bool:
    return _resolve_wna16_gemm() is not None


def is_w8a16_gemm_available() -> bool:
    if _resolve_w8a16_gemm() is not None:
        return True
    from .triton import is_triton_w8a16_available

    return is_triton_w8a16_available()


def wna16_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
    *,
    num_bits: int = 4,
) -> torch.Tensor:
    """Run fused dequantization GEMM without an intermediate FP tensor."""
    if num_bits == 8:
        kernel = _resolve_w8a16_gemm()
        if kernel is not None:
            return kernel(
                x,
                weight_packed,
                weight_scale,
                group_size,
                bias,
            )
        from .triton import triton_w8a16_gemm

        return triton_w8a16_gemm(
            x,
            weight_packed,
            weight_scale,
            group_size,
            bias,
        )
    kernel = _resolve_wna16_gemm()
    if kernel is not None:
        return kernel(x, weight_packed, weight_scale, group_size, bias)
    if kernel is None:
        raise RuntimeError(
            "vllm_fl::wna16_gemm is not built; implement it under "
            "vllm_fl/quantization/wna16/kernels"
        )


__all__ = [
    "is_w8a16_gemm_available",
    "is_wna16_gemm_available",
    "wna16_gemm",
]

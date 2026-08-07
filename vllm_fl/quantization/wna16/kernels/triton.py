# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable Triton W8A16 GEMM for compressed-tensors uint8b128 weights."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _w8a16_gemm_kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            offsets_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
            x_offsets = offsets_m[:, None] * K + offsets_k[None, :]
            x = tl.load(
                x_ptr + x_offsets,
                mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
                other=0.0,
            )

            weight_offsets = offsets_n[:, None] * K + offsets_k[None, :]
            codes = tl.load(
                weight_ptr + weight_offsets,
                mask=(offsets_n[:, None] < N) & (offsets_k[None, :] < K),
                other=128,
            )
            scale_offsets = (
                offsets_n[:, None] * NUM_GROUPS + offsets_k[None, :] // GROUP_SIZE
            )
            scales = tl.load(
                scale_ptr + scale_offsets,
                mask=(offsets_n[:, None] < N) & (offsets_k[None, :] < K),
                other=0.0,
            )
            weights = (codes.to(tl.int16) - 128).to(x.dtype) * scales
            accumulator += tl.dot(x, tl.trans(weights))

        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + offsets_n,
                mask=offsets_n < N,
                other=0.0,
            )
            accumulator += bias[None, :]

        output_offsets = offsets_m[:, None] * N + offsets_n[None, :]
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


def is_triton_w8a16_available() -> bool:
    return triton is not None


def triton_w8a16_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute W8A16 without materializing the dequantized weight."""
    if triton is None:
        raise RuntimeError("Triton is required for the FL W8A16 fallback")
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor")
    if weight_packed.ndim != 2 or weight_packed.dtype != torch.int32:
        raise ValueError("weight_packed must be a 2D int32 tensor")
    if not weight_packed.is_contiguous():
        weight_packed = weight_packed.contiguous()

    output_features = weight_packed.shape[0]
    input_features = x.shape[1]
    if weight_packed.shape[1] * 4 != input_features:
        raise ValueError("uint8b128 weight_packed shape is incompatible with x")
    effective_group_size = input_features if group_size == -1 else group_size
    if effective_group_size <= 0 or input_features % effective_group_size:
        raise ValueError("group_size must divide the input feature dimension")
    num_groups = input_features // effective_group_size
    expected_scales = output_features * num_groups
    if weight_scale.numel() != expected_scales:
        raise ValueError(
            f"weight_scale must contain {expected_scales} values, "
            f"got {weight_scale.numel()}"
        )
    if bias is not None and bias.numel() != output_features:
        raise ValueError("bias must contain one value per output feature")

    weight_bytes = weight_packed.view(torch.uint8)
    scales = weight_scale.reshape(output_features, num_groups).contiguous()
    output = torch.empty(
        (x.shape[0], output_features),
        dtype=x.dtype,
        device=x.device,
    )
    grid = (
        triton.cdiv(x.shape[0], 32),
        triton.cdiv(output_features, 32),
    )
    bias_arg = bias if bias is not None else output
    _w8a16_gemm_kernel[grid](
        x,
        weight_bytes,
        scales,
        bias_arg,
        output,
        M=x.shape[0],
        N=output_features,
        K=input_features,
        GROUP_SIZE=effective_group_size,
        NUM_GROUPS=num_groups,
        HAS_BIAS=bias is not None,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output


__all__ = ["is_triton_w8a16_available", "triton_w8a16_gemm"]

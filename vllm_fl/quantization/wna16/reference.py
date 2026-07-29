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
"""Torch WNA16 reference used only by unit tests.

This module is deliberately absent from the MPLinear and MoE registration
paths. It materializes the dequantized weight and must not be selected for
model inference.
"""

from __future__ import annotations

import torch


def unpack_uint4b8(
    weight_packed: torch.Tensor,
    *,
    in_features: int | None = None,
) -> torch.Tensor:
    """Unpack compressed-tensors uint4b8 int32 words to signed int8."""
    if weight_packed.ndim != 2 or weight_packed.dtype != torch.int32:
        raise ValueError("weight_packed must be a 2D int32 tensor")
    shifts = torch.arange(
        0,
        32,
        4,
        dtype=torch.int32,
        device=weight_packed.device,
    )
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(weight_packed.unsqueeze(-1), shifts),
        0xF,
    ).reshape(weight_packed.shape[0], -1)
    if in_features is not None:
        if in_features < 0 or in_features > codes.shape[1]:
            raise ValueError("in_features is incompatible with weight_packed")
        codes = codes[:, :in_features]
    return (codes.to(torch.int16) - 8).to(torch.int8)


def wna16_gemm_reference(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize dequantized weights for small numerical tests only."""
    if x.ndim < 2:
        raise ValueError("x must have at least two dimensions")
    if group_size <= 0 or x.shape[-1] % group_size:
        raise ValueError("group_size must divide the input feature dimension")
    expected_scale_shape = (
        weight_packed.shape[0],
        x.shape[-1] // group_size,
    )
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"weight_scale must have shape {expected_scale_shape}, "
            f"got {tuple(weight_scale.shape)}"
        )

    values = unpack_uint4b8(
        weight_packed,
        in_features=x.shape[-1],
    )
    scales = weight_scale.repeat_interleave(group_size, dim=1)
    weight = values.to(scales.dtype) * scales
    output = torch.matmul(x, weight.transpose(0, 1))
    if bias is not None:
        output = output + bias
    return output


__all__ = ["unpack_uint4b8", "wna16_gemm_reference"]

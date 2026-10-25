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
"""Portable W8A8 reference operations and activation quantization."""

from __future__ import annotations

import torch


def unpack_uint8b128_int32(
    weight_packed: torch.Tensor,
    *,
    in_features: int | None = None,
) -> torch.Tensor:
    """Unpack compressed-tensors offset-binary INT8 words."""
    if weight_packed.ndim != 2 or weight_packed.dtype != torch.int32:
        raise ValueError("weight_packed must be a 2D int32 tensor")
    codes = weight_packed.contiguous().view(torch.uint8)
    if in_features is not None:
        if in_features < 0 or in_features > codes.shape[1]:
            raise ValueError("in_features is incompatible with weight_packed")
        codes = codes[:, :in_features]
    return (codes.to(torch.int16) - 128).to(torch.int8)


def dynamic_per_token_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match vLLM dynamic_scaled_int8_quant for symmetric per-token INT8."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D [tokens, hidden_size] tensor")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")

    original_shape = x.shape
    # vLLM passes a contiguous input to dynamic_scaled_int8_quant. FlagGems'
    # round kernel has the same contiguity requirement.
    x_2d = x.to(torch.float32).contiguous()
    int8_info = torch.iinfo(torch.int8)
    absmax = x_2d.abs().amax(dim=-1, keepdim=True)
    scale = absmax / int8_info.max
    nonzero = absmax != 0
    safe_absmax = torch.where(nonzero, absmax, torch.ones_like(absmax))
    inv_scale = int8_info.max / safe_absmax
    inv_scale = torch.where(
        nonzero,
        inv_scale,
        torch.zeros_like(inv_scale),
    )
    quantized = (
        (x_2d * inv_scale)
        .round()
        .clamp(int8_info.min, int8_info.max)
        .to(torch.int8)
    )
    return (
        quantized.reshape(original_shape),
        scale.reshape(*original_shape[:-1], 1),
    )


def _normalize_channel_scale(
    weight_scale: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    if weight_scale.numel() != out_features:
        raise ValueError(
            "weight_scale must contain one value per output channel; "
            f"expected {out_features}, got {weight_scale.numel()}"
        )
    return weight_scale.reshape(1, out_features).to(torch.float32)


def w8a8_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference dynamic-token/per-channel W8A8 linear operation."""
    if weight.ndim != 2 or weight.dtype != torch.int8:
        raise ValueError("weight must be a 2D int8 tensor")
    if x.ndim < 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("x and weight have incompatible input dimensions")

    x_2d = x.reshape(-1, x.shape[-1])
    x_q, x_scale = dynamic_per_token_quant_int8(x_2d)
    x_q_2d = x_q.reshape(-1, x_q.shape[-1]).to(torch.int32)
    channel_scale = _normalize_channel_scale(weight_scale, weight.shape[0])
    accumulator = x_q_2d @ weight.to(torch.int32).t()
    output = accumulator.to(torch.float32)
    output = output * x_scale.reshape(-1, 1) * channel_scale
    if bias is not None:
        if bias.numel() != weight.shape[0]:
            raise ValueError("bias must contain one value per output channel")
        output = output + bias.reshape(1, -1).to(output.dtype)
    return output.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])


__all__ = [
    "dynamic_per_token_quant_int8",
    "unpack_uint8b128_int32",
    "w8a8_linear_reference",
]

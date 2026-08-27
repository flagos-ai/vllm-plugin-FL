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
"""Backend-neutral FlagGems/Triton kernels for dynamic W8A8 inference."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@triton.jit
def _round_half_to_even(x):
    """Match torch.round without relying on a vendor-specific libdevice."""
    floor = tl.floor(x)
    fraction = x - floor
    is_odd = tl.abs(floor - 2.0 * tl.floor(floor / 2.0)) > 0.5
    round_up = (fraction > 0.5) | ((tl.abs(fraction - 0.5) < 1e-10) & is_odd)
    return tl.where(round_up, floor + 1.0, floor)


@libentry()
@triton.jit
def _dynamic_per_token_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    values = tl.load(
        input_ptr + token_idx * hidden_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(values), axis=0)
    inv_scale = tl.where(absmax != 0.0, 127.0 / absmax, 0.0)
    quantized = _round_half_to_even(values * inv_scale)
    quantized = tl.minimum(tl.maximum(quantized, -128.0), 127.0)

    tl.store(
        output_ptr + token_idx * hidden_size + offsets,
        quantized.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + token_idx, absmax / 127.0)


def dynamic_per_token_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused symmetric per-token INT8 quantization for ``[tokens, hidden]``."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D [tokens, hidden_size] tensor")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")
    if x.shape[1] == 0:
        raise ValueError("hidden_size must be positive")

    x = x.contiguous()
    output = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(x.shape[1])
    with torch_device_fn.device(x.device):
        _dynamic_per_token_quant_int8_kernel[(x.shape[0],)](
            x,
            output,
            scale,
            hidden_size=x.shape[1],
            BLOCK_SIZE=block_size,
        )
    return output, scale


__all__ = ["dynamic_per_token_quant_int8"]

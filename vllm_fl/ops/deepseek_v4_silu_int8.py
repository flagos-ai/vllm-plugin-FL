# SPDX-License-Identifier: Apache-2.0
"""SwiGLU followed by exact dynamic per-token INT8 quantization."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _round_int8(x):
    return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _silu_mul_dynamic_int8_kernel(
    input_ptr,
    int8_ptr,
    scale_ptr,
    input_stride,
    output_stride,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    row_input = input_ptr + row * input_stride
    gate = tl.load(row_input + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(row_input + hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)

    # Match the native Marlin sequence exactly: SiLU is rounded to BF16 before
    # multiplying by the up projection, and that product is rounded to BF16
    # before dynamic quantization.
    sigmoid = 1.0 / (1.0 + tl.extra.cuda.libdevice.exp(-gate))
    silu_bf16 = (gate * sigmoid).to(tl.bfloat16)
    activated = (silu_bf16.to(tl.float32) * up).to(tl.bfloat16)

    values = tl.where(mask, activated.to(tl.float32), 0.0)
    absmax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-10)
    scale = absmax / 127.0
    quantized = _round_int8(values * (127.0 / absmax))
    row_output = row * output_stride + offsets
    tl.store(int8_ptr + row_output, quantized, mask=mask)
    tl.store(scale_ptr + row, scale)


def silu_mul_dynamic_int8(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse native BF16 SwiGLU semantics with INT8 activation quantization."""
    assert input.ndim == 2 and input.shape[1] % 2 == 0
    rows = input.shape[0]
    hidden_size = input.shape[1] // 2
    int8_output = torch.empty(
        (rows, hidden_size), device=input.device, dtype=torch.int8
    )
    scales = torch.empty((rows, 1), device=input.device, dtype=torch.float32)
    if rows:
        block_size = triton.next_power_of_2(hidden_size)
        num_warps = min(max(block_size // 256, 1), 8)
        _silu_mul_dynamic_int8_kernel[(rows,)](
            input,
            int8_output,
            scales,
            input.stride(0),
            hidden_size,
            hidden_size=hidden_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
    return int8_output, scales

# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference quantization operator implementations."""

from __future__ import annotations

import torch


def modelslim_w8a8_prepare_weight(weight, weight_scale, nz_mode):
    return weight.contiguous(), weight_scale


def modelslim_w8a8_dynamic_linear(x, weight, weight_scale, bias=None):
    """Symmetric per-row int8 quantization, float32 accumulation/dequantization.

    Weight is already transposed [K,N]. This portable numerical oracle uses
    floating accumulation rather than an unavailable portable int8 GEMM. Native
    accumulator/rounding differences require dtype-appropriate tolerance.
    """
    from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8

    quantized, token_scale = dynamic_per_token_quant_int8(x)
    output = (quantized.float() @ weight.float()) * token_scale.float()
    output = output * weight_scale.float().reshape(1, -1)
    if bias is not None:
        output = output + bias.float()
    return output.to(x.dtype)


def dynamic_per_token_quant_int8_torch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8,
    )

    return dynamic_per_token_quant_int8(x)

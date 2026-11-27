# Copyright (c) 2026 BAAI. All rights reserved.
"""Framework weight loading and Linear orchestration for V4.1 projections."""

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)

from vllm_fl.dispatch import resolve_op


class V41OutputProjection(ColumnParallelLinear):
    """Load the original BF16 grouped WO_A weights without quantization."""

    def __init__(self, input_size, output_size, prefix):
        super().__init__(
            input_size,
            output_size,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            return_bias=False,
            prefix=prefix,
        )
        self.grouped_projection = resolve_op("grouped_bf16_linear")

    def forward(self, value):
        # TP <= 8 owns whole output groups; TP16 already gathers one group.
        groups = value.shape[-1] // self.input_size
        if groups <= 1:
            return super().forward(value)
        return self.grouped_projection(value, self.weight, groups)


class _FP32ProjectionMethod(UnquantizedLinearMethod):
    def __init__(self):
        self.project = resolve_op("bf16_linear_fp32")

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("V4.1 FP32 projection does not have bias")
        original = x.shape
        result = self.project(
            x.reshape(-1, original[-1]).contiguous(), layer.weight, layer.unit_scale
        )
        return result.reshape(*original[:-1], layer.weight.shape[0])


class V41FP32Projection(ReplicatedLinear):
    """Replicated BF16 weights with an FP32 result for compressor/router scores."""

    def __init__(self, input_size, output_size, *, prefix):
        super().__init__(
            input_size,
            output_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            return_bias=False,
            prefix=prefix,
        )
        self.register_buffer(
            "unit_scale",
            torch.ones((1,), dtype=torch.float32, device=self.weight.device),
            persistent=False,
        )
        self.quant_method = _FP32ProjectionMethod()


class V41FP32MergedProjection(MergedColumnParallelLinear):
    """Replicated fused checkpoint slices with FP32 output for KV/score."""

    def __init__(self, input_size, output_sizes, *, prefix):
        super().__init__(
            input_size,
            output_sizes,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            return_bias=False,
            disable_tp=True,
            prefix=prefix,
        )
        self.register_buffer(
            "unit_scale",
            torch.ones((1,), dtype=torch.float32, device=self.weight.device),
            persistent=False,
        )
        self.quant_method = _FP32ProjectionMethod()


class V41ClampedSiluAndMul(nn.Module):
    """Shared-expert activation on native gate/up halves via uniform dispatch."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.activation = resolve_op("silu_and_mul_with_clamp")

    def forward(self, inputs):
        if inputs.shape[-1] % 2:
            raise ValueError("shared-expert input must contain equal gate/up halves")
        half = inputs.shape[-1] // 2
        return self.activation(inputs[..., :half], inputs[..., half:], self.limit)


def enable_v41_fp32_gate(layer):
    """Retain the native gate object shared by FusedMoEFactory, fix its GEMM.

    MetaX's native GateLinear tier6 otherwise materializes BF16 logits before
    converting them to FP32. Existing native routing consumes the same module
    and bias parameters; only its LinearMethod is replaced through the public
    projection provider before execution.
    """
    if (
        layer.weight.dtype != torch.bfloat16
        or layer.bias is not None
        or layer.out_dtype != torch.float32
    ):
        raise ValueError("V4.1 gate requires BF16 weight, no bias and FP32 output")
    specialized = (
        "allow_ll_bf16_gemm",
        "allow_dsv3_router_gemm",
        "allow_fp32_router_gemm",
        "allow_bf16x3_router_gemm",
        "allow_cublas_router_gemm",
    )
    if any(getattr(layer, flag, False) for flag in specialized):
        raise NotImplementedError(
            "V4.1 FP32 gate requires the native general Linear path"
        )
    project = _FP32ProjectionMethod()
    layer.register_buffer(
        "unit_scale",
        torch.ones((1,), dtype=torch.float32, device=layer.weight.device),
        persistent=False,
    )
    layer.quant_method = project

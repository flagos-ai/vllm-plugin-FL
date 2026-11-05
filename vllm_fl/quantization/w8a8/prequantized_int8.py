# SPDX-License-Identifier: Apache-2.0
"""Construction-time optional stock-Cutlass INT8 activation consumer.

No op registration, module monkeypatch, role-specific GEMM, loader change, or
communication is performed. Ordinary tensors retain the original native path.
"""

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
)


def _dynamic_symmetric_channelwise(config):
    return (
        config.is_channelwise
        and not config.is_static_input_scheme
        and config.input_symmetric
    )


class PrequantizedDynamicInt8Kernel(CutlassInt8ScaledMMLinearKernel):
    """The native weight processing/GEMM with optional INT8 input reuse."""

    def input_quant_key(self):
        return (
            kInt8DynamicTokenSym
            if _dynamic_symmetric_channelwise(self.config)
            else None
        )

    def apply_weights(self, layer, x, bias=None):
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is None:
            return super().apply_weights(layer, x, bias)
        if self.input_quant_key() is None:
            raise NotImplementedError("dynamic symmetric channelwise consumer required")
        q, scale = qa.data, qa.scale
        shape = qa.orig_shape
        if (
            not isinstance(q, torch.Tensor)
            or not isinstance(scale, torch.Tensor)
            or not isinstance(shape, tuple)
            or len(shape) != 2
            or shape[0] < 1
            or qa.orig_dtype != torch.bfloat16
            or shape[1] < 1
            or shape[1] % 16 != 0
            or q.shape != shape
            or q.dtype != torch.int8
            or not q.is_cuda
            or q.layout != torch.strided
            or q.stride() != (shape[1], 1)
            or q.storage_offset() != 0
            or scale.storage_offset() != 0
            or scale.shape != (shape[0], 1)
            or scale.dtype != torch.float32
            or scale.layout != torch.strided
            or scale.stride() != (1, 1)
            or scale.device != q.device
            or q.requires_grad
            or scale.requires_grad
            or q.is_conj()
            or q.is_neg()
            or scale.is_conj()
            or scale.is_neg()
        ):
            raise NotImplementedError("unsupported prequantized INT8 activation layout")
        weight, weight_scale, input_scale, input_zp, azp_adj = self._get_layer_params(
            layer
        )
        if (
            weight.ndim != 2
            or weight.shape[0] != shape[1]
            or weight.dtype != torch.int8
            or weight.device != q.device
            or weight.shape[1] % 16 != 0
            or weight.storage_offset() != 0
            or weight.layout != torch.strided
            or weight.stride() != (1, shape[1])
            or weight_scale.shape != (weight.shape[1], 1)
            or weight_scale.dtype != torch.float32
            or weight_scale.device != q.device
            or weight_scale.layout != torch.strided
            or weight_scale.stride() != (1, 1)
            or weight_scale.storage_offset() != 0
            or input_scale is not None
            or input_zp is not None
            or azp_adj is not None
            or weight.requires_grad
            or weight_scale.requires_grad
            or weight.is_conj()
            or weight.is_neg()
            or weight_scale.is_conj()
            or weight_scale.is_neg()
        ):
            raise NotImplementedError("native channelwise INT8 weight contract changed")
        # Exact existing native scaled GEMM, output cast, scales and optional
        # bias. Column/row-parallel callers retain their own original comms.
        return ops.cutlass_scaled_mm(
            q,
            weight,
            scale_a=scale,
            scale_b=weight_scale,
            out_dtype=qa.orig_dtype,
            bias=bias,
        )


def maybe_enable_prequantized_int8(kernel, *, enabled):
    """Call once during construction, before normal weight postprocessing.

    The caller resolves policy, role and platform once. This helper performs no
    device query and stores no tensor, stream, forward hook, or environment.
    """
    if (
        not enabled
        or type(kernel) is not CutlassInt8ScaledMMLinearKernel
        or not _dynamic_symmetric_channelwise(kernel.config)
    ):
        return kernel
    return PrequantizedDynamicInt8Kernel(kernel.config, kernel.layer_param_names)


__all__ = ["PrequantizedDynamicInt8Kernel", "maybe_enable_prequantized_int8"]

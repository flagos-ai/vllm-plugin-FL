# SPDX-License-Identifier: Apache-2.0
"""Thin optional gate producer/consumer for the stock QuantizedActivation ABI."""

import torch
from vllm_fl.utils import use_flaggems_op
from torch.nn.modules import module as nn_module

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
)

from .nvidia_oproj import (
    FGVInt8OProjDecodeKernel,
    _graph_or_startup_warm,
    int8_scaled_mm_decode,
)


def _native_dynamic_params(kernel, layer, device):
    c = kernel.config
    if not c.is_channelwise or c.is_static_input_scheme or not c.input_symmetric:
        return None
    weight, scale, input_scale, input_zp, azp_adj = kernel._get_layer_params(layer)
    if (
        weight.shape != (1024, 6144)
        or weight.stride() != (1, 1024)
        or weight.dtype != torch.int8
        or weight.device != device
        or weight.storage_offset() != 0
        or scale.shape != (6144, 1)
        or scale.stride() != (1, 1)
        or scale.dtype != torch.float32
        or scale.device != device
        or scale.storage_offset() != 0
        or input_scale is not None
        or input_zp is not None
        or azp_adj is not None
    ):
        return None
    return weight, scale


def maybe_gate_quant_input(linear, x, factor):
    """Decline before computation unless stock linear/CT/provider contracts hold."""
    if not use_flaggems_op("mul_scaled_int8_quant", default=False):
        return None
    from flaggems_vllm.ops.mul_scaled_int8_quant import (
        mul_scaled_int8_quant,
        supports_mul_scaled_int8_quant,
    )

    if (
        not use_flaggems_op("mul_scaled_int8_quant", default=False)
        or not isinstance(linear, RowParallelLinear)
        or type(linear).forward is not RowParallelLinear.forward
        or not linear.input_is_parallel
        or linear.bias is not None
        or type(linear.quant_method) is not CompressedTensorsLinearMethod
        or not supports_mul_scaled_int8_quant(x, factor)
        or x.shape[0] == 0
        or x.shape[1] != 1024
        or linear._forward_hooks
        or linear._forward_pre_hooks
        or linear._backward_hooks
        or linear._backward_pre_hooks
        or nn_module._global_forward_hooks
        or nn_module._global_forward_pre_hooks
        or nn_module._global_backward_hooks
        or nn_module._global_backward_pre_hooks
    ):
        return None
    scheme = getattr(linear, "scheme", None)
    if (
        not isinstance(scheme, CompressedTensorsW8A8Int8)
        or type(scheme).apply_weights is not CompressedTensorsW8A8Int8.apply_weights
        or type(scheme.kernel) is not FGVInt8OProjDecodeKernel
        or not scheme.kernel._hy4_gate_quant_allowed
        or scheme.kernel.input_quant_key() != kInt8DynamicTokenSym
        or _native_dynamic_params(scheme.kernel, linear, x.device) is None
    ):
        return None
    data, scale = mul_scaled_int8_quant(x, factor)
    return QuantizedActivation(data, scale, x.dtype, x.shape, kInt8DynamicTokenSym)


def apply_prequantized_oproj(kernel, layer, activation, bias):
    """Consume stock INT8 data without changing native weight loading or TP AR."""
    activation = as_quantized_activation(activation, kernel.input_quant_key())
    if activation is None:
        raise TypeError("QuantizedActivation required")
    q, scale = activation.data, activation.scale
    shape = activation.orig_shape
    if (
        not use_flaggems_op("mul_scaled_int8_quant", default=False)
        or activation.quant_key != kInt8DynamicTokenSym
        or activation.orig_dtype != torch.bfloat16
        or not isinstance(shape, tuple)
        or len(shape) != 2
        or not 1 <= shape[0]
        or shape[1] != 1024
        or bias is not None
        or not isinstance(q, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or q.shape != shape
        or q.dtype != torch.int8
        or not q.is_cuda
        or q.stride() != (1024, 1)
        or q.storage_offset() != 0
        or scale.shape != (shape[0], 1)
        or scale.dtype != torch.float32
        or scale.stride() != (1, 1)
        or scale.storage_offset() != 0
        or scale.device != q.device
    ):
        raise NotImplementedError("unsupported prequantized o_proj activation")
    params = _native_dynamic_params(kernel, layer, q.device)
    if params is None:
        raise NotImplementedError("native dynamic channelwise o_proj required")
    weight, weight_scale = params
    # The original FGV kernel selects this GEMM only under the same context/M
    # contract. Prefill and ordinary NONE retain native Cutlass. The stock
    # RowParallelLinear caller performs its original all-reduce afterwards.
    if shape[0] <= 64 and _graph_or_startup_warm():
        return int8_scaled_mm_decode(
            q, weight, scale, weight_scale, out_dtype=activation.orig_dtype
        )
    return ops.cutlass_scaled_mm(
        q,
        weight,
        scale_a=scale,
        scale_b=weight_scale,
        out_dtype=activation.orig_dtype,
        bias=None,
    )


__all__ = ["maybe_gate_quant_input", "apply_prequantized_oproj"]

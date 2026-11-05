# SPDX-License-Identifier: Apache-2.0
"""Opt-in q_a RMSNorm producer and standard INT8 activation handoff.

Only construction policy and metadata dispatch live here. Numeric work belongs
in FlagGems-vllm; weight loading, the native GEMM and TP operations stay stock.
"""

from __future__ import annotations

import inspect
from typing import NamedTuple

import torch
from torch.nn.modules import module as nn_module

from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    expose_input_quant_key,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
)
from vllm.platforms import PlatformEnum, current_platform

from .prequantized_int8 import (
    PrequantizedDynamicInt8Kernel,
    maybe_enable_prequantized_int8,
)
from vllm_fl.dispatch import get_default_manager, get_policy_epoch, is_dump_enabled
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
from vllm_fl.ops.layernorm import RMSNormFL
from vllm_fl.utils import use_flaggems_op


class _Policy(NamedTuple):
    enabled: bool
    manager_id: int
    manager_epoch: int
    policy_epoch: int


def _no_hooks(norm, linear):
    return not any(
        getattr(obj, name)
        for obj in (norm, linear, nn_module)
        for name in (
            (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            )
            if obj is nn_module
            else (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        )
    )


def _original_modules(norm, linear):
    if (
        type(norm) is not RMSNormFL
        or type(linear) is not ColumnParallelLinear
        or type(linear.quant_method) is not CompressedTensorsLinearMethod
        or linear.bias is not None
        or not linear.return_bias
        or norm.variance_size_override is not None
        or not norm.has_weight
        or not norm.pass_weight
        or norm.weight.dtype != torch.bfloat16
        or not 0 < norm.hidden_size <= 4096
        or linear.input_size_per_partition != norm.hidden_size
        or linear.params_dtype != torch.bfloat16
        or not _no_hooks(norm, linear)
    ):
        return False
    # A disabled CustomOp may retain its native/compiled forward method even
    # when its Python object is RMSNormFL. Only the actual OOT path qualifies.
    method = getattr(norm._forward_method, "__func__", None)
    scheme = getattr(linear, "scheme", None)
    return (
        method in (RMSNormFL.forward_cuda, RMSNormFL.forward_oot)
        and isinstance(scheme, CompressedTensorsW8A8Int8)
        and type(scheme).apply_weights is CompressedTensorsW8A8Int8.apply_weights
    )


def configure_q_a_rms_quant(norm, linear):
    """Once after q_b.create_weights, before the original weight postprocess.

    Keep only immutable bool/epoch integers on attention. No tensor, stream,
    runtime activation, callback or per-layer hook is attached or retained.
    """
    if (
        not use_flaggems_op("rms_scaled_int8_quant", default=False)
        or norm is None
        or linear is None
        or current_platform._enum != PlatformEnum.CUDA
        or current_platform.get_device_capability() != (9, 0)
        or not use_flaggems_op("rms_norm")
        or not use_flaggems_op("rms_scaled_int8_quant")
        or is_dump_enabled()
        or not _original_modules(norm, linear)
    ):
        return None
    from flag_gems.modules import normalization

    if normalization.use_c_extension:
        return None
    scheme = linear.scheme
    kernel = scheme.kernel
    if type(kernel) is not CutlassInt8ScaledMMLinearKernel:
        return None
    mgr = get_default_manager()
    fn = inspect.unwrap(mgr.resolve("rms_norm"))
    if (
        getattr(fn, "__func__", None) is not FlagGemsBackend.rms_norm
        or type(getattr(fn, "__self__", None)) is not FlagGemsBackend
    ):
        return None
    selected = maybe_enable_prequantized_int8(kernel, enabled=True)
    if type(selected) is not PrequantizedDynamicInt8Kernel:
        return None
    # Existing normal/packed schemes call this inherited native postprocess
    # after loading. Do not transpose/unpack/change a weight here.
    scheme.kernel = selected
    expose_input_quant_key(linear, selected)
    return _Policy(True, id(mgr), mgr.policy_epoch, get_policy_epoch())


def _native_parameters_match(kernel, linear, x):
    weight, scale, input_scale, input_zp, azp_adj = kernel._get_layer_params(linear)
    k = x.shape[1]
    return (
        weight.ndim == 2
        and weight.shape[0] == k
        and weight.shape[1] % 16 == 0
        and k % 16 == 0
        and weight.dtype == torch.int8
        and weight.device == x.device
        and weight.layout == torch.strided
        and weight.stride() == (1, k)
        and weight.storage_offset() == 0
        and scale.shape == (weight.shape[1], 1)
        and scale.dtype == torch.float32
        and scale.device == x.device
        and scale.layout == torch.strided
        and scale.stride() == (1, 1)
        and scale.storage_offset() == 0
        and not weight.requires_grad
        and not scale.requires_grad
        and not weight.is_conj()
        and not weight.is_neg()
        and not scale.is_conj()
        and not scale.is_neg()
        and input_scale is None
        and input_zp is None
        and azp_adj is None
    )


def maybe_q_a_rms_quant(policy, norm, linear, x):
    """Decline before compute if provider/metadata/compile ABI is unsupported.

    Native FULL, eager and breakable PIECEWISE all use this same path. BCG is
    stream capture, not Dynamo. Only actual torch.compile takes the original
    path, before numeric alias/storage guards can see a FakeTensor.
    """
    if (
        policy is None
        or not policy.enabled
        or torch.compiler.is_compiling()
        or not use_flaggems_op("rms_scaled_int8_quant", default=False)
        or not use_flaggems_op("rms_norm")
    ):
        return None
    mgr = get_default_manager()
    if (
        id(mgr) != policy.manager_id
        or mgr.policy_epoch != policy.manager_epoch
        or get_policy_epoch() != policy.policy_epoch
        or is_dump_enabled()
        or not _original_modules(norm, linear)
    ):
        return None
    from flaggems_vllm import rms_scaled_int8_quant, supports_rms_scaled_int8_quant

    kernel = linear.scheme.kernel
    if (
        type(kernel) is not PrequantizedDynamicInt8Kernel
        or kernel.input_quant_key() != kInt8DynamicTokenSym
        or not supports_rms_scaled_int8_quant(x, norm.weight)
        or x.shape[0] == 0
        or x.shape[1] != norm.hidden_size
        or not _native_parameters_match(kernel, linear, x)
    ):
        return None
    qc, q, scale = rms_scaled_int8_quant(x, norm.weight, norm.variance_epsilon)
    return qc, QuantizedActivation(q, scale, x.dtype, x.shape, kInt8DynamicTokenSym)


__all__ = ["configure_q_a_rms_quant", "maybe_q_a_rms_quant"]

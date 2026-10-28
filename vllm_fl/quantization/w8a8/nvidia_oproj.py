"""Opt-in HY4 o_proj decode adapter, retaining native CUDA selection elsewhere."""

from __future__ import annotations


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
)
from vllm.compilation import monitor
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
)
from vllm.platforms import PlatformEnum, current_platform
from vllm_fl.utils import use_flaggems_op


def int8_scaled_mm_decode(*args, **kwargs):
    """Load the optional public provider only when the selected path runs."""
    from flaggems_vllm.ops.int8_scaled_mm_decode import int8_scaled_mm_decode as impl

    return impl(*args, **kwargs)


def _graph_or_startup_warm():
    if not is_forward_context_available():
        return False
    mode = get_forward_context().cudagraph_runtime_mode
    return mode == CUDAGraphMode.FULL or (
        mode == CUDAGraphMode.NONE and monitor.cudagraph_capturing_enabled
    )


class FGVInt8OProjDecodeKernel(CutlassInt8ScaledMMLinearKernel):
    def __init__(self, c, layer_param_names):
        super().__init__(c, layer_param_names)
        # Per-instance policy snapshot: serving configuration is frozen before
        # construction. Preserve the stock whitelist/blacklist resolver without
        # parsing its platform YAML on every eager/prefill attention forward.
        self._hy4_gate_quant_allowed = use_flaggems_op(
            "mul_scaled_int8_quant", default=False
        )

    def input_quant_key(self):
        return kInt8DynamicTokenSym if self._hy4_gate_quant_allowed else None

    # Native weight postprocessing is inherited unchanged. Only a policy
    # bool is retained; no per-layer tensor/cache/stream is retained.
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if isinstance(x, QuantizedActivation):
            from .gate_quant_input import apply_prequantized_oproj

            return apply_prequantized_oproj(self, layer, x, bias)
        if (
            bias is not None
            or x.ndim != 2
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or not x.is_contiguous()
            or not 1 <= x.shape[0] <= 64
            or x.shape[1] != 1024
            or not _graph_or_startup_warm()
        ):
            return super().apply_weights(layer, x, bias)
        weight, weight_scale, input_scale, input_zp, azp_adj = self._get_layer_params(
            layer
        )
        if (
            weight.shape != (1024, 6144)
            or weight.stride() != (1, 1024)
            or weight.dtype != torch.int8
            or not weight.is_cuda
            or weight.device != x.device
            or weight.storage_offset() != 0
            or weight_scale.shape != (6144, 1)
            or weight_scale.dtype != torch.float32
            or weight_scale.stride() != (1, 1)
            or weight_scale.device != x.device
            or weight_scale.storage_offset() != 0
            or input_scale is not None
            or input_zp is not None
            or azp_adj is not None
        ):
            return super().apply_weights(layer, x, bias)
        # Exactly the selected native CutlassInt8 dynamic symmetric quantizer.
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True
        )
        if x_zp is not None:
            raise RuntimeError("symmetric native INT8 quant returned a zero point")
        return int8_scaled_mm_decode(x_q, weight, x_s, weight_scale, out_dtype=x.dtype)


def maybe_select_nvidia_oproj_kernel(
    kernel, layer_name, input_size_per_partition, output_partition_sizes, params_dtype
):
    if not use_flaggems_op("int8_scaled_mm_decode", default=False):
        return kernel
    if (
        current_platform._enum != PlatformEnum.CUDA
        or current_platform.get_device_capability() != (9, 0)
        or type(kernel) is not CutlassInt8ScaledMMLinearKernel
        or not isinstance(layer_name, str)
        or not layer_name.endswith(".self_attn.o_proj")
        or input_size_per_partition != 1024
        or list(output_partition_sizes) != [6144]
        or params_dtype != torch.bfloat16
    ):
        return kernel
    config = kernel.config
    compilation = get_current_vllm_config().compilation_config
    if (
        not config.is_channelwise
        or config.is_static_input_scheme
        or not config.input_symmetric
        or not use_flaggems_op("int8_scaled_mm_decode")
        or compilation.cudagraph_mode
        not in (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE)
        or compilation.cudagraph_num_of_warmups < 1
    ):
        return kernel
    return FGVInt8OProjDecodeKernel(config, kernel.layer_param_names)


__all__ = ["FGVInt8OProjDecodeKernel", "maybe_select_nvidia_oproj_kernel"]

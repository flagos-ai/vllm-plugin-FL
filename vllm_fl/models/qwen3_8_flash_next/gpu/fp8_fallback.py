# SPDX-License-Identifier: Apache-2.0
"""Load standard FP8 checkpoints into the existing BF16 Linear/MoE paths.

This is model-local loading glue, not a numerical kernel. FP8 values are
converted on the CPU before the vendor weight loader sees them; vLLM's
block_dequant applies checkpoint scales once, before graph capture.
"""

from functools import wraps

import torch

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import block_dequant
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs


def _bf16_weight_loader(loader, layer):
    @wraps(loader)
    def load(param, weight, *args, **kwargs):
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # Some vendor runtimes cannot even copy/cast FP8 device tensors.
            weight = weight.cpu().to(torch.bfloat16)
        result = loader(param, weight, *args, **kwargs)
        layer._qwen_fp8_dequantized = False
        return result

    return load


def _dequantize(weight, scale, block_size):
    # Do this only at load time, on CPU, with the existing vLLM helper.
    # It handles partial edge blocks and accumulates the product in FP32.
    return block_dequant(weight.cpu(), scale.cpu(), block_size).to(
        device=weight.device, dtype=torch.bfloat16
    )


@register_weight_loader_v2_supported_method
class QwenFp8Bf16LinearMethod(UnquantizedLinearMethod):
    def __init__(self, config):
        self.quant_config = config
        self.block_quant = config.weight_block_size is not None

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        loader = _bf16_weight_loader(extra_weight_attrs.pop("weight_loader"), layer)
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            torch.bfloat16,
            weight_loader=loader,
            **extra_weight_attrs,
        )
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = self.quant_config.weight_block_size
        if self.block_quant:
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                layer.weight_block_size,
            )
        scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter if self.block_quant else PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            layer.weight_block_size,
            loader,
        )
        layer.register_parameter(
            "weight_scale_inv" if self.block_quant else "weight_scale", scale
        )
        if self.quant_config.activation_scheme == "static":
            scale = create_fp8_input_scale(output_partition_sizes, loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_qwen_fp8_dequantized", False):
            return
        if self.block_quant:
            layer.weight.data.copy_(
                _dequantize(
                    layer.weight, layer.weight_scale_inv, layer.weight_block_size
                )
            )
        else:
            start = 0
            for index, width in enumerate(layer.logical_widths):
                shard = layer.weight.data[start : start + width]
                shard.copy_(
                    _dequantize(
                        shard,
                        layer.weight_scale[index].reshape(1, 1),
                        list(shard.shape),
                    )
                )
                start += width
        # Preserve the loader attributes on the BF16 parameter. The existing
        # unquantized method owns GEMM selection and any vendor preprocessing.
        super().process_weights_after_loading(layer)
        layer._qwen_fp8_dequantized = True


class QwenFp8Bf16MoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, config, moe):
        super().__init__(moe)
        self.quant_config = config
        self.weight_block_size = config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

    def create_weights(self, layer, **kwargs):
        loader = _bf16_weight_loader(kwargs["weight_loader"], layer)
        device = torch.empty(0).device
        # Reuse upstream's checkpoint names, scale shapes and expert-sharding
        # attributes. Its allocation-only FP8 routine must run on CPU.
        with torch.device("cpu"):
            Fp8MoEMethod.create_weights(
                self, layer, **dict(kwargs, weight_loader=loader)
            )
        for name, parameter in layer.named_parameters(recurse=False):
            if name in ("w13_weight", "w2_weight"):
                parameter.data = torch.empty(
                    parameter.shape, dtype=torch.bfloat16, device=device
                )
            else:
                parameter.data = parameter.data.to(device)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_qwen_fp8_dequantized", False):
            return
        for name in ("w13", "w2"):
            weight = getattr(layer, f"{name}_weight")
            scales = getattr(layer, f"{name}_{self.weight_scale_name}")
            # Bound CPU staging to one expert, including for large MoE layers.
            for index in range(weight.shape[0]):
                expert = weight.data[index]
                if self.block_quant:
                    scale, block = scales[index], self.weight_block_size
                else:
                    scale = scales[index].reshape(-1, 1)
                    block = [expert.shape[0] // scale.shape[0], expert.shape[1]]
                expert.copy_(_dequantize(expert, scale, block))
        super().process_weights_after_loading(layer)
        layer._qwen_fp8_dequantized = True


class QwenFp8Bf16Config(Fp8Config):
    def __init__(self, config):
        # Preserve packed mappings and ignored-layer rules configured by vLLM.
        self.__dict__.update(config.__dict__)
        if self.store_dtype is not None:
            raise NotImplementedError(
                "Qwen BF16 fallback requires ordinary FP8 weights"
            )

    def get_quant_method(self, layer, prefix):
        skipped = is_layer_skipped(
            prefix, self.ignored_layers, self.packed_modules_mapping
        )
        serialized = self.is_checkpoint_fp8_serialized and not skipped
        if isinstance(layer, LinearBase):
            return (
                QwenFp8Bf16LinearMethod(self)
                if serialized
                else UnquantizedLinearMethod()
            )
        if isinstance(layer, RoutedExperts):
            return (
                QwenFp8Bf16MoEMethod(self, layer.moe_config)
                if serialized
                else UnquantizedFusedMoEMethod(layer.moe_config)
            )
        # BF16 attention does not consume FP8 KV scales.
        return None

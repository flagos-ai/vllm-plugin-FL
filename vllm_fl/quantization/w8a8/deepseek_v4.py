# Copyright 2026 FlagOS Contributors

from compressed_tensors.quantization import QuantizationStrategy

from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)

from vllm_fl.dispatch.backends.vendor.metax.patches.scaled_mm import (
    MctlassInt8ScaledMMLinearKernel,
)
from vllm_fl.quantization.w8a8.moe_experts import MetaXTritonW8A8Experts


class DeepseekV4W8A8Int8(CompressedTensorsW8A8Int8):
    def create_weights(self, *args, **kwargs) -> None:
        super().create_weights(*args, **kwargs)
        config = Int8ScaledMMLinearLayerConfig(
            is_channelwise=self.strategy == QuantizationStrategy.CHANNEL,
            is_static_input_scheme=self.is_static_input_scheme,
            input_symmetric=self.input_symmetric,
        )
        self.kernel = MctlassInt8ScaledMMLinearKernel(
            config,
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_zero_point",
                "azp_adj",
            ],
        )


class DeepseekV4W8A8Config(CompressedTensorsConfig):
    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, CompressedTensorsLinearMethod) and type(
            layer.scheme
        ) is CompressedTensorsW8A8Int8:
            scheme = layer.scheme
            layer.scheme = DeepseekV4W8A8Int8(
                scheme.strategy,
                scheme.is_static_input_scheme,
                scheme.input_symmetric,
            )
        elif isinstance(method, CompressedTensorsW8A8Int8MoEMethod):
            method.experts_cls = MetaXTritonW8A8Experts
        return method

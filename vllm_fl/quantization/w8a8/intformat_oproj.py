"""Thin ordinary compressed-tensors INT8 o_proj scheme adapter."""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm_fl.utils import use_flaggems_op
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)

from .int8_mode import is_dynamic_token_int8


class CompressedTensorsIntFormatOProjW8A8Int8(CompressedTensorsW8A8Int8):
    """Retain native storage/loading/apply; select the existing decode kernel."""

    def __init__(self, native_scheme: CompressedTensorsW8A8Int8, layer_name: str):
        super().__init__(
            strategy=native_scheme.strategy,
            is_static_input_scheme=native_scheme.is_static_input_scheme,
            input_symmetric=native_scheme.input_symmetric,
        )
        self.layer_name = layer_name

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ) -> None:
        # Allocate exactly the ordinary native INT8 parameters. No packed
        # int32 reinterpretation, unpack, transpose, or scale conversion here.
        super().create_weights(
            layer=layer,
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
            **kwargs,
        )
        # This selector retains the exact native kernel unless real CUDA SM90,
        # BF16 local K1024/N6144 and FULL_DECODE_ONLY startup gates hold.
        from .nvidia_oproj import maybe_select_nvidia_oproj_kernel

        self.kernel = maybe_select_nvidia_oproj_kernel(
            self.kernel,
            self.layer_name,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
        )


def maybe_wrap_intformat_oproj_scheme(
    scheme,
    weight_quant,
    input_quant,
    quant_format,
    layer_name,
    output_quant=None,
):
    """Opt in only the native ordinary W8A8 o_proj scheme, by exact type."""
    if (
        not use_flaggems_op("int8_scaled_mm_decode", default=False)
        or quant_format != CompressionFormat.int_quantized.value
        or not isinstance(layer_name, str)
        or not layer_name.endswith(".self_attn.o_proj")
        or output_quant is not None
        or type(scheme) is not CompressedTensorsW8A8Int8
    ):
        return scheme
    if (
        weight_quant is None
        or weight_quant.num_bits != 8
        or weight_quant.type != QuantizationType.INT
        or weight_quant.strategy != QuantizationStrategy.CHANNEL
        or not weight_quant.symmetric
        or weight_quant.dynamic
        or not is_dynamic_token_int8(input_quant)
        or scheme.strategy != QuantizationStrategy.CHANNEL
        or scheme.is_static_input_scheme
        or not scheme.input_symmetric
    ):
        return scheme
    return CompressedTensorsIntFormatOProjW8A8Int8(scheme, layer_name)


__all__ = [
    "CompressedTensorsIntFormatOProjW8A8Int8",
    "maybe_wrap_intformat_oproj_scheme",
]

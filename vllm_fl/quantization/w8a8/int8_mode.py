# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Selection policy for FlagOS-Compressor packed INT8 checkpoints."""

from __future__ import annotations

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)


def is_packed_int8_weight(
    weight_quant: QuantizationArgs | None,
    quant_format: str | None,
) -> bool:
    """Match supported packed INT8 weight metadata."""
    if weight_quant is None:
        return False
    return (
        quant_format == CompressionFormat.pack_quantized.value
        and weight_quant.num_bits == 8
        and weight_quant.type == QuantizationType.INT
        and weight_quant.symmetric
        and not weight_quant.dynamic
        and weight_quant.strategy
        in (QuantizationStrategy.CHANNEL, QuantizationStrategy.GROUP)
    )


def is_dynamic_token_int8(input_quant: QuantizationArgs | None) -> bool:
    """Match canonical dynamic per-token INT8 activation metadata."""
    if input_quant is None:
        return False
    return (
        input_quant.num_bits == 8
        and input_quant.type == QuantizationType.INT
        and input_quant.strategy == QuantizationStrategy.TOKEN
        and input_quant.symmetric
        and input_quant.dynamic
    )


def should_use_packed_w8a8(
    weight_quant: QuantizationArgs | None,
    input_quant: QuantizationArgs | None,
    quant_format: str | None,
) -> bool:
    """Select packed W8A8 from the model's compressed-tensors config."""
    if not is_packed_int8_weight(weight_quant, quant_format):
        return False
    if input_quant is None:
        return False
    if not is_dynamic_token_int8(input_quant):
        return False
    if weight_quant.strategy != QuantizationStrategy.CHANNEL:
        raise ValueError(
            "Packed W8A8 requires per-channel weights in quantization_config"
        )
    return True


__all__ = [
    "is_dynamic_token_int8",
    "is_packed_int8_weight",
    "should_use_packed_w8a8",
]

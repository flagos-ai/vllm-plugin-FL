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

import os

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

INT8_MODE_ENV = "VLLM_FL_INT8_MODE"
_VALID_MODES = {"auto", "w8a8", "w8a16"}


def get_int8_inference_mode() -> str:
    """Return the requested runtime activation mode for packed INT8 weights."""
    mode = os.getenv(INT8_MODE_ENV, "auto").strip().lower()
    if mode not in _VALID_MODES:
        choices = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"{INT8_MODE_ENV} must be one of {choices}, got {mode!r}")
    return mode


def is_packed_int8_weight_only(
    weight_quant: QuantizationArgs | None,
    input_quant: QuantizationArgs | None,
    quant_format: str | None,
) -> bool:
    """Match the compressed-tensors contract emitted by FlagOS-Compressor."""
    if weight_quant is None:
        return False
    return (
        quant_format == CompressionFormat.pack_quantized.value
        and input_quant is None
        and weight_quant.num_bits == 8
        and weight_quant.type == QuantizationType.INT
        and weight_quant.symmetric
        and not weight_quant.dynamic
        and weight_quant.strategy
        in (QuantizationStrategy.CHANNEL, QuantizationStrategy.GROUP)
    )


def should_use_packed_w8a8(
    weight_quant: QuantizationArgs | None,
    input_quant: QuantizationArgs | None,
    quant_format: str | None,
) -> bool:
    """Select W8A8 for channelwise packed weights in auto/W8A8 mode."""
    if not is_packed_int8_weight_only(
        weight_quant,
        input_quant,
        quant_format,
    ):
        return False
    mode = get_int8_inference_mode()
    if mode == "w8a8" and weight_quant.strategy != QuantizationStrategy.CHANNEL:
        raise ValueError(
            "Packed W8A8 requires FlagOS-Compressor --strategy channel; "
            "groupwise INT8 weights can run only in W8A16 mode"
        )
    return mode != "w8a16" and weight_quant.strategy == QuantizationStrategy.CHANNEL


__all__ = [
    "INT8_MODE_ENV",
    "get_int8_inference_mode",
    "is_packed_int8_weight_only",
    "should_use_packed_w8a8",
]

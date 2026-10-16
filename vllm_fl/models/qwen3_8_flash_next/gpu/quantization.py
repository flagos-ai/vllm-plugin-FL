# SPDX-License-Identifier: Apache-2.0
"""Quantization policy shared by model and QSA owners."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationConfig


def without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Keep the QSA QKV projection unquantized for ModelOpt FP4.

    Preserve the reference implementation's per-layer exclusion. Other
    quantization configs pass through unchanged; this helper does not establish
    full-model support for any quantized checkpoint.
    """

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


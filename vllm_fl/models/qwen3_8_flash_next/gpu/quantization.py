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


def with_platform_quantization(vllm_config):
    """Select BF16 only for this model when the platform lacks native FP8."""
    import torch

    from vllm.logger import init_logger
    from vllm.platforms import current_platform

    quant_config = getattr(vllm_config, "quant_config", None)
    cache_config = vllm_config.cache_config
    fp8_cache = str(getattr(cache_config, "cache_dtype", "auto")).startswith("fp8")
    fp8_weights = quant_config is not None and quant_config.get_name() == "fp8"
    if not (fp8_weights or fp8_cache) or current_platform.supports_fp8():
        return vllm_config

    from .fp8_fallback import QwenFp8Bf16Config

    if isinstance(quant_config, QwenFp8Bf16Config) and not fp8_cache:
        return vllm_config
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Qwen FP8 fallback requires --dtype bfloat16")
    if fp8_weights:
        vllm_config.quant_config = QwenFp8Bf16Config(quant_config)
    if fp8_cache:
        # The runner allocates caches after model construction from this same
        # config. It must see the same dtype as the attention layer.
        cache_config.cache_dtype = "bfloat16"
    init_logger(__name__).info_once(
        "Qwen FP8 is unavailable on this platform; using existing BF16 "
        "Linear/MoE and KV-cache paths. Serialized FP8 weights are "
        "dequantized once during loading; resident weights use BF16 memory."
    )
    return vllm_config

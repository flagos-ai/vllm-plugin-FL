# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen4Exp's BF16 fallback on devices without FP8 support."""

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def configure_bf16_cache(vllm_config) -> None:
    if current_platform.supports_fp8():
        return
    cache = vllm_config.cache_config
    attention = vllm_config.attention_config
    if cache.cache_dtype.startswith("fp8"):
        cache.cache_dtype = "bfloat16"
        logger.info_once("Qwen4Exp is using BF16 KV cache because FP8 is unavailable.")
    if attention.indexer_kv_dtype == "fp8":
        attention.indexer_kv_dtype = "bf16"
        logger.info_once(
            "Qwen4Exp is using BF16 QSA indexer cache because FP8 is unavailable."
        )


def configure_bf16_weights(vllm_config) -> None:
    if current_platform.vendor_name != "thead" or current_platform.supports_fp8():
        return
    quant_config = vllm_config.quant_config
    if quant_config is None or quant_config.get_name() != "fp8":
        return

    import torch

    from .fp8_fallback import QwenFp8Bf16Config

    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Qwen4Exp FP8 weight fallback requires --dtype bfloat16.")
    if not isinstance(quant_config, QwenFp8Bf16Config):
        vllm_config.quant_config = QwenFp8Bf16Config(quant_config)
        logger.info_once("Qwen4Exp will load FP8 weights into BF16 parameters.")

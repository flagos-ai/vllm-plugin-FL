# Copyright (c) 2025 BAAI. All rights reserved.


import os
import logging
from vllm_fl.utils import get_op_config as _get_op_config


logger = logging.getLogger(__name__)


def register():
    """Register the FL platform."""

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()
    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry

    try:
        from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM  # noqa: F401

        ModelRegistry.register_model(
            "Qwen3NextForCausalLM", "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except ImportError:
        logger.info(
            "From vllm_fl.models.qwen3_next cannot import Qwen3NextForCausalLM, skipped"
        )
    except Exception as e:
        logger.error(f"Register model error: {str(e)}")

    ModelRegistry.register_model(
        "MiniCPMO",
        "vllm_fl.models.minicpmo:MiniCPMO")

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")

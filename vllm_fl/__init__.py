# Copyright (c) 2025 BAAI. All rights reserved.


import os
import logging
from vllm_fl.utils import get_op_config as _get_op_config


logger = logging.getLogger(__name__)


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

    # Register Qwen3Next model
    try:
        ModelRegistry.register_model(
            "Qwen3NextForCausalLM", 
            "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register Qwen3Next model error: {str(e)}")

    # Register MiniCPMO model
    try:
        ModelRegistry.register_model(
            "MiniCPMO",
            "vllm_fl.models.minicpmo:MiniCPMO"
        )
    except Exception as e:
        logger.error(f"Register MiniCPMO model error: {str(e)}")

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")

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


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def register():
    """Register the FL platform."""
    _patch_transformers_compat()

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()
    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry

    # Register Qwen3.5 MoE config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.qwen3_5_moe import Qwen3_5MoeConfig
        _CONFIG_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeConfig
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE config error: {str(e)}")

    # Register Qwen3Next model
    try:
        ModelRegistry.register_model(
            "Qwen3NextForCausalLM",
            "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register Qwen3Next model error: {str(e)}")

    # Register Qwen3.5 MoE model
    try:
        ModelRegistry.register_model(
            "Qwen3_5MoeForConditionalGeneration",
            "vllm_fl.models.qwen3_5:Qwen3_5MoeForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register Qwen3.5 MoE model error: {str(e)}")

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

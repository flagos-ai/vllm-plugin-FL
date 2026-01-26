# Copyright (c) 2025 BAAI. All rights reserved.


import os
import logging

logger = logging.getLogger(__name__)


def register():
    """Register the FL platform."""

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
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

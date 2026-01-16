# Copyright (c) 2025 BAAI. All rights reserved.

def register():
    """Register the FL platform."""

    return "vllm_fl.platform.PlatformFL"


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry
    from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM
    ModelRegistry.register_model(
        "Qwen3NextForCausalLM",
        "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM")
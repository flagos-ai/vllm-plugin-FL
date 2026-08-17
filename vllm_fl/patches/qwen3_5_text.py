# Copyright (c) 2026 BAAI. All rights reserved.
"""Install canonical Qwen3.5 text-only model support at vLLM runtime.

The target vLLM release already provides Qwen3.5 config and model classes. This
module only fills in the missing text-config, causal-model, and verifier
registrations without modifying the vLLM installation.
"""

import logging

from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
)

logger = logging.getLogger(__name__)

_ARCHITECTURES = {
    "Qwen3_5ForCausalLM": "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM": "Qwen3_5MoeForCausalLM",
}


def apply_qwen3_5_text_patches() -> bool:
    """Register canonical Qwen3.5 text-only causal models.

    Repeated calls are safe. Returns ``True`` after installing the runtime
    registrations.
    """
    from vllm.model_executor.models import (
        config as model_config,
        registry as model_registry,
    )
    from vllm.transformers_utils import config as transformers_config

    config_registry = transformers_config._CONFIG_REGISTRY
    config_registry.setdefault("qwen3_5_text", "Qwen3_5TextConfig")
    config_registry.setdefault("qwen3_5_moe_text", "Qwen3_5MoeTextConfig")

    for architecture in _ARCHITECTURES:
        # The causal and conditional variants share the same hybrid-cache
        # verification. Canonical text-only configs already carry causal
        # architecture names and valid RoPE parameters, so no config mutation
        # or architecture conversion is needed here.
        model_config.MODELS_CONFIG_MAP[architecture] = (
            Qwen3_5ForConditionalGenerationConfig
        )

    # Keep the source registries coherent for introspection, then overwrite the
    # already-materialized ModelRegistry with a lazy plugin path.  Importing
    # that path applies the class shim without initializing CUDA here.
    for architecture, class_name in _ARCHITECTURES.items():
        model_registry._TEXT_GENERATION_MODELS.setdefault(
            architecture, ("qwen3_5", class_name)
        )
        model_registry._VLLM_MODELS.setdefault(architecture, ("qwen3_5", class_name))
        model_registry.ModelRegistry.register_model(
            architecture, f"vllm_fl.models.qwen3_5:{class_name}"
        )

    logger.info("Installed Qwen3.5 text-only causal model runtime compatibility")
    return True


__all__ = ["apply_qwen3_5_text_patches"]

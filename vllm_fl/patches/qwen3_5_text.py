# Copyright (c) 2026 BAAI. All rights reserved.
"""Install Qwen3.5 text-only causal model support at vLLM runtime.

The compatibility layer mirrors the text-only config, model, and checkpoint
handling missing from upstream vLLM without modifying the vLLM installation.
"""

import logging

from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

logger = logging.getLogger(__name__)

_ARCHITECTURES = {
    "Qwen3_5ForCausalLM": "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM": "Qwen3_5MoeForCausalLM",
}
_CONDITIONAL_TO_CAUSAL = {
    "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
}
_DEFAULT_ARCHITECTURES = {
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",
}


class Qwen3_5TextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Normalize multimodal architecture names in text-only Qwen configs."""

    def get_architectures(self) -> list[str]:
        architectures = super().get_architectures()
        if not architectures:
            default = _DEFAULT_ARCHITECTURES.get(self.hf_config.model_type)
            normalized = [default] if default is not None else architectures
        else:
            normalized = [
                _CONDITIONAL_TO_CAUSAL.get(arch, arch) for arch in architectures
            ]

        # The runtime model loader consults hf_config.architectures again after
        # ModelArchitectureConfig has been built. Keep both views synchronized
        # so it cannot fall back to a stale VL architecture.
        if normalized != architectures:
            self.hf_config.architectures = normalized.copy()
        return normalized


class Qwen3_5ForCausalLMConfig(Qwen3_5ForConditionalGenerationConfig):
    """Use the upstream cache config checks and remove multimodal RoPE keys."""

    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)

        hf_text_config = vllm_config.model_config.hf_text_config
        rope_parameters = getattr(hf_text_config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)


def apply_qwen3_5_text_patches() -> bool:
    """Register Qwen3.5 text-only causal model support.

    Repeated calls are safe. Returns ``True`` after installing the runtime
    registrations.
    """
    from vllm.model_executor.models import (
        config as model_config,
        registry as model_registry,
    )
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    config_registry = transformers_config._CONFIG_REGISTRY
    config_registry.setdefault("qwen3_5_text", "Qwen3_5TextConfig")
    config_registry.setdefault("qwen3_5_moe_text", "Qwen3_5MoeTextConfig")

    convertors = model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS
    convertors["qwen3_5_text"] = Qwen3_5TextModelArchConfigConvertor
    convertors["qwen3_5_moe_text"] = Qwen3_5TextModelArchConfigConvertor

    for architecture in _ARCHITECTURES:
        model_config.MODELS_CONFIG_MAP[architecture] = Qwen3_5ForCausalLMConfig

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


__all__ = [
    "Qwen3_5ForCausalLMConfig",
    "Qwen3_5TextModelArchConfigConvertor",
    "apply_qwen3_5_text_patches",
]

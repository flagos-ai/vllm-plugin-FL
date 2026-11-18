# SPDX-License-Identifier: Apache-2.0
from transformers import AutoConfig

from vllm.config.engram import _NGRAM_LAYER_FIELDS
from vllm.model_executor.models.registry import ModelRegistry
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry

from .configs import HYV4FlashConfig, HYV4FlashVLConfig


def register_hy4_flash():
    AutoConfig.register("hy_v4_flash", HYV4FlashConfig, exist_ok=True)
    AutoConfig.register("hy_v4_flash_vl", HYV4FlashVLConfig, exist_ok=True)
    # Vendor Flash aliases hy_v4 to the standard cached Hugging Face tokenizer.
    # vLLM 0.32.0 has the implementation but not this model-specific alias.
    if "hy_v4" not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register("hy_v4", "vllm.tokenizers.hf", "CachedHfTokenizer")
    if "hy_v4" not in RENDERER_REGISTRY.renderers:
        RENDERER_REGISTRY.register(
            "hy_v4", "vllm_fl.models.hy_v4_flash.renderer", "HYV4FlashRenderer"
        )
    # vLLM 0.32 exposes an architecture-to-Engram-field table; register this
    # plugin model so the existing configuration validation resolves storage.
    nvidia_models = "vllm_fl.models.hy_v4_flash.nvidia"
    for architecture, module in (
        ("HYV4FlashForCausalLM", "model"),
        ("HYV4FlashVLForConditionalGeneration", "vl_model"),
    ):
        _NGRAM_LAYER_FIELDS[architecture] = "engram_layer_ids"
        ModelRegistry.register_model(
            architecture, f"{nvidia_models}.{module}:{architecture}"
        )

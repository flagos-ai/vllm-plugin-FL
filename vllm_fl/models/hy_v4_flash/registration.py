# SPDX-License-Identifier: Apache-2.0
from transformers import AutoConfig

from vllm.model_executor.models.registry import ModelRegistry

from .configs import HYV4FlashConfig, HYV4FlashVLConfig


def register_hy4_flash():
    AutoConfig.register("hy_v4_flash", HYV4FlashConfig, exist_ok=True)
    AutoConfig.register("hy_v4_flash_vl", HYV4FlashVLConfig, exist_ok=True)
    # Vendor Flash aliases hy_v4 to the standard cached Hugging Face tokenizer.
    # vLLM 0.32.0 has the implementation but not this model-specific alias.
    from vllm.tokenizers.registry import TokenizerRegistry

    if "hy_v4" not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register("hy_v4", "vllm.tokenizers.hf", "CachedHfTokenizer")
    from vllm.renderers.registry import RENDERER_REGISTRY

    if "hy_v4" not in RENDERER_REGISTRY.renderers:
        RENDERER_REGISTRY.register(
            "hy_v4", "vllm_fl.models.hy_v4_flash.renderer", "HYV4FlashRenderer"
        )
    # vLLM 0.32 exposes an architecture-to-Engram-field table; register this
    # plugin model so the existing configuration validation resolves storage.
    from vllm.config.engram import _NGRAM_LAYER_FIELDS

    _NGRAM_LAYER_FIELDS["HYV4FlashForCausalLM"] = "engram_layer_ids"
    ModelRegistry.register_model(
        "HYV4FlashForCausalLM",
        "vllm_fl.models.hy_v4_flash.nvidia.model:HYV4FlashForCausalLM",
    )

    _NGRAM_LAYER_FIELDS["HYV4FlashVLForConditionalGeneration"] = "engram_layer_ids"
    ModelRegistry.register_model(
        "HYV4FlashVLForConditionalGeneration",
        "vllm_fl.models.hy_v4_flash.nvidia.vl_model:HYV4FlashVLForConditionalGeneration",
    )
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    ReasoningParserManager.register_lazy_module(
        "hy_v4", "vllm_fl.models.hy_v4_flash.reasoning_parser", "HYV4ReasoningParser"
    )
    ToolParserManager.register_lazy_module(
        "hy_v4", "vllm_fl.models.hy_v4_flash.tool_parser", "HYV4ToolParser"
    )

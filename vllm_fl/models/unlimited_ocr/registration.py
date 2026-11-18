# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unlimited-OCR checkpoint configuration and general-plugin registration."""

import inspect
import threading

from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config

from . import ARCHITECTURE, CHECKPOINT_MODEL_TYPE, MODEL_TYPE

_MODEL_MODULE = "vllm_fl.models.unlimited_ocr.model"
_MODEL_PATH = f"{_MODEL_MODULE}:{ARCHITECTURE}"
_REGISTRATION_LOCK = threading.Lock()


class UnlimitedOCRConfig(DeepseekVLV2Config):
    model_type = MODEL_TYPE

    def __init__(
        self,
        tile_tag: str = "2D",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int], ...] = ((384, 384),),
        rswa_window: int | None = 128,
        **kwargs,
    ):
        kwargs.setdefault("architectures", [ARCHITECTURE])
        super().__init__(
            tile_tag=tile_tag,
            global_view_pos=global_view_pos,
            candidate_resolutions=candidate_resolutions,
            **kwargs,
        )
        self.rswa_window = rswa_window if rswa_window is not None else 128
        text = self.text_config
        text.architectures = ["DeepseekV2ForCausalLM"]
        # Ordinary sliding-window caching would evict the prompt. The FL
        # R-SWA backend masks decode tokens while retaining the entire KV cache.
        text.sliding_window = None
        if text.qk_nope_head_dim == text.qk_rope_head_dim == 0:
            # Match the native DeepseekAttention projection dimension.
            text.head_dim = text.hidden_size // text.num_attention_heads


class UnlimitedOCRCheckpointConfig(UnlimitedOCRConfig):
    # AutoConfig requires each registered class's model_type to match its key.
    model_type = CHECKPOINT_MODEL_TYPE
    # Transformers v5 checks the concrete class for an explicit initializer.
    __init__ = UnlimitedOCRConfig.__init__


def register_unlimited_ocr_support() -> None:
    """Install the scoped HF aliases and the lazy vLLM model entry."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.tokenizers import registry as tokenizer_registry
    from vllm.transformers_utils import config as transformers_config

    # vLLM 0.24 exposes model registration publicly. Its HF config aliases
    # and incorrect-tokenizer entries are dictionary/set extension points.
    aliases = getattr(transformers_config, "_CONFIG_REGISTRY", None)
    tokenizer_types = getattr(
        tokenizer_registry, "_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS", None
    )
    models = getattr(ModelRegistry, "models", None)
    if not isinstance(aliases, dict) or not isinstance(tokenizer_types, set):
        raise RuntimeError("Required vLLM OCR configuration registries are unavailable")
    if not isinstance(models, dict):
        raise RuntimeError("Required vLLM model registry is unavailable")
    try:
        inspect.signature(ModelRegistry.register_model).bind(ARCHITECTURE, _MODEL_PATH)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Incompatible vLLM ModelRegistry.register_model API"
        ) from exc

    config_entries = {
        MODEL_TYPE: UnlimitedOCRConfig,
        CHECKPOINT_MODEL_TYPE: UnlimitedOCRCheckpointConfig,
    }
    with _REGISTRATION_LOCK:
        existing = {name: aliases.get(name) for name in config_entries}
        for name, cls in config_entries.items():
            if existing[name] is not None and existing[name] is not cls:
                raise RuntimeError(f"OCR config is already registered for {name!r}")
        model = models.get(ARCHITECTURE)
        if model is not None and (
            getattr(model, "module_name", None) != _MODEL_MODULE
            or getattr(model, "class_name", None) != ARCHITECTURE
        ):
            raise RuntimeError(f"Another model is registered for {ARCHITECTURE}")
        tokenizer_entries = config_entries.keys() - tokenizer_types

        try:
            aliases.update(config_entries)
            # The checkpoint labels its fast tokenizer as LlamaTokenizerFast,
            # but the serialized tokenizer uses ByteLevel and must stay intact.
            tokenizer_types.update(tokenizer_entries)
            if model is None:
                ModelRegistry.register_model(ARCHITECTURE, _MODEL_PATH)
        except BaseException:
            if model is None:
                models.pop(ARCHITECTURE, None)
            tokenizer_types.difference_update(tokenizer_entries)
            for name, cls in existing.items():
                if cls is None:
                    aliases.pop(name, None)
                else:
                    aliases[name] = cls
            raise

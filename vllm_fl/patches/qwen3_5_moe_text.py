# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE text-only (flat config) support for vLLM.

Background
----------
Some Qwen3.5-MoE W8A8 checkpoints ship in a *flat / text-only* layout::

    architectures: ["Qwen3_5MoeForCausalLM"]
    model_type:    "qwen3_5_moe_text"
    <all text fields at top level, no text_config / vision_config>
    weights named: model.layers.N.*, model.embed_tokens, model.norm

whereas the layout vLLM 0.24 expects is the multimodal *wrapper*::

    architectures: ["Qwen3_5MoeForConditionalGeneration"]
    model_type:    "qwen3_5_moe"  + nested text_config / vision_config
    weights named: model.language_model.layers.N.*, ...

Two independent blockers result, both patched here.

1. Config type
   ``vllm.transformers_utils.config._CONFIG_REGISTRY`` is keyed on the
   ``model_type`` string and only registers ``qwen3_5_moe``.  A flat config
   declaring ``qwen3_5_moe_text`` misses the registry, falls through to stock
   ``AutoConfig``, and yields ``transformers ... Qwen3_5MoeTextConfig``.
   ``Qwen3_5MoeProcessingInfo.get_hf_config`` then raises::

       TypeError: Invalid type of HuggingFace config.
       Expected type: vllm...Qwen3_5MoeConfig,
       but found type: transformers...Qwen3_5MoeTextConfig

   Fix: register a bridge class under ``qwen3_5_moe_text`` that re-nests the
   flat body into ``text_config`` and subclasses vLLM's ``Qwen3_5MoeConfig``,
   so the isinstance check passes.

2. Weight names
   ``Qwen3_5MoeForCausalLM`` is not a registered architecture; vLLM's
   suffix-fallback rewires it to ``Qwen3_5MoeForConditionalGeneration``.  That
   class inherits ``hf_to_vllm_mapper`` from ``Qwen3VLForConditionalGeneration``
   which only rewrites the ``model.language_model.`` prefix.  Flat-layout
   tensors named ``model.layers.*`` match no rule, pass through unchanged, and
   never line up with the module tree (``language_model.model.layers.*``).

   Fix: install an extended mapper carrying an additional catch-all
   ``"model." -> "language_model.model."`` rule.

Ordering note (important)
-------------------------
``WeightsMapper._map_name`` iterates **every** entry of ``orig_to_new_prefix``
in insertion order and rewrites the key in place as it goes -- it does not stop
at the first match.  The catch-all ``"model."`` rule therefore MUST come last:
``model.visual.`` and ``model.language_model.`` are consumed by their specific
rules first, so an already-rewritten key no longer starts with ``model.`` and
is left alone.  Reversing the order would corrupt those keys.

Because the specific rules are preserved and only a trailing fallback is added,
wrapper-layout checkpoints (e.g. Qwen-0810-W8A8) map exactly as before.

Scope
-----
The mapper is patched on ``Qwen3_5MoeForConditionalGeneration`` -- which does
NOT define ``hf_to_vllm_mapper`` itself -- rather than on the shared
``Qwen3VLForConditionalGeneration`` base.  Setting it on the subclass shadows
the inherited attribute and leaves every other Qwen3-VL model untouched.
"""

import logging

logger = logging.getLogger(__name__)

# Layout of the flat text-only config: these keys live at the top level and
# belong inside ``text_config``. Everything else stays at the wrapper level.
_NON_TEXT_KEYS = frozenset(
    {
        "architectures",
        "model_type",
        "quantization_config",
        "torch_dtype",
        "dtype",
        "transformers_version",
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "language_model_only",
        "tie_word_embeddings",
        "_name_or_path",
        "auto_map",
    }
)


def _build_text_only_config_cls():
    """Build a config class bridging the flat layout onto Qwen3_5MoeConfig."""
    from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig

    class Qwen3_5MoeTextOnlyConfig(Qwen3_5MoeConfig):
        """Accept a flat qwen3_5_moe_text config; present it as the wrapper.

        Subclassing Qwen3_5MoeConfig is what makes the strict isinstance check
        in ``multimodal/processing/context.py`` succeed.
        """

        model_type = "qwen3_5_moe_text"

        def __init__(self, **kwargs):
            # Already-nested input (e.g. a re-load of our own saved config):
            # pass straight through, nothing to re-nest.
            if kwargs.get("text_config") is None:
                text_config = {
                    k: v for k, v in kwargs.items() if k not in _NON_TEXT_KEYS
                }
                if text_config:
                    # The nested body must identify as the text model type so
                    # Qwen3_5DecoderLayer picks the sparse-MoE branch
                    # (qwen3_5.py: `if config.model_type == "qwen3_5_moe_text"`).
                    text_config.setdefault("model_type", "qwen3_5_moe_text")
                    for k in text_config:
                        kwargs.pop(k, None)
                    kwargs["text_config"] = text_config

            # The checkpoint carries no vision weights; the tower is replaced by
            # a StageMissingLayer placeholder when --language-model-only is set,
            # so no vision weights are ever requested. We still build a
            # vision_config whose out_hidden_size matches the text hidden_size
            # (8192) -- exactly as the known-good wrapper config (Qwen-0810)
            # does -- so any dim reference during processor/tower construction
            # sees a consistent value rather than the class default (3584).
            if kwargs.get("vision_config") is None:
                tc = kwargs.get("text_config") or {}
                hidden = tc.get("hidden_size")
                if hidden is not None:
                    kwargs["vision_config"] = {"out_hidden_size": hidden}

            super().__init__(**kwargs)

    return Qwen3_5MoeTextOnlyConfig


def patch_qwen3_5_moe_text_config():
    """Register the flat ``qwen3_5_moe_text`` model_type with vLLM."""
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        if "qwen3_5_moe_text" in _CONFIG_REGISTRY:
            return

        _CONFIG_REGISTRY["qwen3_5_moe_text"] = _build_text_only_config_cls()
        logger.info("FL: registered config for model_type=qwen3_5_moe_text")
    except Exception as e:  # pragma: no cover - defensive, mirrors plugin style
        logger.error(f"FL: register qwen3_5_moe_text config failed: {e}")


def patch_qwen3_5_moe_weight_mapper():
    """Teach Qwen3.5-MoE to load flat ``model.*`` weight names."""
    try:
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5MoeForConditionalGeneration as _Cls,
        )
        from vllm.model_executor.models.utils import WeightsMapper

        if getattr(_Cls, "_fl_patched_text_mapper", False):
            return

        base = _Cls.hf_to_vllm_mapper

        # Preserve every existing rule, then append the catch-all LAST.
        # dicts keep insertion order, and _map_name honours that order.
        prefix = dict(base.orig_to_new_prefix)
        prefix["model."] = "language_model.model."

        _Cls.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_renamings=list(base.orig_to_new_renamings),
            orig_to_new_regex=dict(base.orig_to_new_regex),
            orig_to_new_substr=dict(base.orig_to_new_substr),
            orig_to_new_prefix=prefix,
            orig_to_new_suffix=dict(base.orig_to_new_suffix),
        )
        _Cls._fl_patched_text_mapper = True
        logger.info("FL: extended Qwen3.5-MoE weight mapper for flat layout")
    except Exception as e:  # pragma: no cover
        logger.error(f"FL: patch qwen3_5 weight mapper failed: {e}")


def apply_model_patches():
    """Entry point called from ``vllm_fl.register_model``."""
    patch_qwen3_5_moe_text_config()
    patch_qwen3_5_moe_weight_mapper()

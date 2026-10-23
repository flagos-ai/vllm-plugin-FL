# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.
"""Configuration bridge for checkpoints with ``model_type=hy_v4_vl``.

Ported from the reference HY4-VL implementation
(``vllm/transformers_utils/configs/hy_v4.py`` in the upstream HY4 VL tree) so
that FlagOS hosts can load the vision-language checkpoint without remote code.
The text half reuses :class:`vllm_fl.models.hy4.config.HYV4Config`, which is
what the paired vLLM runtime already knows how to instantiate.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from .config import HYV4Config


class HYV4VLVisionConfig(PretrainedConfig):
    """Vision tower and projector configuration for HY4 VL."""

    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        out_hidden_size: int | None = None,
        hidden_act: str = "gelu",
        rms_norm_eps: float = 1e-5,
        max_image_size: int = 2048,
        interpolate_mode: str = "bilinear",
        cat_extra_token: int = 0,
        perceive_pre_norm: bool = True,
        perceive_post_norm: bool = True,
        projector_type: str | None = None,
        perceive_intermediate_size: int | None = None,
        use_image_newline_token: bool = True,
        use_2d_rope: bool = False,
        rope_2d_theta: float = 10000.0,
        rope_2d_percent: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_image_size = max_image_size
        self.interpolate_mode = interpolate_mode
        self.cat_extra_token = cat_extra_token
        self.perceive_pre_norm = perceive_pre_norm
        self.perceive_post_norm = perceive_post_norm
        self.projector_type = projector_type
        self.perceive_intermediate_size = perceive_intermediate_size
        self.use_image_newline_token = use_image_newline_token
        self.use_2d_rope = use_2d_rope
        self.rope_2d_theta = rope_2d_theta
        self.rope_2d_percent = rope_2d_percent


class HYV4VLConfig(PretrainedConfig):
    """Local HYV4 vision/text composition, including flat checkpoint support.

    The released VL checkpoints store the text fields flat at the top level and
    only nest ``vision_config``; both shapes are accepted here.
    """

    model_type = "hy_v4_vl"
    architectures: list[str] | None = None
    sub_configs = {"text_config": HYV4Config, "vision_config": HYV4VLVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        text_config: Any = None,
        vision_config: Any = None,
        image_start_token_id: int = 120118,
        image_end_token_id: int = 120119,
        image_token_id: int = 120120,
        image_newline_token_id: int = 120121,
        video_start_token_id: int = 120122,
        video_end_token_id: int = 120123,
        video_token_id: int = 120124,
        **kwargs: Any,
    ) -> None:
        metadata_keys = {"model_type", "architectures", "auto_map", "processor_class"}
        outer_kwargs = {key: value for key, value in kwargs.items() if key in metadata_keys}
        if isinstance(text_config, HYV4Config):
            text = text_config
        else:
            # Nested text fields take precedence; flat fields supply missing values.
            text_kwargs = {**kwargs, **(text_config or {})}
            for key in metadata_keys:
                text_kwargs.pop(key, None)
            legacy_rope = text_kwargs.pop("rope_scaling", None)
            rope = dict(legacy_rope or {})
            rope.update(text_kwargs.get("rope_parameters") or {})
            rope.setdefault("rope_type", "default")
            rope.setdefault("mrope_interleaved", True)
            rope.setdefault("mrope_section", [12, 10, 10])
            text_kwargs["rope_parameters"] = rope
            text = HYV4Config(**text_kwargs)
        super().__init__(**outer_kwargs)
        self.text_config = text
        self.vision_config = (
            vision_config
            if isinstance(vision_config, HYV4VLVisionConfig)
            else HYV4VLVisionConfig(**(vision_config or {}))
        )
        if self.vision_config.out_hidden_size is None:
            self.vision_config.out_hidden_size = text.hidden_size
        if self.vision_config.out_hidden_size != text.hidden_size:
            raise ValueError("HYV4 VL projector output must match text hidden_size")
        self.vision_config.text_hidden_size = text.hidden_size
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.image_newline_token_id = image_newline_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id
        for name in (
            "dtype",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ):
            if hasattr(text, name):
                setattr(self, name, getattr(text, name))

    def __getattr__(self, name: str) -> Any:
        """Forward text-side fields that the outer VL config does not define.

        vLLM reads several text fields off ``model_config.hf_config`` directly
        rather than off ``hf_text_config`` — the MLA sparse KV builder takes
        ``hf_config.index_topk``, for example. Those fields live on
        ``text_config`` here, so resolve them through it instead of failing
        with ``AttributeError`` after the KV cache has been sized.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        text = self.__dict__.get("text_config")
        if text is not None and hasattr(text, name):
            return getattr(text, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


__all__ = ["HYV4VLConfig", "HYV4VLVisionConfig"]

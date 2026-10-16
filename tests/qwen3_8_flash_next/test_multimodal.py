"""Core Qwen3.8 framework contracts; numerical tests live in FlagGems-vllm."""

from __future__ import annotations

import types

import torch

from vllm_fl.models.qwen3_8_flash_next.config import (  # noqa: E402
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
    Qwen3_8FlashNextVisionConfig,
)
from vllm_fl.models.qwen3_8_flash_next.gpu.model import (  # noqa: E402
    Qwen3_8FlashNextForConditionalGeneration,
)


def test_composite_config_keeps_typed_vision_and_text_subconfigs():
    config = _build_config()

    assert config.model_type == "qwen4_exp"
    assert isinstance(config.vision_config, Qwen3_8FlashNextVisionConfig)
    assert isinstance(config.text_config, Qwen3_8FlashNextTextConfig)
    assert config.image_token_id == _IMAGE_TOKEN_ID
    assert config.video_token_id == _VIDEO_TOKEN_ID
    assert config.vision_start_token_id == _VISION_START_TOKEN_ID
    assert config.vision_end_token_id == _VISION_END_TOKEN_ID
    assert config.vision_config.spatial_merge_size == 2
    assert config.text_config.hidden_size == 128


def test_embed_input_ids_merges_image_embeddings_at_token_positions():
    from torch import nn

    model = object.__new__(Qwen3_8FlashNextForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model_only = False
    model.use_deepstack = False

    table = torch.randn(512, 8)
    model.language_model = types.SimpleNamespace(
        embed_input_ids=lambda input_ids: torch.nn.functional.embedding(
            input_ids, table
        )
    )
    model._embed_text_input_ids = lambda input_ids, embed_fn, *, is_multimodal=None: (
        embed_fn(input_ids)
    )

    input_ids = torch.tensor([1, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, 3])
    multimodal = torch.randn(2, 8)
    is_multimodal = torch.tensor([False, True, True, False])

    merged = model.embed_input_ids(input_ids, [multimodal], is_multimodal=is_multimodal)

    torch.testing.assert_close(merged[1:3], multimodal)
    torch.testing.assert_close(merged[[0, 3]], table[[1, 3]])


def test_hf_to_vllm_mapper_routes_visual_and_language_model():
    mapper = Qwen3_8FlashNextForConditionalGeneration.hf_to_vllm_mapper
    prefixes = mapper.orig_to_new_prefix

    assert prefixes["model.visual."] == "visual."
    assert prefixes["model.language_model."] == "language_model.model."
    assert prefixes["lm_head."] == "language_model.lm_head."


_TINY_VISION = {
    "deepstack_visual_indexes": [],
    "depth": 1,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 64,
    "in_channels": 3,
    "intermediate_size": 128,
    "model_type": "qwen4_exp",
    "num_heads": 4,
    "num_position_embeddings": 64,
    "out_hidden_size": 128,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}


_TINY_TEXT = {
    "attention_bias": False,
    "head_dim": 32,
    "hc_count": 4,
    "hc_lowrank": 16,
    "hidden_size": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 32,
    "indexer_kv_heads": 1,
    "indexer_n_heads": 2,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 32,
    "model_type": "qwen4_exp_text",
    "moe_intermediate_size": 32,
    "num_attention_heads": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "partial_rotary_factor": 0.25,
    "ple_conv_kernel_size": 4,
    "ple_embed_dim": 128,
    "ple_layer_ids": [2],
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000,
        "rope_type": "default",
    },
    "rope_theta": 1000000,
    "shared_expert_intermediate_size": 32,
    "vocab_size": 256,
}


_IMAGE_TOKEN_ID = 248


_VIDEO_TOKEN_ID = 249


_VISION_START_TOKEN_ID = 250


_VISION_END_TOKEN_ID = 251


def _build_config() -> Qwen3_8FlashNextConfig:
    return Qwen3_8FlashNextConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        model_type="qwen4_exp",
        text_config=dict(_TINY_TEXT),
        vision_config=dict(_TINY_VISION),
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        vision_end_token_id=_VISION_END_TOKEN_ID,
    )

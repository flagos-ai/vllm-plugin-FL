"""Genuine image+text coverage for the Qwen3.8-Flash-Next multimodal path.

The Day0 smoke checkpoint is served with ``language_model_only`` and the
existing suite only exercises ``*ForCausalLM``.  These tests build the
*composite* ``*ForConditionalGeneration`` contract from the real checkpoint
shapes so the visual tower, processor/config registration, token handling,
multimodal embedding merge and M-RoPE positions are covered instead of the
text-only fast path.

The vision/text shapes mirror
``/root/qwen4/day0_delivery/smoke/tiny-qwen4-exp/config.json`` (architecture
``Qwen4ExpForConditionalGeneration``, ``qwen4_exp`` model type).
"""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("vllm")
from vllm.model_executor.models import registry as model_registry  # noqa: E402

from vllm_fl.models.qwen3_8_flash_next.config import (  # noqa: E402
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
    Qwen3_8FlashNextVisionConfig,
)
from vllm_fl.models.qwen3_8_flash_next.gpu.model import (  # noqa: E402
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextForConditionalGeneration,
)

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


class _Grid:
    def __init__(self, values):
        self.data = torch.tensor(values)


class _FakeFeature:
    def __init__(self, offset, length, grid_thw):
        self.modality = "image"
        self.mm_position = types.SimpleNamespace(offset=offset, length=length)
        self.data = {"image_grid_thw": _Grid(grid_thw)}


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


def test_serving_entry_registers_multimodal_architecture():
    from vllm_fl.patches.qwen3_8_flash_next import (
        apply_qwen3_8_flash_next_patches,
    )

    apply_qwen3_8_flash_next_patches()

    for architecture in (
        "Qwen4ExpForConditionalGeneration",
        "Qwen3_8FlashNextForConditionalGeneration",
    ):
        module, class_name = model_registry._MULTIMODAL_MODELS[architecture]
        assert module == "vllm_fl.models.qwen3_8_flash_next"
        resolved = getattr(
            __import__(module, fromlist=[class_name]), class_name
        )
        assert resolved is Qwen3_8FlashNextForConditionalGeneration
        assert architecture not in model_registry._TEXT_GENERATION_MODELS
    for architecture in ("Qwen4ExpForCausalLM", "Qwen3_8FlashNextForCausalLM"):
        assert model_registry._TEXT_GENERATION_MODELS[architecture] == (
            "vllm_fl.models.qwen3_8_flash_next",
            architecture,
        )
        assert architecture not in model_registry._MULTIMODAL_MODELS


def test_multimodal_class_exposes_supported_protocol_surface():
    assert Qwen3_8FlashNextForConditionalGeneration.requires_raw_input_tokens is True
    assert Qwen3_8FlashNextForConditionalGeneration.supports_multimodal_pruning is False
    assert hasattr(Qwen3_8FlashNextForConditionalGeneration, "embed_multimodal")
    assert hasattr(
        Qwen3_8FlashNextForConditionalGeneration, "get_mrope_input_positions"
    )
    assert (
        Qwen3_8FlashNextForConditionalGeneration.get_placeholder_str("image", 0)
        == "<|vision_start|><|image_pad|><|vision_end|>"
    )
    assert (
        Qwen3_8FlashNextForConditionalGeneration.get_placeholder_str("video", 0)
        == "<|vision_start|><|video_pad|><|vision_end|>"
    )


def test_multimodal_processor_registered_for_class():
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLDummyInputsBuilder,
        Qwen3VLMultiModalProcessor,
    )

    from vllm_fl.models.qwen3_8_flash_next.gpu.model import (
        Qwen3_8FlashNextProcessingInfo,
    )

    factories = Qwen3_8FlashNextForConditionalGeneration._processor_factory
    assert factories.info is Qwen3_8FlashNextProcessingInfo
    assert factories.processor is Qwen3VLMultiModalProcessor
    assert factories.dummy_inputs is Qwen3VLDummyInputsBuilder


def test_image_text_mrope_positions_use_vision_grid():
    """Image tokens must receive 3-axis grid positions, not text arange."""

    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    config = _build_config()
    # vision_start | 4 image tokens (2x2 after merge) | vision_end | 3 text
    input_tokens = [
        _VISION_START_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _VISION_END_TOKEN_ID,
        10,
        11,
        12,
    ]
    feature = _FakeFeature(offset=1, length=4, grid_thw=[1, 4, 4])

    positions, delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens, [feature], config
    )

    assert positions.shape == (3, len(input_tokens))
    text_only = torch.arange(len(input_tokens)).unsqueeze(0).expand(3, -1)
    assert not torch.equal(positions, text_only)

    image_span = positions[:, 1:5]
    assert image_span[0].unique().numel() == 1
    assert image_span[1].unique().numel() == 2
    assert image_span[2].unique().numel() == 2
    assert delta == int(positions.max()) + 1 - len(input_tokens)


def test_text_language_model_ignores_mrope_features():
    """Witness the prior false pass: the text-only model discards mm_features."""

    tokens = [_VISION_START_TOKEN_ID, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, 7]
    feature = _FakeFeature(offset=1, length=2, grid_thw=[1, 2, 2])

    positions, delta = Qwen3_8FlashNextForCausalLM.get_mrope_input_positions(
        object(), tokens, [feature]
    )

    assert delta == 0
    torch.testing.assert_close(
        positions, torch.arange(len(tokens)).unsqueeze(0).expand(3, -1)
    )


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
    model._embed_text_input_ids = (
        lambda input_ids, embed_fn, *, is_multimodal=None: embed_fn(input_ids)
    )

    input_ids = torch.tensor([1, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, 3])
    multimodal = torch.randn(2, 8)
    is_multimodal = torch.tensor([False, True, True, False])

    merged = model.embed_input_ids(
        input_ids, [multimodal], is_multimodal=is_multimodal
    )

    torch.testing.assert_close(merged[1:3], multimodal)
    torch.testing.assert_close(merged[[0, 3]], table[[1, 3]])


def test_hf_to_vllm_mapper_routes_visual_and_language_model():
    mapper = Qwen3_8FlashNextForConditionalGeneration.hf_to_vllm_mapper
    prefixes = mapper.orig_to_new_prefix

    assert prefixes["model.visual."] == "visual."
    assert prefixes["model.language_model."] == "language_model.model."
    assert prefixes["lm_head."] == "language_model.lm_head."

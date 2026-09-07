# SPDX-License-Identifier: Apache-2.0

from vllm_fl.configs.glm5_next import Glm5NextTextConfig
from vllm_fl.patches.glm5_next_v024 import (
    Glm5NextModelArchConfigConvertor,
)
from vllm_fl.transformers_utils.processors.glm5_next import (
    _LEGACY_MM_MAX_PIXELS,
    _MAX_VIDEO_TOKENS,
    Glm5NextProcessor,
    _normalize_processor_config,
    glm_sample_frame_indices,
    glm_sample_frame_indices_legacy,
)


class _TokenizerStub:
    init_kwargs = {}

    def __init__(self) -> None:
        self.last_text = None

    def __call__(self, text, **kwargs):
        self.last_text = text
        return {"input_ids": [[11, 12]]}


class _ImageProcessorStub:
    def __call__(self, images, **kwargs):
        return {"pixel_values": "pixels", "image_grid_thw": "grid"}


class _ProcessorCallHarness:
    def __init__(self) -> None:
        self.tokenizer = _TokenizerStub()
        self.image_processor = _ImageProcessorStub()

    def _merge_kwargs(self, *args, **kwargs):
        return {
            "text_kwargs": {
                "return_mm_token_type_ids": False,
            },
            "images_kwargs": {},
            "videos_kwargs": {},
        }


def test_legacy_size_config_is_normalized_without_changing_geometry_mode() -> None:
    config = _normalize_processor_config(
        {
            "size": {
                "shortest_edge": 112 * 112,
                "longest_edge": 100_000_000,
            },
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "patch_expand_factor": 2,
        },
        default_min_tokens=16,
        default_max_tokens=8000,
        is_video=False,
    )

    pixels_per_token = 2 * (14 * 2) ** 2
    assert config["min_image_tokens"] == 8
    assert config["max_image_tokens"] == _LEGACY_MM_MAX_PIXELS // pixels_per_token
    assert config["patch_expand_factor"] == 2
    assert config["resize_mode"] == "resize"
    assert config["sampling_policy"] == "legacy_dynamic"
    assert "size" not in config


def test_token_budget_image_config_keeps_checkpoint_budget() -> None:
    config = _normalize_processor_config(
        {
            "min_image_tokens": 16,
            "max_image_tokens": 8000,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "patch_expand_factor": 1,
            "resize_mode": "pad",
        },
        default_min_tokens=16,
        default_max_tokens=8000,
        is_video=False,
    )

    assert config["min_image_tokens"] == 16
    assert config["max_image_tokens"] == 8000
    assert config["patch_expand_factor"] == 1
    assert config["resize_mode"] == "pad"
    assert config["sampling_policy"] == "fps_interval"


def test_token_budget_video_config_uses_reference_serving_cap() -> None:
    config = _normalize_processor_config(
        {
            "min_image_tokens": 16,
            "max_image_tokens": 240000,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
        },
        default_min_tokens=16,
        default_max_tokens=240000,
        is_video=True,
    )

    assert config["min_image_tokens"] == 16
    assert config["max_image_tokens"] == _MAX_VIDEO_TOKENS


def test_deepseek_sparse_attention_layer_alias() -> None:
    layer_types = ["linear_attention", "deepseek_sparse_attention"]
    config = Glm5NextTextConfig(
        num_hidden_layers=len(layer_types), layer_types=layer_types
    )

    assert config.layer_types == layer_types
    assert config.layers_block_type == ["linear_attention", "attention"]


def test_explicit_zero_head_dim_keeps_nope_mla_layout() -> None:
    config = Glm5NextTextConfig(
        head_dim=0,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
    )
    convertor = Glm5NextModelArchConfigConvertor(config, config)

    assert config.head_dim == 0
    assert convertor.is_deepseek_mla()
    assert convertor.get_head_size() == 512


def test_feature_processor_leaves_prompt_placeholders_unchanged() -> None:
    processor = _ProcessorCallHarness()
    prompt = "before <|image|> after"

    output = Glm5NextProcessor.__call__(processor, images=object(), text=prompt)

    assert processor.tokenizer.last_text == [prompt]
    assert output["image_grid_thw"] == "grid"


def test_video_sampling_remains_config_selectable() -> None:
    new_indices = glm_sample_frame_indices(
        300,
        30.0,
        10.0,
        target_fps=2.0,
        max_frame_count=2048,
        temporal_patch_size=2,
    )
    legacy_indices = glm_sample_frame_indices_legacy(
        300,
        30.0,
        10.0,
        target_fps=3.0,
        max_frame_count=640,
        temporal_patch_size=2,
    )

    assert len(new_indices) == 20
    assert len(legacy_indices) == 60
    assert new_indices[-1] == 299
    assert legacy_indices[-1] == 299

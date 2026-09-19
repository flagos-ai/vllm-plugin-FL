# SPDX-License-Identifier: Apache-2.0
"""Exercise real HF preprocessing against vLLM's deployment reservation."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm_fl.models.glm5_next_multimodal import Glm5NextProcessingInfo
from vllm_fl.transformers_utils.processors.glm5_next import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
)

UNIT = 1568


def make_processor(deployment):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "<|image|>": 1, "<|video|>": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    processor = Glm5NextProcessor(
        Glm5NextImageProcessor(),
        tokenizer,
        Glm5NextVideoProcessor(max_image_tokens=16384),
    )
    processor.configure_serving(deployment)
    info = object.__new__(Glm5NextProcessingInfo)
    config = SimpleNamespace(
        vision_config=SimpleNamespace(
            patch_size=14, spatial_merge_size=2, temporal_patch_size=2
        ),
        video_start_token_id=3,
        video_end_token_id=4,
        image_start_token_id=5,
        image_end_token_id=6,
    )
    info.ctx = SimpleNamespace(
        get_merged_mm_kwargs=lambda kwargs: {**deployment, **kwargs},
        get_hf_config=lambda *args: config,
        get_tokenizer=lambda: tokenizer,
    )
    info._glm5_hf_processor = processor
    return processor, info


@pytest.mark.parametrize(
    "deployment,overrides,expected",
    [
        ({}, {}, 7921),
        ({"max_image_tokens": 16000}, {}, 15876),
        ({"max_image_tokens": 512}, {}, 484),
        ({"max_image_tokens": 16000}, {"max_image_tokens": 512}, 484),
        ({"max_image_tokens": 16000, "max_pixels": 512 * UNIT}, {}, 484),
    ],
)
def test_4096_image_fits_advertised_budget(deployment, overrides, expected):
    processor, info = make_processor(deployment)
    out = processor(
        images=Image.new("RGB", (4096, 4096)),
        text="<|image|>",
        return_tensors="pt",
        **overrides,
    )
    actual = int(out["image_grid_thw"][0].prod()) // 4
    assert actual == expected
    assert actual <= info.get_mm_max_tokens_per_item(34816, {"image": 1})["image"]
    assert out["pixel_values"].shape[0] == actual * 4


@pytest.mark.parametrize(
    "height,width", [(1, 1), (112, 11200), (11200, 112), (600, 900), (900, 600)]
)
def test_aspect_ratios_and_dummy_canvas(height, width):
    processor, info = make_processor({"max_image_tokens": 512})
    out = processor(
        images=Image.new("RGB", (width, height)), text="<|image|>", return_tensors="pt"
    )
    assert int(out["image_grid_thw"].prod()) // 4 <= info.get_max_image_tokens()
    assert int(out["image_grid_thw"].prod()) // 4 == info.get_num_image_tokens(
        image_width=width, image_height=height
    )
    size = info.get_image_size_with_most_features()
    dummy = processor(
        images=Image.new("RGB", size), text="<|image|>", return_tensors="pt"
    )
    # A square would reserve only 484. The maximal aligned canvas reaches 512.
    assert (
        int(dummy["image_grid_thw"].prod()) // 4 == info.get_max_image_tokens() == 512
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_image_tokens": 16000},
        {"max_pixels": 16000 * UNIT},
        {"images_kwargs": {"max_image_tokens": 16000}},
        {"do_resize": False},
        {"merge_size": 4},
        {"size": {"height": 4096}},
    ],
)
def test_unreserved_or_unsupported_request_rejected(overrides):
    processor, info = make_processor({})
    with pytest.raises(ValueError):
        info.get_hf_processor(**overrides)
    with pytest.raises(ValueError):
        processor(images=Image.new("RGB", (28, 28)), text="<|image|>", **overrides)


def test_nested_modality_budget_and_request_do_not_mutate_deployment():
    deployment = {
        "images_kwargs": {"max_image_tokens": 512},
        "videos_kwargs": {"max_image_tokens": 1024, "max_frames": 8},
    }
    processor, info = make_processor(deployment)
    effective = processor.resolve_serving_kwargs(
        {"images_kwargs": {"max_image_tokens": 128}}
    )
    assert effective["images_kwargs"]["max_pixels"] == 128 * UNIT
    assert effective["videos_kwargs"]["max_pixels"] == 1024 * UNIT
    assert info.get_max_image_tokens() == 512
    assert deployment["images_kwargs"]["max_image_tokens"] == 512
    with pytest.raises(ValueError, match="max_frames"):
        processor.resolve_serving_kwargs({"videos_kwargs": {"max_frames": 16}})


@pytest.mark.parametrize("frame_cap", [4, 8])
@pytest.mark.parametrize("fps_kwargs", [{}, {"fps": 0.1}, {"target_fps": 0.1}])
def test_video_override_grid_matches_prompt_and_reservation(frame_cap, fps_kwargs):
    processor, info = make_processor({"max_image_tokens": 512, "max_frames": 8})
    video = np.zeros((60, 112, 112, 3), dtype=np.uint8)
    metadata = dict(total_num_frames=60, fps=2.0, duration=30.0)
    overrides = {"max_frames": frame_cap, "max_image_tokens": 128, **fps_kwargs}
    out = processor(
        videos=[video],
        video_metadata=[metadata],
        text="<|video|>",
        return_tensors="pt",
        **overrides,
    )
    grid = out["video_grid_thw"][0]
    assert int(grid[0]) <= frame_cap // 2
    assert int(grid.prod()) // 4 <= 128
    prompt = info._construct_glm5_video_placeholder(video, metadata, grid, overrides)
    assert prompt.count(processor.image_token_id) == int(grid.prod()) // 4
    assert len(prompt) <= info.get_mm_max_tokens_per_item(34816, {"video": 1})["video"]


def test_presampled_video_cannot_bypass_frame_reservation():
    processor, _ = make_processor({"max_frames": 4})
    with pytest.raises(ValueError, match="max_frames"):
        processor(
            videos=[np.zeros((8, 28, 28, 3), dtype=np.uint8)],
            text="<|video|>",
            do_sample_frames=False,
        )


@pytest.mark.parametrize("frames", [3, 5, 7])
def test_odd_video_frames_reserve_temporal_padding(frames):
    processor, info = make_processor({"max_image_tokens": 128, "max_frames": 8})
    out = processor(
        videos=[np.zeros((frames, 300, 300, 3), dtype=np.uint8)],
        text="<|video|>",
        do_sample_frames=False,
        return_tensors="pt",
    )
    grid = out["video_grid_thw"][0]
    assert int(grid[0]) == (frames + 1) // 2
    assert int(grid.prod()) // 4 <= 128


@pytest.mark.parametrize("cap", [512, 8000, 16000])
def test_profiling_dummy_uses_modality_budget_without_huge_source_canvas(cap):
    from vllm_fl.models.glm5_next_multimodal import Glm5NextDummyInputsBuilder

    processor, info = make_processor(
        {
            "images_kwargs": {"max_image_tokens": cap},
            "videos_kwargs": {"max_frames": 32},
        }
    )
    builder = Glm5NextDummyInputsBuilder(info)
    data = builder.get_dummy_mm_data(34816, {"image": 1, "video": 1}, {})
    image = processor(images=data["image"], text="<|image|>", return_tensors="pt")
    assert int(image["image_grid_thw"].prod()) // 4 == cap
    video, metadata = data["video"][0]
    out = processor(
        videos=[video], text="<|video|>", do_sample_frames=False, return_tensors="pt"
    )
    assert int(out["video_grid_thw"].prod()) // 4 == 16384
    assert video.nbytes <= 16384 * UNIT * 3
    assert int(out["video_grid_thw"][0, 0]) == info._video_frame_cap() // 2


def test_video_prompt_update_closure_keeps_request_options_separate():
    from vllm_fl.models.glm5_next_multimodal import Glm5NextMultiModalProcessor

    processor, info = make_processor({"max_image_tokens": 512, "max_frames": 8})
    mm = object.__new__(Glm5NextMultiModalProcessor)
    mm.info = info
    video = np.zeros((60, 112, 112, 3), dtype=np.uint8)
    metadata = dict(total_num_frames=60, fps=2.0, duration=30.0)
    closures = []
    for frame_cap in (4, 8):
        options = {"max_frames": frame_cap}
        out = processor(
            videos=[video],
            video_metadata=[metadata],
            text="<|video|>",
            return_tensors="pt",
            **options,
        )
        grid = out["video_grid_thw"][0]
        updates = mm._get_prompt_updates(
            {"video": [(video, metadata)]},
            options,
            {"video": [{"video_grid_thw": SimpleNamespace(data=grid)}]},
        )
        update = next(u for u in updates if u.modality == "video")
        closures.append((update.replacement, int(grid.prod()) // 4))
    for replacement, count in closures:
        assert replacement(0).full.count(processor.image_token_id) == count


def test_checkpoint_sampling_rate_survives_hf_default_fps():
    processor, info = make_processor({"max_frames": 32})
    processor.video_processor.fps_interval = 0.1
    video = np.zeros((60, 28, 28, 3), dtype=np.uint8)
    metadata = dict(total_num_frames=60, fps=2.0, duration=30.0)
    out = processor(
        videos=[video], video_metadata=[metadata], text="<|video|>", return_tensors="pt"
    )
    grid = out["video_grid_thw"][0]
    assert int(grid[0]) == 2
    prompt = info._construct_glm5_video_placeholder(video, metadata, grid, {})
    assert prompt.count(processor.image_token_id) == int(grid.prod()) // 4


@pytest.mark.parametrize(
    "deployment,overrides,expected",
    [
        (
            {"images_kwargs": {"max_image_tokens": 16000}},
            {"max_image_tokens": 128},
            128,
        ),
        (
            {"max_image_tokens": 16000},
            {"images_kwargs": {"max_image_tokens": 128}},
            128,
        ),
    ],
)
def test_actual_vllm_context_merge_keeps_request_ceiling(
    deployment, overrides, expected
):
    import torch

    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.processing import InputProcessingContext

    from vllm_fl.models.glm5_next_multimodal import Glm5NextMultiModalProcessor

    processor, info = make_processor(deployment)
    config = MultiModalConfig(mm_processor_kwargs=deployment)
    info.ctx = InputProcessingContext(
        SimpleNamespace(get_multimodal_config=lambda: config, dtype=torch.float32),
        processor.tokenizer,
    )
    mm = object.__new__(Glm5NextMultiModalProcessor)
    mm.info = info
    output = mm._call_hf_processor(
        "<|image|>", {"images": [Image.new("RGB", (600, 900))]}, overrides, {}
    )
    assert int(output["image_grid_thw"].prod()) // 4 <= expected


@pytest.mark.parametrize("sampling_policy", ["fps_interval", "legacy_dynamic"])
@pytest.mark.parametrize(
    "deployment,overrides,expected_frames",
    [
        ({}, {"target_fps": 0.1, "max_frames": 8}, 4),
        ({}, {"videos_kwargs": {"fps": 0.1, "max_frames": 8}}, 4),
        ({"videos_kwargs": {"max_frames": 8}}, {}, 8),
    ],
)
def test_actual_vllm_presampled_video_honors_request_and_deployment(
    deployment, overrides, expected_frames, sampling_policy
):
    import copy

    import torch

    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.processing import InputProcessingContext

    from vllm_fl.models.glm5_next_multimodal import Glm5NextMultiModalProcessor

    processor, info = make_processor(deployment)
    processor.video_processor.sampling_policy = sampling_policy
    if sampling_policy == "legacy_dynamic" and expected_frames == 4:
        expected_frames = 6
    original_context = info.ctx
    info.ctx = InputProcessingContext(
        SimpleNamespace(
            get_multimodal_config=lambda: MultiModalConfig(
                mm_processor_kwargs=deployment
            ),
            dtype=torch.float32,
        ),
        processor.tokenizer,
    )
    mm = object.__new__(Glm5NextMultiModalProcessor)
    mm.info = info
    # Match vLLM's 32-frame decode of a 30-second, 2-fps video.
    video = np.zeros((32, 112, 112, 3), dtype=np.uint8)
    metadata = dict(
        total_num_frames=60,
        fps=2.0,
        duration=30.0,
        frames_indices=np.linspace(0, 59, 32, dtype=int).tolist(),
        do_sample_frames=False,
    )
    original_metadata = copy.deepcopy(metadata)
    output = mm._call_hf_processor(
        "<|video|>", {"videos": [(video, metadata)]}, overrides, {}
    )
    grid = output["video_grid_thw"][0]
    assert int(grid[0]) == expected_frames // 2
    assert metadata == original_metadata
    info.ctx = original_context
    prompt = info._construct_glm5_video_placeholder(video, metadata, grid, overrides)
    assert prompt.count(processor.image_token_id) == int(grid.prod()) // 4
    if expected_frames == 4:
        assert info._get_video_second_idx_glm46v(metadata, len(video), overrides) == [
            0,
            29,
        ]
    elif expected_frames == 6:
        assert info._get_video_second_idx_glm46v(metadata, len(video), overrides) == [
            0,
            10,
            19,
        ]


@pytest.mark.parametrize("sampling_policy", ["fps_interval", "legacy_dynamic"])
def test_presampled_short_video_remains_nonempty_at_low_requested_fps(sampling_policy):
    from vllm_fl.transformers_utils.processors.glm5_next import (
        glm_select_decoded_frames,
    )

    processor, _ = make_processor({})
    processor.video_processor.sampling_policy = sampling_policy
    metadata = SimpleNamespace(
        total_num_frames=1, fps=2.0, duration=0.4, frames_indices=[0]
    )
    rows, source = glm_select_decoded_frames(
        processor.video_processor, metadata, 1, fps=0.01, target_fps=0.01, max_frames=8
    )
    assert rows == source == [0, 0]


def test_profile_inputs_keep_maximal_frames_despite_low_deployment_fps():
    from vllm_fl.models.glm5_next_multimodal import (
        Glm5NextDummyInputsBuilder,
        Glm5NextMultiModalProcessor,
    )

    processor, info = make_processor({"max_frames": 32, "target_fps": 0.01})
    info.ctx.model_config = SimpleNamespace(
        get_multimodal_config=lambda: SimpleNamespace(enable_mm_embeds=False)
    )
    builder = Glm5NextDummyInputsBuilder(info)
    inputs = builder.get_dummy_processor_inputs(34816, {"video": 1}, {})
    assert inputs.hf_processor_mm_kwargs == {
        "videos_kwargs": {"do_sample_frames": False}
    }
    video, metadata = inputs.mm_data_items["video"][0]
    assert len(video) == 32
    options = processor.resolve_serving_kwargs(inputs.hf_processor_mm_kwargs)
    data, kwargs = Glm5NextMultiModalProcessor._get_direct_path_inputs(
        {"videos": [(video, metadata)]}, options
    )
    output = processor(text="<|video|>", **data, **kwargs, return_tensors="pt")
    grid = output["video_grid_thw"][0]
    assert int(grid[0]) == 16
    assert int(grid.prod()) // 4 == 16384
    prompt = info._construct_glm5_video_placeholder(
        video, metadata, grid, inputs.hf_processor_mm_kwargs
    )
    assert prompt.count(processor.image_token_id) == 16384

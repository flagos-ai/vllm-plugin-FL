# SPDX-License-Identifier: Apache-2.0
"""Multimodal contracts and regression coverage."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm_fl.models.glm5.multimodal import Glm5NextProcessingInfo
from vllm_fl.models.glm5.processor import (
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
    assert info.get_hf_processor(**overrides) is processor
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
    from vllm_fl.models.glm5.multimodal import Glm5NextDummyInputsBuilder

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
    from vllm_fl.models.glm5.multimodal import Glm5NextMultiModalProcessor

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

    from vllm_fl.models.glm5.multimodal import Glm5NextMultiModalProcessor

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

    from vllm_fl.models.glm5.multimodal import Glm5NextMultiModalProcessor

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
    from vllm_fl.models.glm5.processor import (
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
    from vllm_fl.models.glm5.multimodal import (
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


VIDEO_CASES = [
    (900, 30.0, 30.0),
    (3000, 30.0, 100.0),
    (300, 30.0, 10.0),
    (48, 2.0, 24.0),
    (125, 25.0, 5.0),
    (72000, 30.0, 2400.0),
]


@pytest.mark.parametrize(("total_frames", "fps", "duration"), VIDEO_CASES)
def test_video_prompt_timestamps_match_encoder_grid(
    total_frames: int, fps: float, duration: float
) -> None:
    """Frame placeholders must equal the sampler's ``grid_t``."""
    torch = pytest.importorskip("torch")
    del torch
    from vllm_fl.models.glm5.processor import (
        Glm5NextVideoProcessor,
        glm_video_timestamp_seconds,
    )

    video_processor = Glm5NextVideoProcessor()
    metadata = SimpleNamespace(
        fps=fps,
        duration=duration,
        total_num_frames=total_frames,
        do_sample_frames=True,
    )

    sampled = list(video_processor.sample_frames(metadata))
    grid_t = len(sampled) // video_processor.temporal_patch_size
    timestamps = glm_video_timestamp_seconds(video_processor, metadata)

    assert len(timestamps) == grid_t
    assert len(sampled) % video_processor.temporal_patch_size == 0


def test_no_sample_frames_uses_provided_indices() -> None:
    pytest.importorskip("torch")
    from vllm_fl.models.glm5.processor import (
        Glm5NextVideoProcessor,
        glm_video_timestamp_seconds,
    )

    video_processor = Glm5NextVideoProcessor()
    metadata = SimpleNamespace(
        fps=30.0,
        duration=10.0,
        total_num_frames=300,
        do_sample_frames=False,
        frames_indices=[0, 10, 20, 30, 40, 50],
    )
    timestamps = glm_video_timestamp_seconds(video_processor, metadata)
    assert timestamps == [0, 0, 1]


@pytest.mark.parametrize("merger", [False, True])
def test_vision_activation_clamps_large_inputs(monkeypatch, merger):
    import torch

    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_fl.models.glm5 import multimodal as vision

    # Projection weights are irrelevant here; construct the actual GLM modules
    # and exercise the activation they install, including its configured limit.
    monkeypatch.setattr(vision, "is_vit_use_data_parallel", lambda: True)
    for name in (
        "MergedColumnParallelLinear",
        "RowParallelLinear",
        "ColumnParallelLinear",
    ):
        monkeypatch.setattr(vision, name, lambda *a, **kw: torch.nn.Identity())
    cls = vision.Glm5NextPatchMerger if merger else vision.Glm5NextVisionMLP
    with set_current_vllm_config(VllmConfig()):
        layer = cls(4, 4, swiglu_limit=2.0)
    x = torch.tensor([[10.0, -10.0, 1.0, -1.0, 10.0, -10.0, 3.0, -3.0]])
    gate = x[:, :4].clamp(max=2.0)
    up = x[:, 4:].clamp(min=-2.0, max=2.0)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(layer.act_fn.forward_native(x), expected)


@pytest.mark.parametrize("epsilon", [1e-5, 2e-4])
def test_vision_uses_configured_rms_epsilon_without_mutating_checkpoint(
    monkeypatch, epsilon
):
    from contextlib import nullcontext

    import torch

    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_fl.models.glm5 import multimodal as vision, runtime

    config = SimpleNamespace(
        text_config=SimpleNamespace(),
        vision_config=SimpleNamespace(
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            depth=1,
            hidden_size=4,
            num_heads=1,
            spatial_merge_size=2,
            out_hidden_size=4,
            intermediate_size=8,
            projection_intermediate_size=8,
            swiglu_limit=10.0,
            rms_norm_eps=epsilon,
        ),
    )
    deployment = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            multimodal_config=SimpleNamespace(
                mm_encoder_tp_mode="data", is_multimodal_pruning_enabled=lambda: False
            ),
        )
    )
    # Construct the real model/tower/norms without unrelated weights or devices.
    monkeypatch.setattr(runtime, "validate_glm5_config", lambda _: None)
    monkeypatch.setattr(vision, "is_vit_use_data_parallel", lambda: True)
    cls = vision.Glm5NextForConditionalGeneration
    monkeypatch.setattr(cls, "_mark_tower_model", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(cls, "_mark_language_model", lambda *a, **kw: nullcontext())
    for name in (
        "MergedColumnParallelLinear",
        "RowParallelLinear",
        "ColumnParallelLinear",
        "QKVParallelLinear",
        "Conv2dLayer",
        "Conv3dLayer",
        "Glm5VisionAttention",
        "MMEncoderAttention",
        "init_vllm_registered_model",
    ):
        monkeypatch.setattr(vision, name, lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr(vision, "get_vit_attn_backend", lambda **kw: None)
    with set_current_vllm_config(VllmConfig()):
        tower = cls(vllm_config=deployment).visual
    x = torch.tensor([[0.001, -0.002, 0.003, -0.004]])
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    cos, sin = tower.rotary_pos_emb.get_cos_sin(2)
    assert cos.dtype == sin.dtype == torch.float32
    for norm in (tower.blocks[0].norm1, tower.blocks[0].norm2, tower.post_layernorm):
        torch.testing.assert_close(norm.forward_native(x), expected)
    for norm in (tower.blocks[0].attn.q_norm, tower.blocks[0].attn.k_norm):
        torch.testing.assert_close(norm.forward_native(x), expected)
    assert tower.merger.post_projection_norm.eps == 1e-5
    assert config.vision_config.rms_norm_eps == epsilon


def test_vision_block_rounds_attention_residual_before_normalization(monkeypatch):
    import torch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    from vllm_fl.models.glm5.multimodal import Glm5NextVisionBlock

    block = Glm5NextVisionBlock.__new__(Glm5NextVisionBlock)
    torch.nn.Module.__init__(block)
    block.norm1 = torch.nn.Identity()
    with set_current_vllm_config(VllmConfig()):
        block.norm2 = RMSNorm(4, eps=1e-5).to(dtype=torch.bfloat16)
    # Execute the actual CPU norm while isolating the residual ordering.
    monkeypatch.setattr(block.norm2, "_forward_method", block.norm2.forward_native)
    block.mlp = torch.nn.Identity()
    x = torch.tensor([[1.0, 2.0, -1.0, -2.0]], dtype=torch.bfloat16)
    attention = torch.tensor(
        [[0.00921630859375, 0.005340576171875, 0.01068115234375, 0.0162353515625]],
        dtype=torch.bfloat16,
    )

    class FixedAttention(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return attention

    block.attn = FixedAttention()
    residual = x + attention
    normalized = residual.float() * torch.rsqrt(
        residual.float().square().mean(-1, keepdim=True) + 1e-5
    )
    expected = residual + normalized.to(x.dtype)
    unrounded = x.float() + attention.float()
    fused = unrounded * torch.rsqrt(unrounded.square().mean(-1, keepdim=True) + 1e-5)
    assert not torch.equal(expected, residual + fused.to(x.dtype))
    actual = block(x, None, None, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connectivity contracts for the GLM5-Next multimodal path.

Two defects were found while auditing PR #454 against the native vLLM 0.24
GLM5.3-Flash runtime (``vllm_glm5_next``, 20260826) and vLLM 0.24's inherited
GLM-4.6V prompt machinery:

1. Video prompt timestamps must come from the exact frame sampler that the
   vision tower consumes, not vLLM's duration-threshold GLM-4.6V re-derivation.
2. The video encoder token ceiling must come from the token-budget pixel cap,
   because ``video_processor.size`` is intentionally unused by this checkpoint.

The vision MLP/merger clamp is **intentionally kept**: this checkpoint declares
``vision_config.model_type = "glm5_next_vision"`` and the validated native
runtime clamps it, even though the older 0808 references mapped vision to
``glm_ocr_vision`` (unclamped).
"""

from types import SimpleNamespace

import pytest

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
    from vllm_fl.transformers_utils.processors.glm5_next import (
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
    from vllm_fl.transformers_utils.processors.glm5_next import (
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

    from vllm_fl.models import glm5_next_multimodal as vision

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

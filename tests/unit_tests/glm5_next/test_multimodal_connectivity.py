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

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
MULTIMODAL = ROOT / "vllm_fl/models/glm5_next_multimodal.py"


def _module() -> ast.Module:
    return ast.parse(MULTIMODAL.read_text())


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _self_attribute_calls(method: ast.FunctionDef, attr: str) -> set[str]:
    return {
        ast.unparse(node.value)
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == attr
    }


def _called_names(node: ast.AST) -> set[str]:
    return {
        ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)
    }


def test_vision_mlp_and_merger_keep_checkpoint_swiglu_clamp() -> None:
    """GLM5.3-Flash declares ``glm5_next_vision`` and clamps the vision SwiGLU.

    This is a regression guard: the checkpoint-specific native runtime clamps,
    so the plugin must not silently drop the clamp.
    """
    module = _module()

    for class_name in ("Glm5NextVisionMLP", "Glm5NextPatchMerger"):
        init = _method(_class(module, class_name), "__init__")
        act_assignments = _self_attribute_calls(init, "act_fn")
        assert act_assignments == {"SiluAndMulWithClamp(swiglu_limit=swiglu_limit)"}, (
            class_name,
            act_assignments,
        )


def test_processing_info_overrides_video_prompt_and_budget() -> None:
    module = _module()
    info = _class(module, "Glm5NextProcessingInfo")
    methods = {node.name for node in info.body if isinstance(node, ast.FunctionDef)}

    assert "get_mm_max_tokens_per_item" in methods
    assert "_get_video_second_idx_glm46v" in methods

    budget = _method(info, "get_mm_max_tokens_per_item")
    budget_calls = _called_names(budget)
    assert "self._get_video_max_pixels" in budget_calls
    # The inherited vLLM 0.24 base reads the unused placeholder size.
    size_subscripts = [
        node
        for node in ast.walk(budget)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "size"
    ]
    assert not size_subscripts

    timestamps = _method(info, "_get_video_second_idx_glm46v")
    timestamp_calls = _called_names(timestamps)
    assert "self.get_video_processor" in timestamp_calls
    assert "glm_video_timestamp_seconds" in timestamp_calls


def _vllm_glm46v_second_idxs(meta_frames, video_fps, duration, temporal_patch_size=2):
    """Replica of vLLM 0.24's ``_get_video_second_idx_glm46v`` frame policy."""
    import numpy as np

    duration = duration or (round((meta_frames - 1) / video_fps) + 1)
    effective_duration = min(duration, 2400)
    if effective_duration <= 30:
        target_fps = 3
    elif effective_duration <= 300:
        target_fps = 1
    else:
        target_fps = 0.5

    extract_t = min(int(effective_duration * target_fps * temporal_patch_size), 640)
    timestamps = [i / video_fps for i in range(meta_frames)]
    max_second = int(duration)
    if meta_frames < extract_t:
        frame_indices = np.linspace(0, meta_frames - 1, extract_t, dtype=int).tolist()
    else:
        frame_indices = []
        current_second = 0.0
        inv_fps = 1 / (temporal_patch_size * target_fps)
        for frame_index in range(meta_frames):
            if timestamps[frame_index] >= current_second:
                current_second += inv_fps
                frame_indices.append(frame_index)
                if current_second >= max_second:
                    break
    if len(frame_indices) < extract_t:
        start, end = (
            (frame_indices[0], frame_indices[-1])
            if frame_indices
            else (0, max(meta_frames - 1, 0))
        )
        frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(frame_indices) > extract_t:
        frame_indices = np.linspace(0, meta_frames - 1, extract_t, dtype=int).tolist()

    seen, uniq = set(), []
    for idx in frame_indices:
        if idx not in seen:
            seen.add(idx)
            uniq.append(int(idx))
    if len(uniq) & 1:
        uniq.append(uniq[-1])
    return [int(idx / video_fps) for idx in uniq][::2]


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


def test_legacy_vllm_frame_policy_would_desync() -> None:
    """Documents the bug the override fixes: vLLM's GLM-4.6V policy differs."""
    pytest.importorskip("torch")
    from vllm_fl.transformers_utils.processors.glm5_next import (
        Glm5NextVideoProcessor,
        glm_video_timestamp_seconds,
    )

    video_processor = Glm5NextVideoProcessor()
    mismatches = 0
    for total_frames, fps, duration in VIDEO_CASES:
        metadata = SimpleNamespace(
            fps=fps,
            duration=duration,
            total_num_frames=total_frames,
            do_sample_frames=True,
        )
        aligned = len(glm_video_timestamp_seconds(video_processor, metadata))
        legacy = len(
            _vllm_glm46v_second_idxs(
                total_frames, fps, duration, video_processor.temporal_patch_size
            )
        )
        if aligned != legacy:
            mismatches += 1
    assert mismatches > 0


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

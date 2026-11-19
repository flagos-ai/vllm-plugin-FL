# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-local Qwen4Exp video routing and metadata validation."""

import inspect
import math
from copy import copy
from numbers import Real

from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, Qwen3VLVideoBackend


def validate_video_pruning(multimodal_config) -> None:
    """Keep vLLM 0.32's supported pruning rates; reject invalid metadata."""
    rate = getattr(multimodal_config, "video_pruning_rate", None)
    if rate is None:
        return
    if not isinstance(rate, Real) or isinstance(rate, bool) or not 0 <= rate < 1:
        raise ValueError("Qwen4Exp video_pruning_rate must be None or in [0, 1).")


def validate_video_fps(fps, *, name: str) -> None:
    """Reject invalid CPU metadata before sampling or timestamp division."""
    valid = isinstance(fps, Real) and not isinstance(fps, bool)
    if valid:
        try:
            valid = math.isfinite(fps) and fps > 0
        except OverflowError:
            valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite positive number, got {fps!r}")


class Qwen4ExpVideoBackend(Qwen3VLVideoBackend):
    """Keep Qwen3.8 FPS validation local to its video backend."""

    @classmethod
    def _prepare_source(cls, source):
        validate_video_fps(source.original_fps, name="source video fps")
        return source

    @classmethod
    def compute_frames_index_to_sample(cls, source, target, **kwargs):
        validate_video_fps(source.original_fps, name="source video fps")
        validate_video_fps(target.fps, name="target video fps")
        return super().compute_frames_index_to_sample(source, target, **kwargs)


def register_video_loader() -> None:
    """Register in each plugin process without remapping shared processors."""
    name = "qwen3_8_flash_next"
    original = getattr(
        Qwen3VLVideoBackend.compute_frames_index_to_sample, "__func__", None
    )
    if not callable(original):
        raise TypeError("Qwen3 video frame selection must be a class method")
    signature = inspect.signature(original)
    signature.bind(Qwen3VLVideoBackend, object(), object(), min_frames=4)
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        raise TypeError("Qwen3 video frame selection must accept keyword options")

    registered = VIDEO_LOADER_REGISTRY.name2class.get(name)
    if registered is not None and registered is not Qwen4ExpVideoBackend:
        raise ValueError(f"Video backend {name!r} is already registered")
    if registered is None:
        VIDEO_LOADER_REGISTRY.register(name)(Qwen4ExpVideoBackend)


def configure_video_loader(model_config) -> None:
    """Select a model-local loader through vLLM's media I/O configuration."""
    multimodal_config = model_config.multimodal_config
    if multimodal_config is None or multimodal_config.language_model_only:
        return
    validate_video_pruning(multimodal_config)
    media_io_kwargs = multimodal_config.media_io_kwargs
    video_kwargs = media_io_kwargs.get("video", {})
    backend = video_kwargs.get("video_backend")
    name = "qwen3_8_flash_next"
    # Retain explicit alternate loaders, including empty/None (env fallback),
    # and vLLM's config/request override precedence. Never remap a shared
    # processor name to a model-specific backend.
    if "video_backend" in video_kwargs and backend not in ("qwen3_vl", name):
        return

    # A config may be shared by another model or passed to another process.
    # Copy only the containers changed here; all other modality settings stay.
    updated_config = copy(multimodal_config)
    updated_config.media_io_kwargs = {
        **media_io_kwargs,
        "video": {**video_kwargs, "video_backend": name},
    }
    registered = VIDEO_LOADER_REGISTRY.name2class.get(name)
    try:
        register_video_loader()
        if backend != name:
            model_config.multimodal_config = updated_config
    except Exception:
        if registered is None:
            VIDEO_LOADER_REGISTRY.name2class.pop(name, None)
        raise

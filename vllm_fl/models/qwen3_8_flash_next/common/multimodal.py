# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input contracts shared by the Qwen3.8 config and video adapters."""

import math
from numbers import Real


def validate_video_pruning(multimodal_config) -> None:
    """EVS is not implemented by this model; never prune only placeholders."""
    rate = getattr(multimodal_config, "video_pruning_rate", None)
    if rate is None:
        return
    if not isinstance(rate, Real) or isinstance(rate, bool) or rate != 0:
        raise ValueError(
            "Qwen3.8-Flash-Next does not support video pruning; "
            "video_pruning_rate must be None or 0."
        )


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

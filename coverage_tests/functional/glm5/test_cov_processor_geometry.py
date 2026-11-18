# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-GPU tests for the glm5 processor's pixel geometry (M1).

``smart_resize`` and the token-budget/pixel-budget translation plus both
released processor-config schemas. Numerical contracts are checked
against scalar arithmetic computed independently in each test.
"""

import math

import pytest

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.processor import (
    _normalize_processor_config,
    _pixel_budget,
    smart_resize,
)

pytestmark = [pytest.mark.gpu]


class TestPixelBudget:
    """Token bounds translate to pixel budgets via the per-token factor."""

    def test_pixel_budget_from_token_bounds(self):
        # factor = temporal * (patch * merge)^2 = 2 * (14 * 2)^2 = 1568
        """Pixel budget from token bounds."""
        assert _pixel_budget(16, 8000, 14, 2, 2) == (16 * 1568, 8000 * 1568)

    def test_pixel_budget_requires_both_bounds(self):
        """Pixel budget requires both bounds."""
        with pytest.raises(ValueError, match="min_image_tokens and max_image_tokens"):
            _pixel_budget(None, 8, 14, 2, 2)
        with pytest.raises(ValueError, match="min_image_tokens and max_image_tokens"):
            _pixel_budget(16, None, 14, 2, 2)


class TestNormalizeProcessorConfig:
    """Both released processor-config schemas normalize to one shape."""

    def test_token_budget_schema_defaults(self):
        """Token budget schema defaults."""
        cfg = _normalize_processor_config(
            {}, default_min_tokens=16, default_max_tokens=8000, is_video=False
        )
        assert cfg["min_image_tokens"] == 16
        assert cfg["max_image_tokens"] == 8000
        assert cfg["patch_expand_factor"] == 1
        assert cfg["resize_mode"] == "pad"
        assert cfg["sampling_policy"] == "fps_interval"

    def test_video_budget_is_capped_for_serving(self):
        """Video budget is capped for serving."""
        cfg = _normalize_processor_config(
            {"max_image_tokens": 240000},
            default_min_tokens=16,
            default_max_tokens=240000,
            is_video=True,
        )
        assert cfg["max_image_tokens"] == 30_000

    def test_legacy_size_schema_keeps_pixel_guard(self):
        """Legacy size schema keeps pixel guard."""
        cfg = _normalize_processor_config(
            {"size": {"shortest_edge": 3136, "longest_edge": 10**9}},
            default_min_tokens=16,
            default_max_tokens=8000,
            is_video=False,
        )
        assert "size" not in cfg
        assert cfg["patch_expand_factor"] == 2
        assert cfg["resize_mode"] == "resize"
        assert cfg["sampling_policy"] == "legacy_dynamic"
        # 1_254_400 pixel ceiling / 1568 pixels per token = 800 tokens.
        assert cfg["max_image_tokens"] == 800
        assert cfg["min_image_tokens"] == math.ceil(3136 / 1568)

    def test_min_above_max_is_clamped_down(self):
        """Min above max is clamped down."""
        cfg = _normalize_processor_config(
            {"min_image_tokens": 500, "max_image_tokens": 200},
            default_min_tokens=16,
            default_max_tokens=8000,
            is_video=False,
        )
        assert cfg["min_image_tokens"] == 200


class TestSmartResize:
    """Upward-aligned canvas under a t * h * w pixel budget."""

    def test_alignment_rounds_up(self):
        """Alignment rounds up."""
        assert smart_resize(2, 100, 100) == (112, 112)

    def test_small_image_is_upscaled_to_min_pixels(self):
        """Small image is upscaled to min pixels."""
        h, w = smart_resize(2, 3, 3, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280)
        assert 2 * h * w >= 56 * 56
        assert h % 28 == 0 and w % 28 == 0

    def test_over_budget_refits_by_binary_search(self):
        """Over budget refits by binary search."""
        h, w = smart_resize(2, 2000, 1000, max_pixels=224 * 224)
        assert 2 * h * w <= 224 * 224
        assert h % 28 == 0 and w % 28 == 0

    def test_min_pixels_upscale_can_refit_within_max(self):
        """The beta-scaled candidate is re-fit when alignment overflows."""
        h, w = smart_resize(
            2, 100, 60, min_pixels=14 * 14 * 4 * 1280, max_pixels=14 * 14 * 4 * 1281
        )
        assert 2 * h * w <= 14 * 14 * 4 * 1281
        assert h % 28 == 0 and w % 28 == 0

    def test_too_small_budget_raises(self):
        # t * h_factor * w_factor = 2 * 28 * 28 = 1568 > budget.
        """Too small budget raises."""
        with pytest.raises(ValueError, match="too small"):
            smart_resize(2, 56, 56, min_pixels=4, max_pixels=1000)

    @pytest.mark.parametrize(
        ("t", "h", "w", "kwargs"),
        [
            (0, 100, 100, {}),
            (2, 0, 100, {}),
            (2, 100, 0, {}),
            (2, 100, 100, dict(min_pixels=0)),
            (2, 100, 100, dict(min_pixels=100, max_pixels=10)),
        ],
    )
    def test_invalid_dimensions_raise(self, t, h, w, kwargs):
        """Invalid dimensions raise."""
        with pytest.raises(ValueError):
            smart_resize(t, h, w, **kwargs)

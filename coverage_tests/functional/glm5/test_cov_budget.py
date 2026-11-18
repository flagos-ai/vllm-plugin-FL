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

"""Single-GPU tests for vllm_fl/models/glm5/budget.py (M1).

The budget module is pure Python over duck-typed processor attributes, so
each test builds small attribute namespaces and checks the resolved
numbers against independently computed references (cell enumeration,
token*unit arithmetic, alias tables) instead of re-calling the same code
path. Raise-path tests pin the exact ValueError messages.
"""

import copy
from types import SimpleNamespace

import pytest

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.budget import (
    NORMALIZED_SERVING_KEYS,
    VisionBudget,
    modality_kwargs,
    resolve_serving_kwargs,
    resolve_vision_budget,
)

pytestmark = [pytest.mark.gpu]

# One vision token covers temporal_patch_size*(patch_size*merge_size)^2
# pixels = 2*(14*2)^2 = 1568.
UNIT = 2 * (14 * 2) ** 2
SPATIAL = 14 * 2 * 1  # patch*merge*patch_expand_factor


def _image_processor(**overrides):
    values = dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        patch_expand_factor=1,
        min_image_tokens=4,
        max_image_tokens=512,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _video_processor(**overrides):
    values = dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        patch_expand_factor=1,
        min_image_tokens=16,
        max_image_tokens=240000,
        max_frame_count_dynamic=2048,
        sampling_policy="fps_interval",
        fps_interval=2.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _processor():
    return SimpleNamespace(
        image_processor=_image_processor(), video_processor=_video_processor()
    )


class TestLargestCanvas:
    """The canvas search maximizes token cells over aligned (h, w) pairs."""

    BUDGET = VisionBudget(6272, 802816, UNIT, 28, 2)

    def _reference(self, budget, num_frames):
        """Enumerate every aligned canvas; return the max-cell (h, w)."""
        frames = num_frames if num_frames is not None else budget.temporal_factor
        frames += -frames % budget.temporal_factor
        cells = budget.max_pixels // (frames * budget.spatial_factor**2)
        best = None
        for height in range(1, cells + 1):
            for width in range(height, min(cells // height, 200 * height) + 1):
                cand = (height * width, height, width)
                if best is None or cand > best:
                    best = cand
        return best[1] * budget.spatial_factor, best[2] * budget.spatial_factor

    @pytest.mark.parametrize("num_frames", [None, 4, 5])
    def test_matches_full_enumeration(self, num_frames):
        """Check the patch budget against every feasible height and width."""
        budget = self.BUDGET
        assert budget.largest_canvas(num_frames) == self._reference(budget, num_frames)

    def test_wide_canvas_beats_square_when_cells_allow(self):
        """(448, 896) carries 16*32 cells; a square would carry 22*22."""
        height, width = self.BUDGET.largest_canvas()
        assert height * width == 448 * 896
        # 512 cells over 2 frames: the widest aligned pair is 16 x 32.
        assert (height // 28, width // 28) == (16, 32)

    def test_num_frames_rounds_up_to_temporal_factor(self):
        """5 frames act as 6 (next multiple of temporal_factor=2)."""
        budget = self.BUDGET
        frames = 5 + (-5 % 2)
        cells = budget.max_pixels // (frames * budget.spatial_factor**2)
        assert cells == 802816 // (6 * 28 * 28)
        assert budget.largest_canvas(5) == budget.largest_canvas(6)

    def test_raises_when_no_frame_fits_one_patch(self):
        """vllm_fl/models/glm5/budget.py:44 rejects unusable budgets."""
        with pytest.raises(ValueError, match="cannot fit one aligned patch"):
            VisionBudget(1568, 1567, UNIT, 28, 2).largest_canvas()
        # 3 frames round to 4; 3135 px cannot fit 4*784.
        with pytest.raises(ValueError, match="cannot fit one aligned patch"):
            VisionBudget(1568, 3135, UNIT, 28, 2).largest_canvas(3)

    def test_exact_fit_yields_single_cell_canvas(self):
        """3136 px with 4 frames is exactly one aligned cell."""
        canvas = VisionBudget(1568, 3136, UNIT, 28, 2).largest_canvas(3)
        assert canvas == (28, 28)


class TestResolveVisionBudget:
    """Attribute values flow into the frozen budget with unit arithmetic."""

    def test_defaults_and_max_tokens(self):
        """Check the default budget and maximum token bound."""
        budget = resolve_vision_budget(_image_processor(), {})
        assert budget.min_pixels == 4 * UNIT
        assert budget.max_pixels == 512 * UNIT
        assert budget.pixels_per_token == UNIT
        assert budget.spatial_factor == SPATIAL
        assert budget.temporal_factor == 2
        assert budget.max_tokens == 512

    def test_pixel_overrides_take_priority_over_tokens(self):
        """Pixel overrides take priority over tokens."""
        budget = resolve_vision_budget(
            _image_processor(), {"min_pixels": 3136, "max_pixels": 100000}
        )
        assert (budget.min_pixels, budget.max_pixels) == (3136, 100000)
        assert budget.max_tokens == 100000 // UNIT

    def test_rejects_size_and_geometry_overrides(self):
        """vllm_fl/models/glm5/budget.py:111 guards the serving contract."""
        processor = _image_processor()
        with pytest.raises(ValueError, match="not size"):
            resolve_vision_budget(processor, {"size": {"longest_edge": 9}})
        for name, value in (
            ("patch_size", 28),
            ("merge_size", 4),
            ("temporal_patch_size", 1),
            ("patch_expand_factor", 2),
        ):
            with pytest.raises(ValueError, match="overriding " + name):
                resolve_vision_budget(processor, {name: value})

    def test_rejects_do_resize_false(self):
        """Rejects do resize false."""
        with pytest.raises(ValueError, match="do_resize=True"):
            resolve_vision_budget(_image_processor(), {"do_resize": False})

    @pytest.mark.parametrize(
        "value",
        [0, -5, 12.5, True],
    )
    def test_rejects_non_positive_token_bounds(self, value):
        """Rejects nonpositive token bounds."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            resolve_vision_budget(_image_processor(), {"max_image_tokens": value})

    def test_rejects_inverted_or_sub_patch_budget(self):
        """Rejects inverted or sub patch budget."""
        with pytest.raises(ValueError, match="empty or smaller than one aligned patch"):
            resolve_vision_budget(
                _image_processor(),
                {"min_image_tokens": 4, "max_image_tokens": 2},
            )
        # max below one aligned patch (2*28*28 pixels).
        with pytest.raises(ValueError, match="empty or smaller than one aligned patch"):
            resolve_vision_budget(
                _image_processor(),
                {"min_pixels": UNIT, "max_pixels": UNIT - 1},
            )


class TestModalityKwargs:
    """Nested modality dictionaries win over flat keys."""

    def test_nested_video_kwargs_shadow_flat(self):
        """Nested video keyword arguments shadow flat."""
        merged = modality_kwargs(
            {"fps": 1, "videos_kwargs": {"fps": 2}, "text_kwargs": {"a": 1}},
            "video",
        )
        assert merged == {"fps": 2}

    def test_nested_image_kwargs_shadow_flat(self):
        """Nested image keyword arguments shadow flat."""
        merged = modality_kwargs(
            {"fps": 1, "images_kwargs": {"fps": 3}, "common_kwargs": {"b": 2}},
            "image",
        )
        assert merged == {"fps": 3}


class TestResolveServingKwargs:
    """Deployment and request kwargs merge into normalized scoped dicts."""

    def test_empty_request_carries_deployment_budget(self):
        """Empty request carries deployment budget."""
        result = resolve_serving_kwargs(_processor(), {"max_image_tokens": 512}, {})
        assert set(result) == {"images_kwargs", "videos_kwargs"}
        images = result["images_kwargs"]
        assert images == {
            "max_image_tokens": 512,
            "min_pixels": 4 * UNIT,
            "max_pixels": 512 * UNIT,
        }
        videos = result["videos_kwargs"]
        # Videos use their own processor defaults clamped by the
        # deployment max_image_tokens=512.
        assert videos["max_image_tokens"] == 512
        assert videos["min_pixels"] == 16 * UNIT
        assert videos["max_pixels"] == 512 * UNIT
        assert videos["max_frames"] == 1024
        assert videos["fps"] == videos["target_fps"] == 2.0

    def test_flat_normalized_keys_are_rescoped_into_modality_dicts(self):
        """NORMALIZED_SERVING_KEYS leave the top level for nested scopes."""
        result = resolve_serving_kwargs(
            _processor(),
            {"max_pixels": 200000, "do_normalize": False},
            {},
        )
        assert "max_pixels" not in result
        assert result["do_normalize"] is False
        assert result["images_kwargs"]["max_pixels"] == 200000

    def test_request_narrows_deployment_budget(self):
        """Request narrows deployment budget."""
        result = resolve_serving_kwargs(
            _processor(),
            {"max_image_tokens": 16000},
            {"images_kwargs": {"max_image_tokens": 512}},
        )
        assert result["images_kwargs"] == {
            "max_image_tokens": 512,
            "min_pixels": 4 * UNIT,
            "max_pixels": 512 * UNIT,
        }

    def test_max_frames_clamped_by_pixel_frame_cap(self):
        # 32 tokens = 50176 px -> 64 aligned frames cap.
        """Max frames clamped by pixel frame cap."""
        result = resolve_serving_kwargs(
            _processor(),
            {},
            {"videos_kwargs": {"max_image_tokens": 32, "max_frames": 64}},
        )
        assert result["videos_kwargs"]["max_frames"] == 64
        result = resolve_serving_kwargs(
            _processor(),
            {},
            {"videos_kwargs": {"max_image_tokens": 32, "max_frames": 128}},
        )
        assert result["videos_kwargs"]["max_frames"] == 64

    def test_max_frames_exceeding_capabilities_rejected(self):
        """vllm_fl/models/glm5/budget.py:190 guards frame-count bounds."""
        cases = [
            {"max_frames": 3},
            {"max_frames": 5},  # not a multiple of temporal_patch_size=2
            {"max_frames": -2},
            {"max_frames": 4096},
            {"max_frames": True},
            {"max_frames": "16"},
        ]
        for case in cases:
            with pytest.raises(ValueError, match="max_frames exceeds deployment limit"):
                resolve_serving_kwargs(_processor(), {}, {"videos_kwargs": case})

    @pytest.mark.parametrize(
        ("fps_kwarg", "fps_value"),
        [
            ("fps", 0),
            ("fps", -1.5),
            ("fps", float("inf")),
            ("fps", True),
            ("fps", "fast"),
            ("target_fps", float("nan")),
        ],
    )
    def test_invalid_fps_rejected(self, fps_kwarg, fps_value):
        """Invalid FPS rejected."""
        with pytest.raises(ValueError, match="fps must be finite and positive"):
            resolve_serving_kwargs(
                _processor(), {}, {"videos_kwargs": {fps_kwarg: fps_value}}
            )

    @pytest.mark.parametrize(
        ("aliases", "expected"),
        [
            ({"target_fps": 4.0, "fps_interval": 6.0}, 4.0),
            ({"fps_interval": 6.0}, 6.0),
            ({"fps": 5, "target_fps": 9}, 5),
            ({"fps": 5}, 5),
        ],
    )
    def test_fps_alias_resolution(self, aliases, expected):
        """Resolve legacy and current video sampling rate fields."""
        result = resolve_serving_kwargs(_processor(), {}, {"videos_kwargs": aliases})
        videos = result["videos_kwargs"]
        assert videos["fps"] == expected
        assert videos["target_fps"] == expected

    def test_sampling_policy_override_rejected(self):
        """vllm_fl/models/glm5/budget.py:168 pins the sampling policy."""
        with pytest.raises(ValueError, match="cannot override sampling_policy"):
            resolve_serving_kwargs(
                _processor(),
                {},
                {"videos_kwargs": {"sampling_policy": "legacy_dynamic"}},
            )

    def test_request_budget_above_deployment_rejected(self):
        """Request budget above deployment rejected."""
        for nested in ("images_kwargs", "videos_kwargs"):
            with pytest.raises(ValueError, match="exceeds deployment limit"):
                resolve_serving_kwargs(
                    _processor(),
                    {
                        nested: {
                            "min_image_tokens": 1,
                            "max_image_tokens": 4,
                        }
                    },
                    {nested: {"max_image_tokens": 512}},
                )

    def test_inputs_are_not_mutated(self):
        """Inputs are not mutated."""
        deployment = {
            "max_image_tokens": 512,
            "videos_kwargs": {"max_frames": 16},
        }
        request = {"videos_kwargs": {"max_frames": 8}}
        deploy_snap = copy.deepcopy(deployment)
        request_snap = copy.deepcopy(request)
        resolve_serving_kwargs(_processor(), deployment, request)
        assert deployment == deploy_snap
        assert request == request_snap

    def test_ceilings_pin_the_limit_and_scope(self):
        """Verify the caps that bound each deployment request."""
        ceiling = {
            "image": VisionBudget(1568, 400000, UNIT, 28, 2),
            "video": VisionBudget(16 * UNIT, 400000, UNIT, 28, 2),
        }
        # Empty request: scoped output carries the pinned ceiling itself.
        result = resolve_serving_kwargs(
            _processor(), {"max_image_tokens": 999999}, {}, ceilings=ceiling
        )
        assert result["images_kwargs"]["max_pixels"] == 400000
        # Non-empty request resolving above the ceiling raises.
        with pytest.raises(ValueError, match="exceeds deployment limit"):
            resolve_serving_kwargs(
                _processor(),
                {"max_image_tokens": 999999},
                {"do_normalize": False},
                ceilings=ceiling,
            )

    def test_normalized_key_list_covers_budget_options(self):
        """The normalization contract lists exactly the budget options."""
        assert set(NORMALIZED_SERVING_KEYS) == {
            "min_pixels",
            "max_pixels",
            "min_image_tokens",
            "max_image_tokens",
            "max_frames",
            "max_frame_count_dynamic",
            "fps",
            "target_fps",
            "fps_interval",
            "do_sample_frames",
            "sampling_policy",
        }

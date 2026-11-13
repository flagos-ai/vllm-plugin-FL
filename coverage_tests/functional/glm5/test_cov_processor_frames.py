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

"""Single-GPU tests for the glm5 processor's frame samplers (M1).

``glm_sample_frame_indices`` (fps-interval policy) and its legacy
size-budget variant, the decoded-frame mapping and the prompt-visible
timestamp derivation. Each test re-derives the greedy timestamp walk,
its under/over-pick linspace fixups and the dedup/pad step with plain
scalar code inside the test itself.
"""

import math

import numpy as np
import pytest

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.processor import (
    glm_sample_frame_indices,
    glm_sample_frame_indices_legacy,
    glm_select_decoded_frames,
    glm_video_timestamp_seconds,
)

pytestmark = [pytest.mark.gpu]


def _ref_greedy_walk(total_frames, fps, duration, target_fps, tps=2):
    """Independent re-derivation of the greedy timestamp walk only."""
    picks, current = [], 0.0
    step = 1 / (tps * target_fps)
    for index in range(total_frames):
        if index / fps >= current:
            current += step
            picks.append(index)
            if current >= int(duration):
                break
    return picks


def _ref_greedy(total_frames, fps, duration, target_fps, max_frames, tps=2):
    """Independent re-derivation of the sampler's walk and its fixups."""
    if not duration:
        duration = (round((total_frames - 1) / fps) + 1) if fps else 0
    extract_t = max(tps, int(duration * target_fps))
    extract_t = min(extract_t, int(max_frames))
    if total_frames < extract_t:
        return [math.floor(i * total_frames / extract_t) for i in range(extract_t)]
    picks = _ref_greedy_walk(total_frames, fps, duration, target_fps, tps)
    if len(picks) < extract_t:  # under-pick: spread between first and last
        if not picks:
            start, end = 0, max(total_frames - 1, 0)
        else:
            start, end = picks[0], picks[-1]
        return np.linspace(start, end, extract_t, dtype=int).tolist()
    if len(picks) > extract_t:  # over-pick: re-spread over the whole clip
        return np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
    return picks


def _even_pad(indices):
    """Deduplicate and pad to an even temporal-patch multiple."""
    uniq = list(dict.fromkeys(int(i) for i in indices))
    if len(uniq) & 1:
        uniq.append(uniq[-1])
    return uniq


class TestGlmSampleFrameIndices:
    """The fps-interval sampler against a scalar reference walk."""

    def test_dense_clip_matches_greedy_reference(self):
        """Dense clip matches greedy reference."""
        total, fps, duration = 100, 25.0, 4.0
        got = glm_sample_frame_indices(
            total, fps, duration, target_fps=2.0, max_frame_count=8
        )
        assert got == _even_pad(_ref_greedy(total, fps, duration, 2.0, 8))

    def test_short_clip_spreads_at_even_timestamps(self):
        # 3 frames, extract_t = 4 -> floor(i * 3 / 4) spread, padded to even.
        """Short clip spreads at even timestamps."""
        got = glm_sample_frame_indices(3, 4.0, 1.0, target_fps=4.0, max_frame_count=8)
        assert got == _even_pad([math.floor(i * 3 / 4) for i in range(4)])

    def test_defaults_come_from_module_constants(self):
        """Defaults come from module constants."""
        from vllm_fl.models.glm5 import processor as P

        # fps=2.0 / max 2048 defaults; duration drives the count.
        got = glm_sample_frame_indices(10, 4.0, 2.5)
        expected = _ref_greedy(
            10, 4.0, 2.5, P.GLM_VIDEO_DEFAULT_FPS, P.GLM_VIDEO_DEFAULT_MAX_FRAMES
        )
        assert got == _even_pad(expected)
        assert got == [0, 2, 4, 6, 9, 9]

    def test_derived_duration_when_zero(self):
        # duration=0 falls back to (round((n-1)/fps) + 1) seconds.
        """Derived duration when zero."""
        got = glm_sample_frame_indices(9, 4.0, 0, target_fps=2.0, max_frame_count=6)
        derived = (round(8 / 4.0) + 1) / 1.0
        assert got == _even_pad(_ref_greedy(9, 4.0, derived, 2.0, 6))

    @pytest.mark.parametrize(
        ("total", "fps", "duration", "kwargs"),
        [
            (0, 4.0, 1.0, {}),
            (10, 0.0, 1.0, {}),
            (10, 4.0, 1.0, dict(target_fps=0.0)),
            (10, 4.0, 1.0, dict(max_frame_count=1)),
            (10, 4.0, -1.0, {}),
        ],
    )
    def test_invalid_inputs_raise(self, total, fps, duration, kwargs):
        """Invalid inputs raise."""
        with pytest.raises(ValueError):
            glm_sample_frame_indices(total, fps, duration, **kwargs)


class TestGlmSampleFrameIndicesLegacy:
    """The size-budget config's sampler (duration scaled by temporal patch)."""

    def test_matches_reference_with_default_cap(self):
        """Legacy extract_t scales by the temporal patch; the walk does not."""
        total, fps, duration = 10, 4.0, 2.5
        got = glm_sample_frame_indices_legacy(total, fps, duration, target_fps=2.0)
        extract_t = min(max(2, int(min(duration, 2400) * 2.0 * 2)), 640)
        picks = _ref_greedy_walk(total, fps, duration, 2.0)
        spread = np.linspace(picks[0], picks[-1], extract_t, dtype=int).tolist()
        assert got == _even_pad(spread)

    def test_short_clip_spreads_and_pads(self):
        """Short clip spreads and pads."""
        got = glm_sample_frame_indices_legacy(3, 4.0, 1.0, target_fps=2.0)
        # extract_t = min(max(2, int(1.0 * 2 * 2)), 640) = 4; 3 < 4 -> floor spread.
        assert got == _even_pad([math.floor(i * 3 / 4) for i in range(4)])

    def test_max_frames_caps_the_extract(self):
        """Max frames caps the extract."""
        got = glm_sample_frame_indices_legacy(
            100, 25.0, 4.0, target_fps=2.0, max_frame_count=8
        )
        picks = _ref_greedy_walk(100, 25.0, 4.0, 2.0)
        # 16 picks over-pick the cap of 8: re-spread over the whole clip.
        spread = np.linspace(0, 99, 8, dtype=int).tolist()
        assert len(picks) == 16
        assert got == _even_pad(spread)


class TestGlmSelectDecodedFrames:
    """Mapping GLM's picks onto the decoder's cached frame timeline."""

    @staticmethod
    def _metadata(**attrs):
        return type("Metadata", (), attrs)()

    @staticmethod
    def _nearest_rows(source, desired):
        """Independent nearest-neighbor snap with even-timeline padding."""
        source_arr = np.asarray(source, dtype=np.int64)
        wanted = np.asarray(desired)
        top = len(source_arr) - 1
        right = np.searchsorted(source_arr, wanted).clip(0, top)
        left = (right - 1).clip(0, top)
        nearest = np.where(
            np.abs(source_arr[left] - wanted) <= np.abs(source_arr[right] - wanted),
            left,
            right,
        )
        rows = list(dict.fromkeys(int(i) for i in nearest))
        rows += [rows[-1]] * (-len(rows) % 2)
        return rows, source_arr[rows].tolist()

    def test_nearest_available_frames_are_selected(self):
        # frames_indices=None -> the decoder timeline is range(num_frames).
        """Nearest available frames are selected."""
        desired = [0, 14, 28, 42, 56, 70]

        class FakeVP:
            temporal_patch_size = 2

            def sample_frames(self, timeline, **kwargs):
                assert timeline.total_num_frames == 4
                return desired

        rows, source = glm_select_decoded_frames(
            FakeVP(),
            self._metadata(
                fps=4.0, duration=1.0, total_num_frames=4, frames_indices=None
            ),
            4,
        )
        expected_rows, expected_source = self._nearest_rows(range(4), desired)
        assert rows == expected_rows
        assert source == expected_source

    def test_pre_sampled_source_snaps_to_decoder_indices(self):
        # frames_indices present: desired picks snap inside the cached list.
        """Presampled source snaps to decoder indices."""
        desired = [0, 14, 28]

        class FakeVP:
            temporal_patch_size = 2

            def sample_frames(self, timeline, **kwargs):
                return desired

        rows, source = glm_select_decoded_frames(
            FakeVP(),
            self._metadata(
                fps=4.0,
                duration=1.0,
                total_num_frames=4,
                frames_indices=[0, 10, 20, 30],
            ),
            4,
        )
        expected_rows, expected_source = self._nearest_rows([0, 10, 20, 30], desired)
        assert rows == expected_rows
        assert source == expected_source

    def test_presampled_video_without_sampling(self):
        """Presampled video without sampling."""

        class FakeVP:
            temporal_patch_size = 2

            def sample_frames(self, timeline, **kwargs):
                raise AssertionError("must not sample a pre-sampled video")

        rows, source = glm_select_decoded_frames(
            FakeVP(),
            self._metadata(fps=4.0, total_num_frames=4, frames_indices=[0, 1, 5, 7]),
            4,
            do_sample_frames=False,
        )
        assert rows == [0, 1, 2, 3]
        assert source == [0, 1, 5, 7]

    def test_missing_source_indices_with_mismatched_count_raises(self):
        """Missing source indices with mismatched count raises."""

        class FakeVP:
            temporal_patch_size = 2

            def sample_frames(self, timeline, **kwargs):
                raise AssertionError("must not be reached")

        with pytest.raises(ValueError, match="source frame indices"):
            glm_select_decoded_frames(
                FakeVP(), self._metadata(fps=4.0, total_num_frames=5), 4
            )

    def test_non_monotonic_source_indices_raise(self):
        """Nonmonotonic source indices raise."""

        class FakeVP:
            temporal_patch_size = 2

            def sample_frames(self, timeline, **kwargs):
                raise AssertionError("must not be reached")

        with pytest.raises(ValueError, match="Invalid pre-sampled"):
            glm_select_decoded_frames(
                FakeVP(),
                self._metadata(fps=4.0, total_num_frames=3, frames_indices=[0, 2, 1]),
                3,
            )


class TestGlmVideoTimestampSeconds:
    """Prompt-visible second indices: every temporal-patch-th pick."""

    @staticmethod
    def _vp(picks):
        class FakeVP:
            def sample_frames(self, metadata, **kwargs):
                return picks

        return FakeVP()

    def test_seconds_are_every_second_pick(self):
        """Take timestamps from every second sampled frame."""
        vp = self._vp([0, 2, 4, 6, 9, 9])
        metadata = type("M", (), dict(fps=4.0, duration=2.5, total_num_frames=10))()
        seconds = glm_video_timestamp_seconds(vp, metadata, target_fps=2.0)
        all_seconds = [int(i / 4.0) for i in [0, 2, 4, 6, 9, 9]]
        assert seconds == all_seconds[::2]

    def test_presampled_frames_skip_the_sampler(self):
        """Presampled frames skip the sampler."""

        class FakeVP:
            def sample_frames(self, metadata, **kwargs):
                raise AssertionError("must not sample pre-sampled frames")

        metadata = type(
            "M", (), dict(fps=4.0, frames_indices=[1, 3, 5], do_sample_frames=False)
        )()
        assert glm_video_timestamp_seconds(FakeVP(), metadata) == [0, 1]

    def test_empty_selection_returns_no_timestamps(self):
        """Empty selection returns no timestamps."""
        vp = self._vp([])
        metadata = type(
            "M", (), dict(fps=None, frames_indices=[], do_sample_frames=False)
        )()
        assert glm_video_timestamp_seconds(vp, metadata) == []

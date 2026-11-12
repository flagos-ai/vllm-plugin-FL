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

"""Single-GPU tests for vllm_fl/models/glm5/processor.py (M1).

The processor classes' dispatch surface: ``sample_frames`` policy
routing, the RGB channel contract, the CPU-side patchify geometry and
the host-side call paths (token counting, mm-token flags, text decode).
No checkpoint is loaded and no engine is started. Frame-sampler math
lives in ``test_cov_processor_frames.py``; budget/resize contracts in
``test_cov_processor_geometry.py``.
"""

import numpy as np
import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from .test_cov_processor_frames import _even_pad, _ref_greedy, _ref_greedy_walk
from vllm_fl.models.glm5.processor import (
    Glm5NextImageProcessor,
    Glm5NextVideoProcessor,
)

pytestmark = [pytest.mark.gpu]

TWO_FRAMES = __import__(
    "transformers.video_utils", fromlist=["VideoMetadata"]
).VideoMetadata


class TestProcessorSampling:
    """sample_frames dispatch and the RGB conversion contract."""

    def test_fps_interval_policy_matches_reference(self):
        """Compare FPS interval sampling with independently computed timestamps."""
        processor = Glm5NextVideoProcessor()
        metadata = TWO_FRAMES(fps=4.0, duration=2.5, total_num_frames=10)
        got = processor.sample_frames(metadata, fps=2.0).tolist()
        assert got == _even_pad(_ref_greedy(10, 4.0, 2.5, 2.0, 2048))

    def test_legacy_dynamic_policy_duration_cap(self):
        """Legacy dynamic policy duration cap."""
        processor = Glm5NextVideoProcessor(sampling_policy="legacy_dynamic")
        metadata = TWO_FRAMES(fps=4.0, duration=2.5, total_num_frames=10)
        got = processor.sample_frames(metadata, fps=2.0).tolist()
        # extract_t = min(max(2, int(2.5*2*2)), 640) = 10 -> all ten frames.
        assert got == _even_pad(list(range(10)))

    @pytest.mark.parametrize(
        ("duration", "tier_fps"),
        [(30.0, 3.0), (300.0, 1.0), (301.0, 0.5)],
    )
    def test_legacy_dynamic_auto_tier_by_duration(self, duration, tier_fps):
        """Without a request override the tier picks fps from duration."""
        processor = Glm5NextVideoProcessor(sampling_policy="legacy_dynamic")
        total = int(duration * 2)
        metadata = TWO_FRAMES(fps=2.0, duration=duration, total_num_frames=total)
        got = processor.sample_frames(metadata).tolist()
        # The tier's fps drives extract_t; the walk keeps the same interval.
        extract_t = min(max(2, int(min(duration, 2400) * tier_fps * 2)), 640)
        picks = _ref_greedy_walk(total, 2.0, duration, tier_fps)
        if len(picks) == extract_t:
            expected = picks + [picks[-1]] * (-len(picks) % 2)
        else:
            spread = np.linspace(0, total - 1, extract_t, dtype=int).tolist()
            expected = list(dict.fromkeys(spread))
            expected += [expected[-1]] * (-len(expected) % 2)
        assert got == expected

    def test_missing_metadata_raises(self):
        """Missing metadata raises."""
        processor = Glm5NextVideoProcessor()
        with pytest.raises(ValueError, match="no video metadata"):
            processor.sample_frames(None)
        with pytest.raises(ValueError, match="no video metadata"):
            processor.sample_frames(
                TWO_FRAMES(fps=None, duration=1.0, total_num_frames=4)
            )

    def test_convert_to_rgb_channel_contract(self):
        """Check RGB conversion for the supported image channel layouts."""
        gray = torch.zeros(2, 1, 4, 4)
        assert Glm5NextVideoProcessor.convert_to_rgb(None, gray).shape == (2, 3, 4, 4)
        rgb = torch.zeros(2, 3, 4, 4)
        assert Glm5NextVideoProcessor.convert_to_rgb(None, rgb) is rgb
        with pytest.raises(ValueError, match="1, 3 or 4 channels"):
            Glm5NextVideoProcessor.convert_to_rgb(None, torch.zeros(2, 5, 4, 4))

    def test_convert_to_rgb_composites_alpha_over_white(self):
        """Convert to RGB composites alpha over white."""
        rgba = torch.zeros(2, 4, 3, 3)  # (N, C=4, H, W)
        rgba[..., 3, :, :] = 128  # alpha ~ 0.502
        out = Glm5NextVideoProcessor.convert_to_rgb(None, rgba)
        alpha = 128 / 255.0
        expected = (1 - alpha) * 255
        assert torch.allclose(
            out[:, 0], torch.full_like(out[:, 0], expected), atol=1e-2
        )


class TestProcessorPatchify:
    """End-to-end CPU patchify on tiny tensors (device-agnostic math)."""

    IMAGE_KWARGS = dict(
        do_resize=True,
        size=None,
        resample=None,
        do_rescale=False,
        rescale_factor=1 / 255.0,
        do_normalize=False,
        image_mean=None,
        image_std=None,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        patch_expand_factor=1,
        resize_mode="pad",
        min_image_tokens=16,
        max_image_tokens=64,
        disable_grouping=False,
        return_tensors=None,
    )

    def test_image_patchify_shape_and_grid(self):
        """Image patchify shape and grid."""
        processor = Glm5NextImageProcessor()
        out = processor._preprocess([torch.rand(2, 3, 56, 56)], **self.IMAGE_KWARGS)
        grid = out["image_grid_thw"].tolist()
        assert grid == [[1, 8, 8]]
        # patches flattened: n_patches * (C * t * patch * patch)
        assert out["pixel_values"].shape == (64, 3 * 2 * 14 * 14)

    def test_video_patchify_pads_odd_frame_counts(self):
        """Video patchify pads odd frame counts."""
        processor = Glm5NextVideoProcessor()
        out = processor._preprocess(
            [torch.rand(3, 3, 56, 56)],
            do_rescale=False,
            do_normalize=False,
            return_tensors=None,
            video_metadata=TWO_FRAMES(fps=4.0, duration=1.0, total_num_frames=3),
        )
        # 3 frames are padded to 4 -> grid_t = 2.
        assert out["video_grid_thw"].tolist() == [[2, 6, 6]]

    def test_video_frame_cap_rejects_oversized_decodes(self):
        """Video frame cap rejects oversized decodes."""
        processor = Glm5NextVideoProcessor()
        # 4 frames padded to the temporal patch size is 4, above a cap of 2.
        with pytest.raises(ValueError, match="max_frames"):
            processor._preprocess(
                [torch.rand(4, 3, 56, 56)],
                do_rescale=False,
                do_normalize=False,
                return_tensors=None,
                video_metadata=TWO_FRAMES(fps=4.0, duration=1.0, total_num_frames=4),
                max_frames=2,
            )


class TestProcessorCallPaths:
    """Host-side composition: token counting and text decode hand-off."""

    @staticmethod
    def _processor(image_calls=None, video_calls=None):
        class Tok:
            init_kwargs = {}

            def __call__(self, text, **kwargs):
                return {"input_ids": [[1, 7, 7, 2]]}

            def batch_decode(self, outputs, **kwargs):
                return ["decoded"]

        class Img:
            def __call__(self, images, **kwargs):
                if image_calls is not None:
                    image_calls.append(kwargs)
                return {"pixel_values": "PV"}

        class Vid:
            def __call__(self, videos, **kwargs):
                if video_calls is not None:
                    video_calls.append(kwargs)
                return {"pixel_values_videos": "PVV", "video_metadata": "MD"}

        # Glm5NextProcessor composition is exercised through its mixin methods
        # on a bare instance (no checkpoint needed).
        from vllm_fl.models.glm5.processor import Glm5NextProcessor

        proc = Glm5NextProcessor.__new__(Glm5NextProcessor)
        proc.tokenizer = Tok()
        proc.image_processor = Img()
        proc.video_processor = Vid()
        proc.image_token_id = 7
        return proc

    def test_call_merges_all_modalities(self):
        """Call merges all modalities."""
        proc = self._processor()
        out = proc._call_with_resolved_kwargs(images=[1], videos=[2], text="hi")
        assert set(out) == {"input_ids", "pixel_values", "pixel_values_videos"}

    def test_mm_token_type_ids_flag_marks_image_tokens(self):
        """Multimodal token type ids flag marks image tokens."""
        proc = self._processor()
        out = proc._call_with_resolved_kwargs(
            images=[1], videos=[2], text="hi", return_mm_token_type_ids=True
        )
        assert out["mm_token_type_ids"] == [[0, 1, 1, 0]]

    def test_num_multimodal_tokens_counts_grids(self):
        """Num multimodal tokens counts grids."""
        proc = self._processor()
        proc.image_processor.merge_size = 2
        proc.image_processor.get_number_of_image_patches = (
            lambda height, width, kwargs: 8
        )
        proc.video_processor.merge_size = 2
        proc.video_processor.get_number_of_video_patches = (
            lambda frames, height, width, kwargs: 16
        )
        data = proc._get_num_multimodal_tokens(
            image_sizes=[(56, 56)], video_sizes=[(4, 56, 56)]
        )
        assert data.num_image_tokens == [8 // 4]
        assert data.num_image_patches == [8]
        assert data.num_video_tokens == [16 // 4]

    def test_post_process_decodes_text(self):
        """Post process decodes text."""
        proc = self._processor()
        assert proc.post_process_image_text_to_text([[1, 2]]) == ["decoded"]

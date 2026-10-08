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

"""vLLM 0.24 adapter for MiniCPM-V 4.7."""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.minicpmv import MiniCPMVDummyInputsBuilder
from vllm.model_executor.models.minicpmv4_6 import (
    Idefics2VisionTransformer,
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6Merger,
    MiniCPMV4_6MultiModalProcessor,
    MiniCPMV4_6ProcessingInfo,
    MiniCPMV4_6ViTWindowAttentionMerger,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.tokenizers import get_tokenizer

_CANVAS_TOKEN_ATTRS = {
    "image_start_id": "image_start_token",
    "image_end_id": "image_end_token",
    "slice_start_id": "slice_start_token",
    "slice_end_id": "slice_end_token",
}


def _resolve_canvas_token_ids(config: Any, tokenizer: Any) -> None:
    """Populate marker IDs missing from pre-upstream MiniCPM-V 4.7 configs."""
    for config_attr, tokenizer_attr in _CANVAS_TOKEN_ATTRS.items():
        if getattr(config, config_attr, None) is not None:
            continue
        token = getattr(tokenizer, tokenizer_attr, None)
        if token is None:
            raise ValueError(f"Tokenizer is missing {tokenizer_attr}")
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise ValueError(f"Tokenizer cannot resolve {tokenizer_attr}={token!r}")
        setattr(config, config_attr, int(token_id))

    if getattr(config, "newline_id", None) is None:
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(
                "MiniCPM-V 4.7 Canvas M-RoPE requires newline to encode to "
                f"one token, got {newline_ids}"
            )
        config.newline_id = int(newline_ids[0])


def _frame_end_idx(
    last_crop_end_idx: int,
    input_ids: list[int],
    markers: tuple[int, ...],
    limit: int,
) -> int:
    end_idx = last_crop_end_idx
    while end_idx < limit and input_ids[end_idx] in markers:
        end_idx += 1
    return end_idx


def _group_visual_frames(
    config: Any,
    input_ids: list[int],
    mm_token_type_ids: list[int],
) -> list[tuple[int, int, list[tuple[int, int, int, int, list[tuple[int, int]]]]]]:
    """Group thumbnail and slice spans using the official 4.7 canvas rules."""
    markers = (
        config.slice_start_id,
        config.slice_end_id,
        config.image_start_id,
        config.image_end_id,
        config.newline_id,
    )
    slice_start_id = markers[0]

    seq_len = len(input_ids)
    groups = []
    open_slices = None
    previous_end_idx = 0
    for modality, run in itertools.groupby(
        enumerate(mm_token_type_ids), lambda item: item[1]
    ):
        if modality == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1)

        if open_slices is not None and input_ids[crop[0] - 1] == slice_start_id:
            open_slices.append(crop)
        else:
            open_slices = []
            frame = (modality, crop[0], crop[1], open_slices)
            gap = input_ids[previous_end_idx : crop[0] - 1]
            adjacent = bool(groups) and all(token_id in markers for token_id in gap)
            if adjacent:
                groups[-1][1].append(frame)
            else:
                groups.append((crop[0] - 1, [frame]))
        previous_end_idx = crop[1]

    visual_groups = []
    for group_start_idx, frames in groups:
        frames_with_ends = [
            (
                modality,
                thumb_start_idx,
                thumb_end_idx,
                _frame_end_idx(
                    slices[-1][1] if slices else thumb_end_idx,
                    input_ids,
                    markers,
                    seq_len,
                ),
                slices,
            )
            for modality, thumb_start_idx, thumb_end_idx, slices in frames
        ]
        group_end_idx = frames_with_ends[-1][3]
        visual_groups.append((group_start_idx, group_end_idx, frames_with_ends))
    return visual_groups


def _get_vision_position_ids(
    *,
    start_position: int,
    grid_hw: list[int] | torch.Tensor,
    canvas_height: int = 0,
    canvas_width: int = 0,
    h_offset: int = 0,
    w_offset: int = 0,
    spatial_merge_size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    llm_grid_h = int(grid_hw[0]) // spatial_merge_size
    llm_grid_w = int(grid_hw[1]) // spatial_merge_size
    canvas_height = canvas_height or llm_grid_h
    canvas_width = canvas_width or llm_grid_w

    h_coords = (
        torch.linspace(0, canvas_height - 1, llm_grid_h, device=device).round().long()
        + h_offset
    )
    w_coords = (
        torch.linspace(0, canvas_width - 1, llm_grid_w, device=device).round().long()
        + w_offset
    )
    h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
    t_grid = torch.zeros_like(h_grid)
    return torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1) + start_position


def _get_canvas_mrope_positions(
    config: Any,
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    target_sizes: torch.Tensor | None,
    target_sizes_videos: torch.Tensor | None,
    *,
    downsample_mode: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Return official MiniCPM-V 4.7 Canvas M-RoPE positions for one prompt."""
    downsample_mode = downsample_mode or config.downsample_mode
    merge_factor = 2 if downsample_mode == "4x" else 4
    device = input_ids.device
    seq_len = len(input_ids)
    position_ids = torch.arange(seq_len, device=device).expand(3, -1).clone()
    grid_iters = {
        1: iter(target_sizes.tolist()) if target_sizes is not None else None,
        2: iter(target_sizes_videos.tolist())
        if target_sizes_videos is not None
        else None,
    }

    current_input_ids = input_ids.tolist()
    current_mm_token_type_ids = mm_token_type_ids.tolist()
    current_pos = 0
    current_idx = 0
    for group_start_idx, group_end_idx, frames in _group_visual_frames(
        config, current_input_ids, current_mm_token_type_ids
    ):
        if group_start_idx > current_idx:
            text_len = group_start_idx - current_idx
            position_ids[:, current_idx:group_start_idx] = (
                torch.arange(text_len, device=device) + current_pos
            )
            current_pos += text_len

        frame_cursor_idx = group_start_idx
        for modality, thumb_start_idx, thumb_end_idx, frame_end_idx, slices in frames:
            grid_iter = grid_iters[modality]
            if grid_iter is None:
                raise ValueError(f"Missing target sizes for modality {modality}")
            target_sizes_thumb = next(grid_iter)
            frame_start_idx = thumb_start_idx - 1

            if frame_start_idx > frame_cursor_idx:
                gap_len = frame_start_idx - frame_cursor_idx
                position_ids[:, frame_cursor_idx:frame_start_idx] = (
                    torch.arange(gap_len, device=device) + current_pos
                )
                current_pos += gap_len + 1

            frame_end_idx = min(frame_end_idx, group_end_idx)
            canvas_origin = current_pos
            halo_before_canvas = max(canvas_origin - 1, 0)

            llm_slice_h, llm_slice_w = 0, 0
            num_cols = 0
            canvas_height = target_sizes_thumb[0] // merge_factor
            canvas_width = target_sizes_thumb[1] // merge_factor
            target_sizes_first_slice = None
            if slices:
                num_cols = len(slices)
                for k in range(len(slices) - 1):
                    if slices[k + 1][0] - slices[k][1] > 2:
                        num_cols = k + 1
                        break
                num_rows = len(slices) // num_cols if num_cols > 0 else 1
                if num_rows * num_cols != len(slices):
                    num_rows, num_cols = 1, len(slices)
                target_sizes_first_slice = next(grid_iter)
                llm_slice_h = target_sizes_first_slice[0] // merge_factor
                llm_slice_w = target_sizes_first_slice[1] // merge_factor
                canvas_height = num_rows * llm_slice_h
                canvas_width = num_cols * llm_slice_w

            position_ids[:, frame_start_idx:frame_end_idx] = canvas_origin
            position_ids[1:, frame_start_idx] = halo_before_canvas
            if thumb_end_idx < frame_end_idx:
                position_ids[1, thumb_end_idx] = canvas_origin + canvas_height
                position_ids[2, thumb_end_idx] = canvas_origin + canvas_width

            position_ids[:, thumb_start_idx:thumb_end_idx] = _get_vision_position_ids(
                start_position=canvas_origin,
                grid_hw=target_sizes_thumb,
                canvas_height=canvas_height,
                canvas_width=canvas_width,
                spatial_merge_size=merge_factor,
                device=device,
            )

            for k, (slice_start_idx, slice_end_idx) in enumerate(slices):
                h_off = (k // num_cols) * llm_slice_h
                w_off = (k % num_cols) * llm_slice_w
                target_sizes_slice = (
                    target_sizes_first_slice if k == 0 else next(grid_iter)
                )
                slice_h = target_sizes_slice[0] // merge_factor
                slice_w = target_sizes_slice[1] // merge_factor

                slice_start_marker_idx = slice_start_idx - 1
                if slice_start_marker_idx >= frame_start_idx:
                    position_ids[1, slice_start_marker_idx] = canvas_origin + h_off
                    position_ids[2, slice_start_marker_idx] = canvas_origin + w_off
                if slice_end_idx < frame_end_idx:
                    position_ids[1, slice_end_idx] = canvas_origin + h_off + slice_h - 1
                    position_ids[2, slice_end_idx] = canvas_origin + w_off + slice_w - 1

                position_ids[:, slice_start_idx:slice_end_idx] = (
                    _get_vision_position_ids(
                        start_position=canvas_origin,
                        grid_hw=target_sizes_slice,
                        h_offset=h_off,
                        w_offset=w_off,
                        spatial_merge_size=merge_factor,
                        device=device,
                    )
                )

            for k in range(len(slices) - 1):
                gap_start_idx, gap_end_idx = slices[k][1], slices[k + 1][0]
                if gap_end_idx - gap_start_idx <= 2:
                    continue
                boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                right_edge_w = num_cols * llm_slice_w
                for newline_idx in range(gap_start_idx + 1, gap_end_idx - 1):
                    position_ids[1, newline_idx] = canvas_origin + boundary_h
                    position_ids[2, newline_idx] = canvas_origin + right_edge_w

            current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
            frame_cursor_idx = frame_end_idx

        if frame_cursor_idx < group_end_idx:
            trail_len = group_end_idx - frame_cursor_idx
            position_ids[:, frame_cursor_idx:group_end_idx] = (
                torch.arange(trail_len, device=device) + current_pos
            )
            current_pos += trail_len
        current_idx = group_end_idx

    if current_idx < seq_len:
        position_ids[:, current_idx:seq_len] = (
            torch.arange(seq_len - current_idx, device=device) + current_pos
        )
    rope_delta = int(position_ids.max().item() + 1 - seq_len)
    return position_ids, rope_delta


def _gather_target_sizes(
    mm_features: list[MultiModalFeatureSpec],
    *,
    modality: str,
    key: str,
) -> torch.Tensor | None:
    chunks = []
    for feature in sorted(mm_features, key=lambda item: item.mm_position.offset):
        if feature.modality != modality:
            continue
        if feature.data is None or key not in feature.data:
            raise ValueError(f"Missing {key} for cached {modality} feature")
        value = feature.data[key].data
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        chunks.append(tensor.reshape(-1, 2).to(dtype=torch.long, device="cpu"))
    return torch.cat(chunks) if chunks else None


class MiniCPMV4_7ProcessingInfo(MiniCPMV4_6ProcessingInfo):
    def get_model_version(self):
        return (4, 7)


def _select_language_model_cls(text_config: Any) -> type[nn.Module]:
    model_type = getattr(text_config, "model_type", "")
    if model_type == "qwen3_5_moe_text" or getattr(text_config, "num_experts", 0):
        return Qwen3_5MoeForCausalLM
    return Qwen3_5ForCausalLM


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMV4_6MultiModalProcessor,
    info=MiniCPMV4_7ProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMV4_7ForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    """MiniCPM-V 4.7 with a config-selected Qwen3.5 text tower."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # The 4.6 constructor hard-codes the dense Qwen3.5 class. Reuse its
        # multimodal components and select the text implementation from the
        # checkpoint config so both dense and MoE variants are supported.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        if getattr(config, "mrope_mode", None) == "canvas" and any(
            getattr(config, name, None) is None
            for name in (*_CANVAS_TOKEN_ATTRS, "newline_id")
        ):
            tokenizer = get_tokenizer(
                vllm_config.model_config.tokenizer,
                revision=vllm_config.model_config.tokenizer_revision,
                tokenizer_mode=vllm_config.model_config.tokenizer_mode,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
            )
            _resolve_canvas_token_ids(config, tokenizer)

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        with self._mark_tower_model(vllm_config, {"image"}):
            self.vpm = Idefics2VisionTransformer(
                config.vision_config,
                quant_config=quant_config,
                apply_encoder_attention_mask=True,
                prefix=maybe_prefix(prefix, "vpm"),
            )
            if config.drop_vision_last_layer:
                self.vpm.encoder.layers = self.vpm.encoder.layers[:-1]

            self.vit_merger = MiniCPMV4_6ViTWindowAttentionMerger(
                config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vit_merger"),
            )
            self.merger = MiniCPMV4_6Merger(
                hidden_size=config.vision_config.hidden_size,
                llm_embed_dim=config.text_config.hidden_size,
            )

        text_model_type = getattr(config.text_config, "model_type", "")
        language_model_cls = _select_language_model_cls(config.text_config)

        with self._mark_language_model(vllm_config):
            saved_model_type = config.model_type
            config.model_type = text_model_type
            try:
                self.language_model = language_model_cls(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                )
            finally:
                config.model_type = saved_model_type

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        if getattr(self.config, "mrope_mode", None) != "canvas" or not mm_features:
            seq_len = len(input_tokens)
            positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
            return positions, 0

        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.config.image_token_id] = 1
        mm_token_type_ids[input_ids == self.config.video_token_id] = 2

        target_sizes = _gather_target_sizes(
            mm_features,
            modality="image",
            key="tgt_sizes",
        )
        target_sizes_videos = _gather_target_sizes(
            mm_features,
            modality="video",
            key="video_tgt_sizes",
        )
        return _get_canvas_mrope_positions(
            self.config,
            input_ids,
            mm_token_type_ids,
            target_sizes,
            target_sizes_videos,
        )


def _install_transformers_video_doc_compat() -> None:
    """Backfill a docstring-only symbol used by the supplied processor."""
    import transformers.video_processing_utils as video_processing_utils

    if not hasattr(video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING"):
        video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING = ""


def register_minicpmv4_7() -> None:
    _install_transformers_video_doc_compat()
    ModelRegistry.register_model(
        "MiniCPMV4_7ForConditionalGeneration",
        MiniCPMV4_7ForConditionalGeneration,
    )

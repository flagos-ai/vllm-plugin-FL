# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.
"""Hy4 multimodal input adaptation, vision tower and language composition."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from functools import partial
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldConfig
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import (
    _cached_encode as cached_encode,
    replace_token_matches,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .model_weights import hyv4_packed_modules_mapping
from vllm_fl.models.hy_v4.config import HYV4VLConfig
from vllm_fl.models.hy_v4.mm_input_norm import IdentityInputNorm, build_mm_input_norm
from vllm_fl.models.hy_v4.processor import (
    HYV4VLProcessor,
    adaptive_resize,
    smart_resize,
    vision_token_count,
    vision_token_grid,
)

# Multimodal request processing


class HYV4VLDataParser(MultiModalDataParser):
    def _parse_image_data(self, data):
        if isinstance(data, dict) or self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed image embeddings")
        return super()._parse_image_data(data)

    def _parse_video_data(self, data):
        if isinstance(data, dict) or self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed video embeddings")
        return super()._parse_video_data(data)


class HYV4VLProcessingInfo(BaseProcessingInfo):
    def _merge_and_resolve_mm_processor_kwargs(self, kwargs):
        # v0.28 has the merge API but not flat-to-scoped kwarg routing.
        from typing import get_type_hints

        processor = self.get_hf_processor()
        merged = dict(self.ctx.get_merged_mm_kwargs(kwargs))
        scopes = {
            "images_kwargs": get_type_hints(processor.image_processor.valid_kwargs),
            "videos_kwargs": get_type_hints(processor.video_processor.valid_kwargs),
        }
        for key, value in tuple(merged.items()):
            destinations = [scope for scope, fields in scopes.items() if key in fields]
            for scope in destinations:
                scoped = dict(merged.get(scope, {}))
                scoped.setdefault(key, value)
                merged[scope] = scoped
            if destinations:
                del merged[key]
        return merged

    def get_hf_config(self) -> HYV4VLConfig:
        return self.ctx.get_hf_config(HYV4VLConfig)

    def get_hf_processor(self, **kwargs) -> HYV4VLProcessor:
        return self.ctx.get_hf_processor(HYV4VLProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_max_frames_per_video(self) -> int:
        processor = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        return int(options.get("max_frames", processor.max_frames))

    def get_data_parser(self):
        return HYV4VLDataParser(video_needs_metadata=True)

    def get_image_size_with_most_features(self):
        processor = self.get_hf_processor().image_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "images_kwargs", {}
        )
        factor = processor.patch_size * processor.merge_size
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        minimum = options.get("min_pixels", processor.size["shortest_edge"])
        cells = max(1, maximum // factor**2)
        candidates = []
        for rows in range(1, math.isqrt(cells * 200) + 1):
            cols = min(cells // rows, rows * 200)
            if cols and rows <= cols * 200:
                candidates.append((rows * (cols + 1), rows, cols))
        _, rows, cols = max(candidates)
        height, width = smart_resize(
            rows * factor, cols * factor, factor, minimum, maximum
        )
        return ImageSize(width=width, height=height)

    def get_dummy_video_shape(self, seq_len, mm_counts):
        processor = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        config = self.get_hf_config().vision_config
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        max_frames = options.get("max_frames", processor.max_frames)
        side = options.get("video_max_image_size", processor.video_max_image_size)
        available = max(1, seq_len // max(1, sum(mm_counts.values())))
        best = (
            0,
            1,
            config.patch_size * config.spatial_merge_size,
            config.patch_size * config.spatial_merge_size,
        )
        for frames in range(1, min(max_frames, available) + 1):
            height, width = adaptive_resize(
                side,
                side,
                frames,
                config.patch_size,
                config.spatial_merge_size,
                maximum,
                side,
                options.get(
                    "timestamp_token_per_frame", processor.timestamp_token_per_frame
                ),
                options.get(
                    "timestamp_token_reserve", processor.timestamp_token_reserve
                ),
            )
            grid = (frames, height // config.patch_size, width // config.patch_size)
            count = vision_token_count(grid, config, "video")
            overhead = frames * (processor.timestamp_token_per_frame + 2) + 2
            if count + overhead <= available and count > best[0]:
                best = (count, frames, height, width)
        return best[1:]

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        config = self.get_hf_config().vision_config
        size = self.get_image_size_with_most_features()
        image_tokens = vision_token_count(
            (1, size.height // config.patch_size, size.width // config.patch_size),
            config,
            "image",
        )
        video = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        factor = config.patch_size * config.spatial_merge_size
        max_frames = options.get("max_frames", video.max_frames)
        max_pixels = options.get("max_pixels", video.size["longest_edge"])
        video_tokens = 2 * (max_pixels // factor**2 + max_frames)
        video_tokens += 2 * max_frames * int(bool(config.cat_extra_token))
        return {"image": image_tokens, "video": video_tokens}


class HYV4VLDummyInputsBuilder(BaseDummyInputsBuilder[HYV4VLProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        processor = self.info.get_hf_processor()
        return "".join(
            processor.get_placeholder(modality) * mm_counts.get(modality, 0)
            for modality in ("image", "video")
        )

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        size = self.info.get_image_size_with_most_features()
        frames, height, width = self.info.get_dummy_video_shape(seq_len, mm_counts)
        videos = self._get_dummy_videos(
            width=width,
            height=height,
            num_frames=frames,
            num_videos=mm_counts.get("video", 0),
            overrides=mm_options.get("video"),
        )
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": [
                (
                    video,
                    {
                        "fps": 2.0,
                        "duration": len(video) / 2,
                        "total_num_frames": len(video),
                        "frames_indices": list(range(len(video))),
                        "do_sample_frames": False,
                        "video_backend": "opencv",
                    },
                )
                for video in videos
            ],
        }


class HYV4VLMultiModalProcessor(BaseMultiModalProcessor[HYV4VLProcessingInfo]):
    def apply(self, inputs, timing_ctx):
        processor = self.info.get_hf_processor(**inputs.hf_processor_mm_kwargs)
        prompt = inputs.prompt
        if isinstance(prompt, str):
            prompt = cached_encode(
                processor.tokenizer, prompt, add_special_tokens=False
            )
        for modality in ("image", "video"):
            if inputs.mm_data_items.get_count(modality, strict=False):
                target = [
                    getattr(processor, f"{modality}_start_token_id"),
                    getattr(processor, f"{modality}_token_id"),
                    getattr(processor, f"{modality}_end_token_id"),
                ]
                prompt = replace_token_matches(
                    prompt,
                    target,
                    cached_encode(
                        processor.tokenizer, f"<{modality}>", add_special_tokens=False
                    ),
                )
        inputs = replace(inputs, prompt=prompt)
        options = self.info.ctx.get_merged_mm_kwargs(inputs.hf_processor_mm_kwargs)
        intervals = options.get("second_per_grid_ts")
        if intervals is not None:
            videos = inputs.mm_data_items.get("video", ())
            if len(intervals) != len(videos):
                raise ValueError("second_per_grid_ts must match video count")
            normalized = []
            for item, interval in zip(videos, intervals):
                if item is None:
                    raise ValueError("Video timing overrides require media metadata")
                frames, metadata = item
                normalized.append(
                    (frames, {**metadata, "second_per_grid_t": float(interval)})
                )
            mm_items = MultiModalDataItems(
                {**inputs.mm_data_items, "video": VideoProcessorItems(normalized)}
            )
            kwargs = dict(inputs.hf_processor_mm_kwargs)
            kwargs.pop("second_per_grid_ts", None)
            inputs = replace(
                inputs, mm_data_items=mm_items, hf_processor_mm_kwargs=kwargs
            )
        return super().apply(inputs, timing_ctx)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        result = {}
        for modality, pixel_key in (
            ("image", "pixel_values"),
            ("video", "pixel_values_videos"),
        ):
            grid_key = f"{modality}_grid_thw"
            grid = hf_inputs.get(grid_key, torch.empty((0, 3), dtype=torch.long))
            result[pixel_key] = MultiModalFieldConfig.flat_from_sizes(
                modality, grid.prod(-1)
            )
            result[grid_key] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
            if modality == "video":
                tau = self.info.get_hf_config().vision_config.temporal_patch_size
                result["video_timestamps"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", (grid[:, 0] + tau - 1) // tau, keep_on_cpu=True
                )
        return result

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def replacement(index: int, modality: str) -> PromptUpdateDetails:
            item = out_mm_kwargs[modality][index]
            grid = item[f"{modality}_grid_thw"].data
            timestamps = item["video_timestamps"].data if modality == "video" else None
            tokens = [
                getattr(processor, f"{modality}_start_token_id"),
                *processor.get_replacement(modality, grid, timestamps),
                getattr(processor, f"{modality}_end_token_id"),
            ]
            token_id = getattr(processor, f"{modality}_token_id")
            return PromptUpdateDetails.select_token_id(tokens, token_id)

        return [
            PromptReplacement(
                modality=modality,
                target=cached_encode(
                    processor.tokenizer, f"<{modality}>", add_special_tokens=False
                ),
                replacement=partial(replacement, modality=modality),
            )
            for modality in ("image", "video")
        ]


# Vision transformer and projector


class HYV4VisionAttention(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        use_dp = is_vit_use_data_parallel(config.num_attention_heads)
        tp = 1 if use_dp else get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads // tp
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_dp,
        )
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=use_dp,
        )
        self.attn = MMEncoderAttention(
            self.num_heads, self.head_dim, self.head_dim**-0.5, prefix=f"{prefix}.attn"
        )
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True, enable_fp32_compute=True
        )

    def forward(
        self, x, cu_seqlens, max_seqlen, cos=None, sin=None, sequence_lengths=None
    ):
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        if cos is not None:
            shape = (*q.shape[:2], self.num_heads, self.head_dim)
            q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
            q, k = self.apply_rotary_emb(torch.cat((q, k), dim=0), cos, sin).chunk(
                2, dim=0
            )
        if self.attn.attn_backend == AttentionBackendEnum.FLASHINFER:
            q, k = q.contiguous(), k.contiguous()
        out = self.attn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        return self.o_proj(out.reshape(*x.shape[:2], -1))[0]


class HYV4VisionMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        use_dp = is_vit_use_data_parallel(config.num_attention_heads)
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_h_to_4h",
            disable_tp=use_dp,
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h",
            disable_tp=use_dp,
        )
        self.act_fn = get_act_fn(config.hidden_act)

    def forward(self, x):
        x = self.act_fn(self.dense_h_to_4h(x)[0])
        return self.dense_4h_to_h(x)[0]


class HYV4VisionBlock(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = HYV4VisionAttention(
            config, quant_config, f"{prefix}.self_attn"
        )
        self.mlp = HYV4VisionMLP(config, quant_config, f"{prefix}.mlp")

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.input_layernorm(x), **kwargs)
        return x + self.mlp(self.post_attention_layernorm(x))


class HYV4VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_edge = config.max_image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.position_embedding = nn.Embedding(
            self.position_edge**2 + 1, config.hidden_size, dtype=torch.float32
        )

    def _patch_conv(self, pixel_values: torch.Tensor) -> torch.Tensor:
        config = self.config
        patches = pixel_values.reshape(
            -1, config.num_channels, config.patch_size, config.patch_size
        )
        return self.patch_embedding(patches).flatten(2).squeeze(-1).unsqueeze(0)

    def compute_pos_embeds(
        self, grid_thw: list[list[int]], dtype: torch.dtype
    ) -> torch.Tensor:
        """Interpolated position embedding per patch, ``(num_patches, hidden)``."""
        positions = (
            self.position_embedding.weight[1:]
            .float()
            .reshape(1, self.position_edge, self.position_edge, self.config.hidden_size)
            .permute(0, 3, 1, 2)
        )
        positional = []
        for t, h, w in grid_thw:
            resized = F.interpolate(
                positions,
                scale_factor=(
                    (h + 0.1) / self.position_edge,
                    (w + 0.1) / self.position_edge,
                ),
                mode=self.config.interpolate_mode,
                align_corners=False,
            )
            positional.append(
                resized.flatten(2).transpose(1, 2).to(dtype).repeat(1, t, 1)
            )
        return torch.cat(positional, dim=1).squeeze(0)

    def forward(self, pixel_values, grid_thw):
        patches = self._patch_conv(pixel_values)
        pos_embeds = self.compute_pos_embeds(grid_thw, patches.dtype)
        return patches + pos_embeds.unsqueeze(0)

    def forward_with_pos_embeds(
        self, pixel_values: torch.Tensor, pos_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Packed path: patch conv plus precomputed position embeddings."""
        return self._patch_conv(pixel_values) + pos_embeds.unsqueeze(0)


class HYV4VisionPatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, output, merge = (
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
        )
        hidden = config.perceive_intermediate_size or dim * merge**2
        self.extra_mlp = config.projector_type is None
        if config.projector_type == "rmsnorm_mlp_adapter":
            self.norm = RMSNorm(dim, eps=config.rms_norm_eps)
            self.proj = nn.Sequential(
                nn.Conv2d(dim, hidden, merge, stride=merge),
                nn.GELU(),
                nn.Conv2d(hidden, output, 1),
            )
            newline_size = output
        elif config.projector_type is None:
            self.proj = nn.Sequential(
                nn.Conv2d(dim, dim * merge, merge, stride=merge),
                nn.GELU(),
                nn.Conv2d(dim * merge, hidden, 1),
            )
            self.mlp = nn.Linear(hidden, output)
            self.image_sep = nn.Parameter(torch.empty(output))
            newline_size = hidden
        else:
            raise ValueError(
                f"Unsupported HYV4 projector_type: {config.projector_type}"
            )
        if config.use_image_newline_token:
            self.image_newline = nn.Parameter(torch.empty(newline_size))
        if config.cat_extra_token:
            self.image_begin = nn.Parameter(torch.empty(output))
            self.image_end = nn.Parameter(torch.empty(output))
        self.before_rms = (
            RMSNorm(dim, eps=config.rms_norm_eps)
            if self.extra_mlp and config.perceive_pre_norm
            else None
        )
        self.after_rms = (
            RMSNorm(output, eps=config.rms_norm_eps)
            if self.extra_mlp and config.perceive_post_norm
            else None
        )

    def forward(self, x):
        config = self.config
        if not self.extra_mlp:
            x = self.norm(x)
        if self.before_rms is not None:
            x = self.before_rms(x)
        dtype = x.dtype
        x = self.proj(x.permute(0, 3, 1, 2).contiguous())
        if x.shape[0] > 1:
            x = F.avg_pool3d(
                x.permute(1, 0, 2, 3),
                (config.temporal_patch_size, 1, 1),
                stride=(config.temporal_patch_size, 1, 1),
            ).permute(1, 0, 2, 3)
            x = F.avg_pool2d(
                x, config.spatial_patch_size, stride=config.spatial_patch_size
            )
        b, c, h, _ = x.shape
        if config.use_image_newline_token:
            x = torch.cat(
                (
                    x,
                    self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype),
                ),
                dim=-1,
            )
        x = x.flatten(2).transpose(1, 2)
        if self.extra_mlp:
            x = self.mlp(x)
        if config.cat_extra_token:
            x = torch.cat(
                (
                    self.image_begin.reshape(1, 1, -1).expand(b, 1, -1).to(dtype),
                    x,
                    self.image_end.reshape(1, 1, -1).expand(b, 1, -1).to(dtype),
                ),
                dim=1,
            )
        x = x.reshape(-1, x.shape[-1])
        return self.after_rms(x) if self.after_rms is not None else x


class HYV4VisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        quant_config=None,
        prefix="",
        input_norm: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.input_norm = input_norm or IdentityInputNorm()
        self.embeddings = HYV4VisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [
                HYV4VisionBlock(config, quant_config, f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.perceive = HYV4VisionPatchMerger(config)
        self.attn_backend = get_vit_attn_backend(
            head_size=config.hidden_size // config.num_attention_heads,
            dtype=torch.get_default_dtype(),
        )
        use_dp = is_vit_use_data_parallel(config.num_attention_heads)
        self.tp_size = 1 if use_dp else get_tensor_model_parallel_world_size()
        self.rotary_pos_emb = None
        if config.use_2d_rope:
            if config.rope_2d_percent != 1.0:
                raise ValueError("HYV4 vision supports only full 2D RoPE")
            self.rotary_pos_emb = get_rope(
                head_size=config.hidden_size // config.num_attention_heads,
                max_position=self.embeddings.position_edge + 1,
                is_neox_style=True,
                rope_parameters={
                    "rope_theta": config.rope_2d_theta,
                    "partial_rotary_factor": 0.5,
                },
                dtype=torch.float32,
            )

    @property
    def dtype(self):
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.embeddings.patch_embedding.weight.device

    def _rope_cos_sin(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.rotary_pos_emb is not None
        max_grid = max(max(h, w) for _, h, w in grid_thw)
        cos_table, sin_table = self.rotary_pos_emb.get_cos_sin(max_grid)
        indices = [
            torch.stack(
                (torch.arange(h).repeat_interleave(w), torch.arange(w).repeat(h)),
                dim=-1,
            ).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        indices = torch.cat(indices).to(cos_table.device)
        return cos_table[indices].flatten(1), sin_table[indices].flatten(1)

    def _encode(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        """Patch embedding plus ViT layers over the packed patch sequence."""
        config = self.config
        lengths = [h * w for t, h, w in grid_thw for _ in range(t)]
        if sum(lengths) != pixel_values.shape[0] or not lengths or min(lengths) <= 0:
            raise ValueError(
                "HYV4 pixel_values length does not match positive grid_thw"
            )
        hidden = self.embeddings(
            self.input_norm(pixel_values.to(device=self.device), self.dtype), grid_thw
        )
        cu = np.cumsum([0, *lengths], dtype=np.int32)
        max_seq = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu),
            dtype=torch.int32,
        )
        seq_lens = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu, self.device
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend, cu, config.hidden_size, self.tp_size, self.device
        )
        cos = sin = None
        if self.rotary_pos_emb is not None:
            cos, sin = self._rope_cos_sin(grid_thw)
        for layer in self.layers:
            hidden = layer(
                hidden,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seq,
                cos=cos,
                sin=sin,
                sequence_lengths=seq_lens,
            )
        return hidden

    def _iter_item_states(
        self, hidden: torch.Tensor, grid_thw: list[list[int]]
    ) -> Iterable[torch.Tensor]:
        """Per-item ``(frames, h, w, hidden)`` slices with temporal padding."""
        config = self.config
        offset = 0
        for t, h, w in grid_thw:
            states = hidden[:, offset : offset + t * h * w].reshape(t, h, w, -1)
            offset += t * h * w
            if t > 1 and t % config.temporal_patch_size:
                pad = config.temporal_patch_size - t % config.temporal_patch_size
                states = torch.cat((states, states[-1:].expand(pad, -1, -1, -1)), dim=0)
            yield states

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> list[torch.Tensor]:
        hidden = self._encode(pixel_values, grid_thw)
        return [
            self.perceive(states) for states in self._iter_item_states(hidden, grid_thw)
        ]

    def prepare_encoder_cudagraph_metadata(
        self,
        grid_thw: list[list[int]],
        *,
        device: torch.device,
        max_seqlen_override: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Precompute the packed-forward inputs that depend only on grids."""
        grid_thw = [[int(v) for v in grid] for grid in grid_thw]
        metadata: dict[str, torch.Tensor | None] = {}
        metadata["pos_embeds"] = self.embeddings.compute_pos_embeds(
            grid_thw, self.dtype
        ).to(device)
        if self.rotary_pos_emb is not None:
            cos, sin = self._rope_cos_sin(grid_thw)
            metadata["rope_cos"] = cos.to(device)
            metadata["rope_sin"] = sin.to(device)
        else:
            metadata["rope_cos"] = metadata["rope_sin"] = None
        lengths = [h * w for t, h, w in grid_thw for _ in range(t)]
        cu = np.cumsum([0, *lengths], dtype=np.int32)
        max_seqlen = (
            max_seqlen_override
            if max_seqlen_override is not None
            else MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu)
        )
        metadata["max_seqlen"] = torch.tensor(max_seqlen, dtype=torch.int32)
        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend, cu, self.config.hidden_size, self.tp_size, device
        )
        return metadata

    def forward_packed(
        self,
        patches: torch.Tensor,
        pos_embeds: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Static-shape packed ViT forward for encoder CUDA graph capture/replay.

        ``patches`` holds the *embedded* patches (input norm + patch conv
        applied eagerly at replay time): both conv2d kernels (patch embed and
        merger) are cudnn calls whose results are not invariant to input
        shape (measured on GB200), so they cannot run on padded buffers
        without changing results bit-wise. Everything left in the graph —
        pos-embed add, per-layer LayerNorm/GEMM/RoPE/FlashAttn-varlen — is
        row-invariant and verified bitwise-identical to the eager path.

        All grid-dependent host logic must be precomputed via
        :meth:`prepare_encoder_cudagraph_metadata`. Returns the padded packed
        hidden states ``(1, num_patch_slots, hidden_size)``; the per-item
        merger then runs eagerly on the real rows (in
        ``postprocess_encoder_output``).
        """
        hidden = patches.unsqueeze(0) + pos_embeds.unsqueeze(0)
        for layer in self.layers:
            hidden = layer(
                hidden,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cos=rope_cos,
                sin=rope_sin,
                sequence_lengths=None,
            )
        return hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            for projection, shard in (
                ("q_proj", "q"),
                ("k_proj", "k"),
                ("v_proj", "v"),
            ):
                if f".{projection}." in name:
                    name = name.replace(f".{projection}.", ".qkv.")
                    param = params[name]
                    param.weight_loader(param, weight, shard)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded


# Vision-language model and MRoPE positions


class HYV4VLImagePixelInputs(TensorSchema):
    """Flattened image patches and per-image [T, H, W] grids."""

    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class HYV4VLVideoPixelInputs(TensorSchema):
    """Flattened video patches and per-video [T, H, W] grids."""

    pixel_values_videos: Annotated[torch.Tensor, TensorShape("np", "cps")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]


@MULTIMODAL_REGISTRY.register_processor(
    HYV4VLMultiModalProcessor,
    info=HYV4VLProcessingInfo,
    dummy_inputs=HYV4VLDummyInputsBuilder,
)
class HYV4VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsMRoPE,
    SupportsPP,
):
    """HYV4 vision and language composition through the native model registry."""

    supports_mm_device_do_normalize = True

    packed_modules_mapping = {
        **hyv4_packed_modules_mapping(),
        "qkv": ["q_proj", "k_proj", "v_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vit.vit.": "visual.",
            "vit.": "visual.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
        orig_to_new_substr={
            "self_attn.query_layernorm": "self_attn.q_norm",
            "self_attn.key_layernorm": "self_attn.k_norm",
            "mlp.gate.wg": "mlp.gate",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality in ("image", "video"):
            return f"<{modality}>"
        raise ValueError(f"Unsupported HYV4 VL modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.get_multimodal_config()
        enabled = any(
            self.multimodal_config.get_limit_per_prompt(m) for m in ("image", "video")
        )
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = None
            if enabled:
                quant = vllm_config.quant_config
                self.visual = HYV4VisionTransformer(
                    self.config.vision_config,
                    quant_config=quant,
                    prefix=maybe_prefix(prefix, "visual"),
                    input_norm=build_mm_input_norm(vllm_config.model_config),
                )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config.text_config,
                architectures=["HYV4ForCausalLM"],
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> HYV4VLImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(
                [item.reshape(-1, item.shape[-1]) for item in pixel_values]
            )
        elif isinstance(pixel_values, torch.Tensor) and pixel_values.ndim > 0:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        grid = kwargs.pop("image_grid_thw", None)
        if isinstance(grid, list):
            grid = torch.stack(grid)
        if isinstance(grid, torch.Tensor) and grid.ndim > 0:
            grid = grid.reshape(-1, grid.shape[-1])
        config = self.config.vision_config
        return HYV4VLImagePixelInputs(
            pixel_values=pixel_values,
            image_grid_thw=grid,
            resolve_bindings={"cps": config.num_channels * config.patch_size**2},
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> HYV4VLVideoPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values_videos", None)
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(
                [item.reshape(-1, item.shape[-1]) for item in pixel_values]
            )
        elif isinstance(pixel_values, torch.Tensor) and pixel_values.ndim > 0:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        grid = kwargs.pop("video_grid_thw", None)
        if isinstance(grid, list):
            grid = torch.stack(grid)
        if isinstance(grid, torch.Tensor) and grid.ndim > 0:
            grid = grid.reshape(-1, grid.shape[-1])
        config = self.config.vision_config
        return HYV4VLVideoPixelInputs(
            pixel_values_videos=pixel_values,
            video_grid_thw=grid,
            resolve_bindings={"cps": config.num_channels * config.patch_size**2},
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if (
            kwargs.get("image_embeds") is not None
            or kwargs.get("video_embeds") is not None
        ):
            raise ValueError("HYV4 VL does not support precomputed visual embeddings")
        outputs = []
        parsers = {
            "pixel_values": ("image", self._parse_and_validate_image_input),
            "pixel_values_videos": ("video", self._parse_and_validate_video_input),
        }
        for key, value in kwargs.items():
            if key not in parsers or value is None:
                continue
            modality, parse_input = parsers[key]
            if self.visual is None:
                raise ValueError(
                    "HYV4 visual inputs are disabled by limit_mm_per_prompt"
                )
            mm_input = parse_input(**kwargs)
            assert mm_input is not None
            grids = mm_input[f"{modality}_grid_thw"].tolist()
            counts = [
                vision_token_count(g, self.config.vision_config, modality)
                for g in grids
            ]
            embeds = self.visual(mm_input[key], grids)
            for embedding, count in zip(embeds, counts, strict=True):
                if embedding.shape != (count, self.config.text_config.hidden_size):
                    raise ValueError(
                        f"{modality} visual embeddings do not match token grid"
                    )
            outputs.extend(embeds)
        return tuple(outputs)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()

    def get_mm_mapping(self):
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.perceive",
            tower_model=["visual.embeddings", "visual.layers"],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["visual."] if self.visual is None else None,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        tokens = torch.tensor(input_tokens, dtype=torch.long)
        positions = torch.arange(len(input_tokens)).repeat(3, 1)
        if not input_tokens or not mm_features:
            return positions, 0
        data = MultiModalFeatureSpec.gather_kwargs(
            mm_features, {"image_grid_thw", "video_grid_thw"}
        )
        image_grids = [grid.tolist() for grid in data.get("image_grid_thw", [])]
        video_grids = [grid.tolist() for grid in data.get("video_grid_thw", [])]
        config = self.config
        vision = config.vision_config
        extra = int(bool(vision.cat_extra_token))
        starts = torch.where(tokens == config.image_start_token_id)[0].tolist()
        ends = torch.where(tokens == config.image_end_token_id)[0].tolist()
        if len(starts) != len(ends):
            raise ValueError("HYV4 image/frame start and end tokens must be paired")
        start_pos = start_index = mm_index = scan = 0
        image_index = video_index = -1
        inside_video = False
        while mm_index < len(starts):
            marker = starts[mm_index]
            pos = marker + 1 + extra
            while scan < marker:
                if input_tokens[scan] == config.video_start_token_id:
                    inside_video = True
                elif input_tokens[scan] == config.video_end_token_id:
                    inside_video = False
                scan += 1
            is_video = inside_video or (
                image_index + 1 >= len(image_grids)
                and video_index + 1 < len(video_grids)
            )
            if start_pos < pos:
                positions[:, start_pos:pos] = (
                    torch.arange(pos - start_pos) + start_index
                )
                start_index += pos - start_pos
            if is_video:
                video_index += 1
                if video_index >= len(video_grids):
                    raise ValueError("Video placeholders do not match video_grid_thw")
                frames, rows, cols = vision_token_grid(
                    video_grids[video_index], vision, "video"
                )
            else:
                image_index += 1
                if image_index >= len(image_grids):
                    raise ValueError("Image placeholders do not match image_grid_thw")
                frames, rows, cols = vision_token_grid(
                    image_grids[image_index], vision, "image"
                )
            if mm_index + frames > len(starts) or mm_index + frames > len(ends):
                raise ValueError("Missing HYV4 image/frame boundary tokens")
            count = rows * cols
            for frame in range(frames):
                begin = starts[mm_index + frame] + 1 + extra
                end = begin + count
                pad_id = config.video_token_id if is_video else config.image_token_id
                if end + extra != ends[mm_index + frame] or not torch.all(
                    tokens[begin - extra : end + extra] == pad_id
                ):
                    raise ValueError(
                        "HYV4 visual placeholders do not match "
                        "frame boundaries and grid"
                    )
                positions[0, begin:end] = start_index
                positions[1, begin:end] = (
                    torch.arange(rows).repeat_interleave(cols) + start_index
                )
                positions[2, begin:end] = torch.arange(cols).repeat(rows) + start_index
                start_index += max(rows, cols)
                if is_video and frame + 1 < frames:
                    bridge_start = ends[mm_index + frame]
                    bridge_end = starts[mm_index + frame + 1] + 1
                    positions[:, bridge_start:bridge_end] = (
                        torch.arange(bridge_end - bridge_start) + start_index
                    )
                    start_index += bridge_end - bridge_start
            start_pos = ends[mm_index + frames - 1] if is_video else pos + count
            mm_index += frames
        if start_pos < len(input_tokens):
            positions[:, start_pos:] = (
                torch.arange(len(input_tokens) - start_pos) + start_index
            )
        if image_index + 1 != len(image_grids) or video_index + 1 != len(video_grids):
            raise ValueError("Unused HYV4 media grids after position construction")
        return positions, int(positions.max()) + 1 - len(input_tokens)

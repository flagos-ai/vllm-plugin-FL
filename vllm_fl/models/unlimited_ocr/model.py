# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unlimited-OCR with vLLM's DeepSeek-OCR vision tower and DeepSeek-V2 MHA."""

import math
from collections.abc import Mapping, Sequence
from types import MethodType

import torch
from PIL import Image, ImageOps
from transformers import PreTrainedTokenizerBase

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.deepencoder import (
    RelPosAttention,
    add_decomposed_rel_pos,
)
from vllm.model_executor.models.deepseek_ocr import (
    DeepseekOCRDummyInputsBuilder,
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
)
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import MultiModalProcessingInfo
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    DeepseekOCRProcessor,
    dynamic_preprocess,
)

from vllm_fl.dispatch.ops import scaled_dot_product_attention
from vllm_fl.models.unlimited_ocr import UnlimitedOCRConfig

logger = init_logger(__name__)
_UNLIMITED_OCR_MAX_CROPS = 32


def image_token_count(
    base_size: int, image_size: int, width_tiles: int, height_tiles: int
) -> int:
    patch_size = 16
    downsample_ratio = 4
    base_grid = math.ceil((base_size // patch_size) / downsample_ratio)
    local_grid = math.ceil((image_size // patch_size) / downsample_ratio)
    # Each row adds a newline and the global view adds one separator.
    count = base_grid * (base_grid + 1) + 1
    if width_tiles > 1 or height_tiles > 1:
        count += height_tiles * local_grid * (width_tiles * local_grid + 1)
    return count


class UnlimitedOCRProcessor(DeepseekOCRProcessor):
    """DeepSeek-OCR vision layout with Unlimited-OCR's crop policy."""

    # Transformers' ProcessorMixin retains only named positional-or-keyword
    # parameters when constructing the processor from checkpoint and overrides.
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        image_size: int = IMAGE_SIZE,
        base_size: int = BASE_SIZE,
        strategy: str = "v1",
        max_crops: int = _UNLIMITED_OCR_MAX_CROPS,
        **kwargs,
    ) -> None:
        if strategy != "v1":
            raise ValueError("Unlimited-OCR requires the v1 image token layout")
        super().__init__(
            tokenizer=tokenizer,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
            image_mean=image_mean,
            image_std=image_std,
            normalize=normalize,
            image_token=image_token,
            pad_token=pad_token,
            add_special_token=add_special_token,
            sft_format=sft_format,
            mask_prompt=mask_prompt,
            ignore_id=ignore_id,
            image_size=image_size,
            base_size=base_size,
            strategy=strategy,
            **kwargs,
        )
        self.max_crops = max_crops

    def tokenize_with_images(
        self,
        conversation: str,
        images: list[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        if len(images) > 1 and cropping:
            logger.warning_once(
                "Unlimited-OCR is using the global view for a multi-image prompt."
            )
            cropping = False

        assert conversation.count(self.image_token) == len(images)
        text_splits = conversation.split(self.image_token)
        token_ids = [self.bos_id] if bos else []
        image_mask = [False] if bos else []
        global_views = []
        local_views = []
        crop_grids = []
        num_image_tokens = []
        image_shapes = []

        for text, image in zip(text_splits, images):
            text_ids = self.encode(text, bos=False, eos=False)
            token_ids.extend(text_ids)
            image_mask.extend([False] * len(text_ids))
            image_shapes.append(image.size)

            tiles = []
            width_tiles = height_tiles = 1
            if cropping and (
                image.width > self.image_size or image.height > self.image_size
            ):
                tiles, (width_tiles, height_tiles) = dynamic_preprocess(
                    image, image_size=self.image_size, max_num=self.max_crops
                )
            if not cropping and self.image_size <= 640:
                # Tiny/Small resize to a square. Base/Large preserve aspect ratio.
                image = image.resize((self.image_size, self.image_size))

            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            global_views.append(self.image_transform(global_view))
            crop_grids.append([width_tiles, height_tiles])
            if width_tiles > 1 or height_tiles > 1:
                local_views.extend(self.image_transform(tile) for tile in tiles)

            count = image_token_count(
                self.base_size, self.image_size, width_tiles, height_tiles
            )
            num_image_tokens.append(count)
            token_ids.extend([self.image_token_id] * count)
            image_mask.extend([True] * count)

        tail_ids = self.encode(text_splits[-1], bos=False, eos=eos)
        token_ids.extend(tail_ids)
        image_mask.extend([False] * len(tail_ids))
        # Match the inherited processor contract: the prompt excludes its
        # terminal EOS. Inference generates the terminal EOS itself.
        assert token_ids[-1] == self.eos_id
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0)
        input_ids[input_ids < 0] = self.pad_id
        return (
            input_ids,
            torch.stack(global_views)
            if global_views
            else torch.zeros((0, 3, self.base_size, self.base_size)),
            torch.stack(local_views)
            if local_views
            else torch.zeros((0, 3, self.image_size, self.image_size)),
            torch.tensor(image_mask[:-1], dtype=torch.bool),
            torch.tensor(crop_grids, dtype=torch.long).reshape(-1, 2),
            num_image_tokens,
            image_shapes,
        )


class UnlimitedOCRProcessingInfo(DeepseekOCRProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(UnlimitedOCRConfig)

    def get_hf_processor(self, **kwargs: object):
        defaults = dict(
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
            crop_mode=CROP_MODE,
            strategy="v1",
            max_crops=_UNLIMITED_OCR_MAX_CROPS,
        )
        return self.ctx.get_hf_processor(
            UnlimitedOCRProcessor,
            **{**defaults, **self.ctx.get_merged_mm_kwargs(kwargs)},
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()
        cropping = self.ctx.get_merged_mm_kwargs({}).get("crop_mode", CROP_MODE)
        height = processor.image_size
        if cropping:
            # A vertical grid maximizes the number of row separators.
            height *= max(1, processor.max_crops)
        return ImageSize(width=processor.image_size, height=height)


class UnlimitedOCRMultiModalProcessor(DeepseekOCRMultiModalProcessor):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            if isinstance(images, ImageEmbeddingItems):
                count = images.get_feature_size(item_idx)
            else:
                grid = out_mm_kwargs["image"][item_idx]["images_spatial_crop"].data
                width_tiles, height_tiles = grid.tolist()
                count = image_token_count(
                    hf_processor.base_size,
                    hf_processor.image_size,
                    width_tiles,
                    height_tiles,
                )
            return [image_token_id] * count

        return [
            PromptReplacement(
                modality="image", target=[image_token_id], replacement=get_replacement
            )
        ]

    def _cached_apply_hf_processor(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        # Cropping depends on the total number of images. Per-item cache keys
        # only preserve that distinction for the single-image case.
        if inputs.mm_data_items.get_count("image", strict=False) > 1:
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)


def ocr_sam_forward(self: RelPosAttention, x: torch.Tensor) -> torch.Tensor:
    # Keep the native SAM projection and relative-position geometry.
    batch, height, width, _ = x.shape
    heads = self.num_heads
    tokens = height * width
    qkv = self.qkv(x).reshape(batch, tokens, 3, heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, batch * heads, tokens, -1).unbind(0)
    attn_bias = None
    if self.use_rel_pos:
        spatial_size = (height, width)
        rel_h, rel_w = add_decomposed_rel_pos(
            q, self.rel_pos_h, self.rel_pos_w, spatial_size, spatial_size
        )
        attn_bias = (rel_h + rel_w).reshape(batch, heads, tokens, tokens)

    output = scaled_dot_product_attention(
        q.view(batch, heads, tokens, -1),
        k.view(batch, heads, tokens, -1),
        v.view(batch, heads, tokens, -1),
        attn_bias,
    )
    output = (
        output.view(batch, heads, height, width, -1)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch, height, width, -1)
    )
    return self.proj(output)


def install_ocr_sam_attention(encoder: torch.nn.Module) -> None:
    """Bind before loading weights and preserve the encoder's module structure."""
    blocks = [
        layer for layer in encoder.modules() if isinstance(layer, RelPosAttention)
    ]
    if not blocks:
        raise RuntimeError("Unlimited-OCR SAM attention layers were not found")
    for layer in blocks:
        layer.forward = MethodType(ocr_sam_forward, layer)


@MULTIMODAL_REGISTRY.register_processor(
    UnlimitedOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not isinstance(self.sam_model, StageMissingLayer):
            install_ocr_sam_attention(self.sam_model)

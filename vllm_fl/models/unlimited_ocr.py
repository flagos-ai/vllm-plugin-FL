# Copyright (c) 2026 BAAI. All rights reserved.
"""Reuse native Unlimited-OCR with reference multi-page base preprocessing."""

from dataclasses import replace

from PIL import ImageOps

from vllm.model_executor.models.deepseek_ocr import DeepseekOCRDummyInputsBuilder
from vllm.model_executor.models.unlimited_ocr import (
    UnlimitedOCRForCausalLM as NativeUnlimitedOCRForCausalLM,
    UnlimitedOCRMultiModalProcessor as NativeUnlimitedOCRMultiModalProcessor,
    UnlimitedOCRProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItem, MultiModalKwargsItems
from vllm.transformers_utils.processors.deepseek_ocr import BASE_SIZE


class UnlimitedOCRMultiModalProcessor(NativeUnlimitedOCRMultiModalProcessor):
    def _cached_apply_hf_processor(self, inputs, timing_ctx):
        if inputs.mm_data_items.get_count("image", strict=False) <= 1:
            return super()._cached_apply_hf_processor(inputs, timing_ctx)

        local_size = self.info.get_hf_processor(
            **inputs.hf_processor_mm_kwargs
        ).image_size
        kwargs = dict(inputs.hf_processor_mm_kwargs)
        kwargs.setdefault("image_size", kwargs.get("base_size", BASE_SIZE))
        kwargs["crop_mode"] = False
        # Include the effective base/crop mode in per-image hashes before
        # processing, so single-page crops cannot alias multi-page views.
        inputs = replace(inputs, hf_processor_mm_kwargs=kwargs)
        result = super()._cached_apply_hf_processor(inputs, timing_ctx)

        # Empty local-crop fields must retain the local resolution, not the
        # larger global-view size. Otherwise batching pads real local crops
        # from sibling requests and changes their visual-token geometry.
        image_items = []
        for item in result.kwargs["image"]:
            crops = item.get("images_crop")
            if crops is not None and crops.data is not None and not crops.data.numel():
                item = MultiModalKwargsItem(item)
                shape = (*crops.data.shape[:-2], local_size, local_size)
                item["images_crop"] = replace(crops, data=crops.data.new_empty(shape))
            image_items.append(item)
        return result._replace(
            kwargs=MultiModalKwargsItems({**result.kwargs, "image": image_items})
        )

    def _call_hf_processor(self, hf_data, hf_kwargs):
        images = hf_data.get("images", ())
        if len(images) > 1:
            processor = self.info.get_hf_processor(**hf_kwargs)
            size = processor.image_size
            background = tuple(int(v * 255) for v in processor.image_transform.mean)
            # The native no-crop path resizes to a square. Pad first to retain
            # the page's aspect ratio as the reference infer_multi path does.
            hf_data = {
                **hf_data,
                "images": [
                    ImageOps.pad(image, (size, size), color=background)
                    for image in images
                ],
            }
        return super()._call_hf_processor(hf_data, hf_kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    UnlimitedOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(NativeUnlimitedOCRForCausalLM):
    """Native model and weight loader with corrected multi-page processing."""

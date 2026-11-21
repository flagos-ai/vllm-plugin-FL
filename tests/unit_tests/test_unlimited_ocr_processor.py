# Copyright (c) 2026 BAAI. All rights reserved.
"""Reference multi-page geometry and cache identities for Unlimited-OCR."""

import json

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.utils import group_and_batch_mm_kwargs

from vllm_fl.models.unlimited_ocr import UnlimitedOCRForCausalLM


@pytest.fixture(scope="module")
def processor(tmp_path_factory):
    path = tmp_path_factory.mktemp("unlimited-ocr-processor")
    config = {
        "model_type": "unlimited_ocr",
        "architectures": ["UnlimitedOCRForCausalLM"],
        "candidate_resolutions": [[1024, 1024]],
        "language_config": {
            "hidden_size": 1280,
            "num_attention_heads": 10,
            "num_key_value_heads": 10,
            "num_hidden_layers": 12,
            "max_position_embeddings": 32768,
            "use_mla": False,
            "q_lora_rank": None,
            "kv_lora_rank": None,
            "qk_nope_head_dim": 0,
            "qk_rope_head_dim": 0,
            "v_head_dim": 128,
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    backend = Tokenizer(
        WordLevel(
            {"<s>": 0, "</s>": 1, "<pad>": 2, "<image>": 3, "<unk>": 4},
            unk_token="<unk>",
        )
    )
    backend.post_processor = TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 0)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=["<image>"],
    )
    tokenizer.save_pretrained(path)
    cfg = ModelConfig(
        model=str(path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=0,
    )
    context = InputProcessingContext(model_config=cfg, tokenizer=tokenizer)
    return UnlimitedOCRForCausalLM._processor_factory.build_processor(context)


def _apply(processor, images, kwargs=None):
    text = "<image>" * len(images) + "document"
    ids = processor.info.get_tokenizer().encode(text, add_special_tokens=True)
    inputs = ProcessorInputs(
        prompt=ids,
        mm_data_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={} if kwargs is None else kwargs,
    )
    return processor.apply(inputs, TimingContext(enabled=False))


def test_single_and_multiple_pages_have_distinct_encoder_hashes(processor):
    first = Image.new("RGB", (2048, 2048), "red")
    second = Image.new("RGB", (2048, 2048), "blue")
    single = _apply(processor, [first])
    kwargs = {}
    double = _apply(processor, [first, second], kwargs)
    repeated = _apply(processor, [first, second], kwargs)
    assert kwargs == {}
    assert single["mm_hashes"]["image"][0] != double["mm_hashes"]["image"][0]
    assert double["mm_hashes"] == repeated["mm_hashes"]
    assert [p.length for p in single["mm_placeholders"]["image"]] == [1913]
    assert [p.length for p in double["mm_placeholders"]["image"]] == [273, 273]
    assert first.size == second.size == (2048, 2048)


def test_multiple_pages_keep_aspect_ratio_at_base_resolution(processor):
    # Known solid-color regions give an independent pixel oracle after padding.
    landscape = Image.new("RGB", (2048, 1024), "red")
    portrait = Image.new("RGB", (1024, 2048), "blue")
    result = _apply(processor, [landscape, portrait])
    data = result["mm_kwargs"].get_data()
    pixels = data["pixel_values"]
    assert pixels.shape == (2, 3, 1024, 1024)
    expected = torch.full_like(pixels, 127 / 255 * 2 - 1)
    expected[0, :, 256:768] = torch.tensor([1, -1, -1])[:, None, None]
    expected[1, :, :, 256:768] = torch.tensor([-1, -1, 1])[:, None, None]
    torch.testing.assert_close(pixels, expected, rtol=0, atol=1e-6)
    assert data["images_spatial_crop"].tolist() == [[1, 1], [1, 1]]
    assert landscape.size == (2048, 1024) and portrait.size == (1024, 2048)


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (1, 2, 0), (0, 1, 0, 2)])
def test_mixed_page_batch_preserves_local_crops(processor, order):
    page = Image.new("RGB", (2048, 2048), "red")
    page.paste("blue", (1024, 0, 2048, 1024))
    page.paste("green", (0, 1024, 1024, 2048))
    page.paste("white", (1024, 1024, 2048, 2048))
    single = _apply(processor, [page])["mm_kwargs"]
    double = _apply(processor, [page, Image.new("RGB", (2048, 2048), "blue")])[
        "mm_kwargs"
    ]
    items = [*single["image"], *double["image"]]
    selected = [items[i] for i in order]
    batches = list(group_and_batch_mm_kwargs([("image", item) for item in selected]))
    assert len(batches) == 1
    modality, count, data = batches[0]
    assert modality == "image" and count == len(order)
    expected_crops = torch.cat(
        [
            item["images_crop"].data
            for item in selected
            if item["images_crop"].data.numel()
        ]
    )
    assert data["images_crop"].shape == (16 * order.count(0), 3, 640, 640)
    torch.testing.assert_close(data["images_crop"], expected_crops, rtol=0, atol=0)
    torch.testing.assert_close(
        data["pixel_values"],
        torch.stack([item["pixel_values"].data for item in selected]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("local_size", [512, 640])
def test_empty_crops_keep_requested_local_resolution(processor, local_size):
    page = Image.new("RGB", (2048, 2048), "red")
    kwargs = {"image_size": local_size}
    result = _apply(processor, [page, page], kwargs)
    data = result["mm_kwargs"].get_data()
    assert data["images_crop"].shape == (0, 3, local_size, local_size)
    assert data["pixel_values"].shape == (2, 3, 1024, 1024)
    assert kwargs == {"image_size": local_size}

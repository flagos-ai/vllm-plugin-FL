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

# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
)

from vllm_fl.models import minicpmv4_7

CANVAS_CONFIG = SimpleNamespace(
    mrope_mode="canvas",
    downsample_mode="16x",
    image_token_id=100,
    video_token_id=101,
    image_start_id=10,
    image_end_id=11,
    slice_start_id=12,
    slice_end_id=13,
    newline_id=14,
)


CANVAS_CASES = {
    "single_image": {
        "input_ids": [1, 10, 100, 100, 100, 100, 11, 2],
        "grids": [[8, 8]],
        "positions": [
            [0, 1, 1, 1, 1, 1, 1, 4],
            [0, 0, 1, 1, 2, 2, 3, 4],
            [0, 0, 1, 2, 1, 2, 3, 4],
        ],
        "delta": -3,
    },
    "single_image_with_trailing_newline": {
        "input_ids": [1, 10, 100, 100, 100, 100, 11, 14, 2],
        "grids": [[8, 8]],
        "positions": [
            [0, 1, 1, 1, 1, 1, 1, 1, 4],
            [0, 0, 1, 1, 2, 2, 3, 1, 4],
            [0, 0, 1, 2, 1, 2, 3, 1, 4],
        ],
        "delta": -4,
    },
    "sliced_image": {
        "input_ids": [
            1,
            10,
            100,
            100,
            100,
            100,
            11,
            12,
            100,
            13,
            12,
            100,
            13,
            14,
            12,
            100,
            13,
            12,
            100,
            13,
            2,
        ],
        "grids": [[8, 8], [4, 4], [4, 4], [4, 4], [4, 4]],
        "positions": [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
            [0, 0, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4],
            [0, 0, 1, 2, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 2, 2, 2, 4],
        ],
        "delta": -16,
    },
    "two_images": {
        "input_ids": [1, 10, 100, 100, 100, 100, 11, 2, 10, 100, 100, 100, 100, 11, 3],
        "grids": [[8, 8], [8, 8]],
        "positions": [
            [0, 1, 1, 1, 1, 1, 1, 4, 5, 5, 5, 5, 5, 5, 8],
            [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8],
            [0, 0, 1, 2, 1, 2, 3, 4, 4, 5, 6, 5, 6, 7, 8],
        ],
        "delta": -6,
    },
    "two_video_frames": {
        "input_ids": [1, 10, 101, 101, 101, 101, 11, 10, 101, 101, 101, 101, 11, 2],
        "video_grids": [[8, 8], [8, 8]],
        "positions": [
            [0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7],
            [0, 0, 1, 2, 1, 2, 3, 3, 4, 5, 4, 5, 6, 7],
        ],
        "delta": -6,
    },
    "image_then_video": {
        "input_ids": [
            1,
            10,
            100,
            100,
            100,
            100,
            11,
            2,
            10,
            101,
            101,
            101,
            101,
            11,
            10,
            101,
            101,
            101,
            101,
            11,
            3,
        ],
        "grids": [[8, 8]],
        "video_grids": [[8, 8], [8, 8]],
        "positions": [
            [0, 1, 1, 1, 1, 1, 1, 4, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 11],
            [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11],
            [0, 0, 1, 2, 1, 2, 3, 4, 4, 5, 6, 5, 6, 7, 7, 8, 9, 8, 9, 10, 11],
        ],
        "delta": -9,
    },
}


def test_selects_dense_qwen3_5_text_tower():
    text_config = SimpleNamespace(model_type="qwen3_5_text")
    assert minicpmv4_7._select_language_model_cls(text_config) is Qwen3_5ForCausalLM


def test_selects_moe_qwen3_5_text_tower_by_model_type():
    text_config = SimpleNamespace(model_type="qwen3_5_moe_text", num_experts=256)
    assert minicpmv4_7._select_language_model_cls(text_config) is Qwen3_5MoeForCausalLM


def test_selects_moe_qwen3_5_text_tower_by_expert_count():
    text_config = SimpleNamespace(model_type="qwen3_5_text", num_experts=8)
    assert minicpmv4_7._select_language_model_cls(text_config) is Qwen3_5MoeForCausalLM


def test_processing_info_reports_version_4_7():
    info = object.__new__(minicpmv4_7.MiniCPMV4_7ProcessingInfo)
    assert info.get_model_version() == (4, 7)


def test_registers_minicpmv4_7_architecture(monkeypatch):
    registered = {}

    monkeypatch.setattr(
        minicpmv4_7.ModelRegistry,
        "register_model",
        lambda architecture, model: registered.__setitem__(architecture, model),
    )
    minicpmv4_7.register_minicpmv4_7()

    assert registered == {
        "MiniCPMV4_7ForConditionalGeneration": (
            minicpmv4_7.MiniCPMV4_7ForConditionalGeneration
        )
    }


def test_video_doc_compat_preserves_existing_symbol(monkeypatch):
    import transformers.video_processing_utils as video_processing_utils

    sentinel = object()
    monkeypatch.setattr(
        video_processing_utils,
        "BASE_VIDEO_PROCESSOR_DOCSTRING",
        sentinel,
        raising=False,
    )

    minicpmv4_7._install_transformers_video_doc_compat()

    assert video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING is sentinel


def test_video_doc_compat_backfills_missing_symbol(monkeypatch):
    import transformers.video_processing_utils as video_processing_utils

    monkeypatch.delattr(
        video_processing_utils,
        "BASE_VIDEO_PROCESSOR_DOCSTRING",
        raising=False,
    )

    minicpmv4_7._install_transformers_video_doc_compat()

    assert video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING == ""


def test_constructor_restores_model_type_when_text_tower_raises(monkeypatch):
    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class RaisingLanguageModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("text tower construction failed")

    config = SimpleNamespace(
        model_type="minicpmv4_7",
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            num_experts=256,
            hidden_size=128,
        ),
        vision_config=SimpleNamespace(hidden_size=64),
        drop_vision_last_layer=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            multimodal_config=SimpleNamespace(mm_encoder_tp_mode="weights"),
        ),
        quant_config=None,
    )

    monkeypatch.setattr(minicpmv4_7, "Idefics2VisionTransformer", DummyModule)
    monkeypatch.setattr(
        minicpmv4_7,
        "MiniCPMV4_6ViTWindowAttentionMerger",
        DummyModule,
    )
    monkeypatch.setattr(minicpmv4_7, "MiniCPMV4_6Merger", DummyModule)
    monkeypatch.setattr(
        minicpmv4_7,
        "_select_language_model_cls",
        lambda text_config: RaisingLanguageModel,
    )
    monkeypatch.setattr(
        minicpmv4_7.MiniCPMV4_7ForConditionalGeneration,
        "_mark_tower_model",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        minicpmv4_7.MiniCPMV4_7ForConditionalGeneration,
        "_mark_language_model",
        lambda *args, **kwargs: nullcontext(),
    )

    with pytest.raises(RuntimeError, match="text tower construction failed"):
        minicpmv4_7.MiniCPMV4_7ForConditionalGeneration(vllm_config=vllm_config)

    assert config.model_type == "minicpmv4_7"


@pytest.mark.parametrize("case", CANVAS_CASES.values(), ids=CANVAS_CASES.keys())
def test_canvas_mrope_matches_transformers_golden_layouts(case):
    input_ids = torch.tensor(case["input_ids"], dtype=torch.long)
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == CANVAS_CONFIG.image_token_id] = 1
    mm_token_type_ids[input_ids == CANVAS_CONFIG.video_token_id] = 2
    grids = case.get("grids")
    video_grids = case.get("video_grids")

    positions, delta = minicpmv4_7._get_canvas_mrope_positions(
        CANVAS_CONFIG,
        input_ids,
        mm_token_type_ids,
        None if grids is None else torch.tensor(grids),
        None if video_grids is None else torch.tensor(video_grids),
    )

    assert positions.tolist() == case["positions"]
    assert delta == case["delta"]


def test_vllm_mrope_entrypoint_gathers_feature_target_sizes():
    input_ids = CANVAS_CASES["single_image"]["input_ids"]
    feature = SimpleNamespace(
        modality="image",
        mm_position=SimpleNamespace(offset=2),
        data={
            "tgt_sizes": SimpleNamespace(data=torch.tensor([[8, 8]], dtype=torch.long))
        },
    )
    model = SimpleNamespace(config=CANVAS_CONFIG)

    positions, delta = (
        minicpmv4_7.MiniCPMV4_7ForConditionalGeneration.get_mrope_input_positions(
            model,
            input_ids,
            [feature],
        )
    )

    assert positions.tolist() == CANVAS_CASES["single_image"]["positions"]
    assert delta == -3


def test_text_only_mrope_stays_sequential():
    model = SimpleNamespace(config=CANVAS_CONFIG)
    positions, delta = (
        minicpmv4_7.MiniCPMV4_7ForConditionalGeneration.get_mrope_input_positions(
            model,
            [4, 5, 6, 7],
            [],
        )
    )

    assert positions.tolist() == [[0, 1, 2, 3]] * 3
    assert delta == 0


def test_resolve_canvas_token_ids_from_pre_upstream_tokenizer():
    config = SimpleNamespace(
        image_start_id=None,
        image_end_id=None,
        slice_start_id=None,
        slice_end_id=None,
        newline_id=None,
    )
    ids = {
        "<image>": 10,
        "</image>": 11,
        "<slice>": 12,
        "</slice>": 13,
    }
    tokenizer = SimpleNamespace(
        image_start_token="<image>",
        image_end_token="</image>",
        slice_start_token="<slice>",
        slice_end_token="</slice>",
        convert_tokens_to_ids=ids.__getitem__,
        encode=lambda text, add_special_tokens=False: [14],
    )

    minicpmv4_7._resolve_canvas_token_ids(config, tokenizer)

    assert (
        config.image_start_id,
        config.image_end_id,
        config.slice_start_id,
        config.slice_end_id,
        config.newline_id,
    ) == (10, 11, 12, 13, 14)

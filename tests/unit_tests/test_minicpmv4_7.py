# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from torch import nn

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
)

from vllm_fl.models import minicpmv4_7


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

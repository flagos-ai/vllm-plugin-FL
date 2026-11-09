# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.minicpmo import MiniCPMOProcessingInfo
from vllm.multimodal.encoder_budget import MultiModalBudget

from vllm_fl.patches.minicpmo_audio import patch_minicpmo_audio_budget


@pytest.fixture
def unpatched_info(monkeypatch):
    """Exercise the actual upstream inheritance without loading a checkpoint."""
    cls = MiniCPMOProcessingInfo
    name = "get_mm_max_tokens_per_item"
    original = cls.__dict__.get(name)
    if original is not None:
        delattr(cls, name)
    info = cls(SimpleNamespace())
    monkeypatch.setattr(info, "get_model_version", lambda: (4, 5))
    monkeypatch.setattr(info, "get_max_image_tokens", lambda: 1024)
    monkeypatch.setattr(info, "get_max_video_tokens", lambda *_args: 1536)
    monkeypatch.setattr(
        info, "get_hf_config", lambda: SimpleNamespace(audio_pool_step=5)
    )
    yield info
    if original is not None:
        setattr(cls, name, original)
    elif name in cls.__dict__:
        delattr(cls, name)


def _budget(info, limits):
    model_config = SimpleNamespace(
        max_model_len=2048,
        get_multimodal_config=lambda: SimpleNamespace(enable_mm_embeds=False),
    )
    info.ctx = SimpleNamespace(model_config=model_config)
    info.allowed_mm_limits = limits
    scheduler_config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        max_num_encoder_input_tokens=2048,
        encoder_cache_size=2048,
        disable_chunked_mm_input=False,
        enable_chunked_prefill=True,
    )
    registry = SimpleNamespace(
        create_processor=lambda _config: SimpleNamespace(info=info)
    )
    return MultiModalBudget(
        SimpleNamespace(model_config=model_config, scheduler_config=scheduler_config),
        registry,
        enable_cache=False,
    )


@pytest.mark.parametrize("pool_step,audio_tokens", [(2, 750), (5, 300)])
def test_audio_only_encoder_budget(
    unpatched_info, monkeypatch, pool_step, audio_tokens
):
    info = unpatched_info
    monkeypatch.setattr(
        info, "get_hf_config", lambda: SimpleNamespace(audio_pool_step=pool_step)
    )
    limits = {"image": 0, "video": 0, "audio": 2}
    assert _budget(info, limits).get_encoder_budget() == 0

    assert patch_minicpmo_audio_budget() is True
    budget = _budget(info, limits)
    assert budget.mm_max_toks_per_item == {"audio": audio_tokens}
    assert budget.get_encoder_budget() == 2048
    assert budget.mm_max_items_per_prompt == {"audio": 2}
    patched = MiniCPMOProcessingInfo.get_mm_max_tokens_per_item
    assert patch_minicpmo_audio_budget() is False
    assert MiniCPMOProcessingInfo.get_mm_max_tokens_per_item is patched


def test_preserves_image_video_budgets(unpatched_info):
    info = unpatched_info
    limits = {"image": 1, "video": 1, "audio": 2}
    before = dict(_budget(info, limits).mm_max_toks_per_item)
    assert patch_minicpmo_audio_budget() is True
    assert _budget(info, limits).mm_max_toks_per_item == {**before, "audio": 300}


@pytest.mark.parametrize("image_limit", [0, 1])
def test_disabled_audio_stays_disabled(unpatched_info, image_limit):
    assert patch_minicpmo_audio_budget() is True
    budget = _budget(unpatched_info, {"image": image_limit, "video": 0, "audio": 0})
    assert "audio" not in budget.mm_max_toks_per_item
    assert budget.get_encoder_budget() == (2048 if image_limit else 0)


def test_preserves_native_override(unpatched_info):
    def native(self, seq_len, mm_counts):
        return {"audio": 123}

    MiniCPMOProcessingInfo.get_mm_max_tokens_per_item = native
    assert patch_minicpmo_audio_budget() is False
    assert MiniCPMOProcessingInfo.get_mm_max_tokens_per_item is native

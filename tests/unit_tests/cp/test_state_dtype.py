"""VLLM_FL_GDN_STATE_BF16: default off, BF16 storage only when opted in."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

from vllm_fl.cp import state_dtype


@pytest.fixture
def restore_hook():
    original = Qwen3_5ForConditionalGenerationConfig.__dict__[
        "verify_and_update_config"
    ]
    yield
    Qwen3_5ForConditionalGenerationConfig.verify_and_update_config = original


def make_config(requested="auto", dtype=torch.bfloat16, spec=None):
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype=requested),
        model_config=SimpleNamespace(
            dtype=dtype, hf_text_config=SimpleNamespace(mamba_ssm_dtype="float32")
        ),
        speculative_config=spec,
    )


def test_default_off(monkeypatch):
    monkeypatch.delenv(state_dtype.ENV, raising=False)
    assert not state_dtype.enabled()


def test_rejects_bad_value(monkeypatch):
    monkeypatch.setenv(state_dtype.ENV, "true")
    with pytest.raises(ValueError):
        state_dtype.enabled()


def test_unpatched_keeps_hf_float32(restore_hook):
    config = make_config()
    Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(config)
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


def test_sets_bf16_and_is_idempotent(restore_hook):
    state_dtype.apply_gdn_state_bf16()
    hook = Qwen3_5ForConditionalGenerationConfig.verify_and_update_config
    state_dtype.apply_gdn_state_bf16()
    assert Qwen3_5ForConditionalGenerationConfig.verify_and_update_config is hook
    config = make_config()
    hook(config)
    assert config.cache_config.mamba_ssm_cache_dtype == "bfloat16"


@pytest.mark.parametrize(
    "kwargs",
    [dict(requested="float32"), dict(dtype=torch.float16), dict(spec=object())],
)
def test_rejects_conflicts(restore_hook, kwargs):
    state_dtype.apply_gdn_state_bf16()
    with pytest.raises(ValueError):
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(
            make_config(**kwargs)
        )

"""Keep Kunlunxin hybrid-cache detection across shared worker merges."""

from types import SimpleNamespace

import pytest

from vllm_fl.worker.worker import _is_hybrid_kv_model


@pytest.mark.parametrize("mode", ["align", "all"])
def test_mamba_cache_mode_is_hybrid(mode):
    config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode=mode))
    assert _is_hybrid_kv_model(config)


def test_model_hybrid_flag():
    config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True))
    assert _is_hybrid_kv_model(config)


@pytest.mark.parametrize(
    "layer_types, expected",
    [
        ([], False),
        (["full_attention"], False),
        (["full_attention", "linear_attention"], True),
    ],
)
def test_layer_type_fallback(layer_types, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=layer_types)
        )
    )
    assert _is_hybrid_kv_model(config) is expected


def test_unresolved_hybrid_property_falls_back():
    class ModelConfig:
        hf_text_config = SimpleNamespace(layer_types=["linear_attention"])

        @property
        def is_hybrid(self):
            raise RuntimeError("not initialized")

    assert _is_hybrid_kv_model(SimpleNamespace(model_config=ModelConfig()))


def test_missing_optional_configuration_is_not_hybrid():
    assert not _is_hybrid_kv_model(SimpleNamespace())

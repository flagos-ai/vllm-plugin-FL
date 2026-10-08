from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
    _requires_flash_kv_cache,
)


def _config(layer_types):
    text_config = SimpleNamespace(layer_types=layer_types)
    model_config = SimpleNamespace(hf_text_config=text_config)
    return SimpleNamespace(model_config=model_config)


@pytest.mark.parametrize(
    ("layer_types", "expected"),
    [
        (None, False),
        ([], False),
        (["full_attention"] * 4, False),
        (["linear_attention"] * 4, False),
        (["linear_attention", "full_attention"], True),
    ],
)
def test_flash_kv_cache_is_selected_from_attention_layout(
    layer_types,
    expected,
):
    assert _requires_flash_kv_cache(_config(layer_types)) is expected

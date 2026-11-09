# Copyright 2026 FlagOS Contributors

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "vendor,switch,cache,omitted",
    [
        ("metax", "1", "auto", True),
        ("nvidia", "1", "auto", False),
        ("metax", "0", "auto", False),
        ("metax", "1", "fp8", False),
    ],
)
def test_release_attention_descales(monkeypatch, vendor, switch, cache, omitted):
    from vllm_fl.dispatch.backends.flaggems.impl import attention_utils

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", switch)
    monkeypatch.setattr(
        attention_utils, "current_platform", SimpleNamespace(vendor_name=vendor)
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    scales = attention_utils.attention_descales(layer, cache, (2, 3))
    if omitted:
        assert scales == (None, None, None)
    else:
        for actual, source in zip(
            scales, (layer._q_scale, layer._k_scale, layer._v_scale)
        ):
            assert actual.shape == (2, 3)
            assert actual.data_ptr() == source.data_ptr()

# Copyright (c) 2026 BAAI. All rights reserved.

import os
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

if os.environ.get("FL_TEST_PLATFORM") != "enflame":
    pytest.skip("GCU attention contracts require Enflame", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl import attention


def test_supported_sizes_without_config_preserve_gcu_block_limits(monkeypatch):
    monkeypatch.setattr(attention, "get_current_vllm_config_or_none", lambda: None)
    assert attention.AttentionGCUBackend.get_supported_kernel_block_sizes() == [
        16,
        32,
        64,
    ]


@pytest.mark.skipif(
    Version(version("vllm")) < Version("0.32.0"), reason="0.32 spec API"
)
def test_upstream_cache_spec_customization_without_config(monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import customize_attention_spec

    monkeypatch.setattr(attention, "get_current_vllm_config_or_none", lambda: None)
    original = FullAttentionSpec(
        block_size=32, num_kv_heads=2, head_size=16, dtype=torch.bfloat16
    )
    customized = customize_attention_spec(attention.AttentionGCUBackend, original)
    assert customized.block_stride_alignment.base == 64
    assert customized.block_size == original.block_size
    assert customized.dtype == original.dtype


@pytest.mark.skipif(
    Version(version("vllm")) >= Version("0.32.0"),
    reason="legacy pre-0.32 cache stride API",
)
@pytest.mark.parametrize(
    "layout,include_layers,expected",
    [
        ("NHD", False, (0, 1, 2, 3, 4)),
        ("NHD", True, (2, 0, 1, 3, 4, 5)),
        ("HND", False, (0, 1, 3, 2, 4)),
        ("HND", True, (2, 4, 0, 1, 3, 5)),
    ],
)
def test_legacy_cache_stride_order_remains_available(
    monkeypatch, layout, include_layers, expected
):
    from vllm.v1.attention.backends import utils

    monkeypatch.setattr(utils, "get_kv_cache_layout", lambda: layout)
    assert (
        attention.AttentionGCUBackend.get_kv_cache_stride_order(include_layers)
        == expected
    )

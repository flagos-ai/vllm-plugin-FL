# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU views over the real vLLM 0.32 KV cache specification and allocator."""

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.sunrise.impl.kv_cache import split_kv_cache


def test_logical_cache_split_preserves_storage_and_block_ids():
    logical = torch.zeros(3, 2, 16, 16)
    key, value = split_kv_cache(logical, 8)
    assert key.shape == value.shape == (3, 16, 2, 8)
    assert key.untyped_storage().data_ptr() == logical.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == logical.untyped_storage().data_ptr()
    key[2, 5, 1].fill_(3)
    value[2, 5, 1].fill_(7)
    torch.testing.assert_close(logical[2, 1, 5, :8], torch.full((8,), 3.0))
    torch.testing.assert_close(logical[2, 1, 5, 8:], torch.full((8,), 7.0))
    assert not logical[:2].any()


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 2, 16, 15)])
def test_unsupported_cache_shape_fails_explicitly(shape):
    with pytest.raises(ValueError, match="dense unquantized KV"):
        split_kv_cache(torch.zeros(shape), 8)


@pytest.mark.parametrize("layout_name", ["LBNHC", "LBHNC", "BLNHC", "BHLNC"])
def test_native_v032_packed_views_keep_layers_and_kv_disjoint(layout_name):
    kv = pytest.importorskip("vllm.v1.kv_cache_interface")
    # This test intentionally uses the actual 0.32 dataclasses and stride builder,
    # including the new layers/layer_stride fields; no legacy spec stubs.
    spec = kv.FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
    )
    layout = kv.KVCacheLayout[layout_name]
    strides = kv.compute_layout_strides(spec, 3, 2, layout)
    allocation = kv.KVCacheTensor(
        size=2 * 3 * spec.page_size_bytes,
        layers=["model.layers.0.self_attn", "model.layers.1.self_attn"],
        layer_stride=strides[0],
        block_stride=strides[1],
    )
    raw = torch.zeros(allocation.size, dtype=torch.int8)
    views = kv.create_kv_cache_views(raw, spec, 3, layout, allocation)
    assert len(views) == 2
    key, value = split_kv_cache(views[0], spec.head_size)
    key[2, 5, 1].fill_(3)
    value[2, 5, 1].fill_(7)
    torch.testing.assert_close(views[0][2, 1, 5, :8].float(), torch.full((8,), 3.0))
    torch.testing.assert_close(views[0][2, 1, 5, 8:].float(), torch.full((8,), 7.0))
    assert not views[0][:2].any()
    assert not views[1].any()
    assert key.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()


def test_native_v2_discovers_and_customizes_specs_without_global_config(monkeypatch):
    kv = pytest.importorskip("vllm.v1.kv_cache_interface")
    from types import SimpleNamespace

    from vllm.config import get_current_vllm_config_or_none, vllm as config_module
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
    from vllm.v1.worker.utils import customize_attention_spec

    from vllm_fl.dispatch.backends.vendor.sunrise.impl.attention import (
        AttentionFLBackend,
    )

    monkeypatch.setattr(config_module, "_current_vllm_config", None)
    assert get_current_vllm_config_or_none() is None
    spec = kv.FullAttentionSpec(
        block_size=128, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
    )

    class Layer(AttentionLayerBase):
        def get_attn_backend(self):
            return AttentionFLBackend

        def get_kv_cache_spec(self, config):
            return spec

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"layer": Layer()}),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    customized = customize_attention_spec(AttentionFLBackend, spec)
    assert customized.block_size == 128
    assert customized.block_stride_alignment.base == 64
    discovered = get_kv_cache_spec(config)
    assert discovered == {"layer": customized}
    assert get_current_vllm_config_or_none() is None

# Copyright 2026 FlagOS Contributors
"""CPU contracts for the NVIDIA RoPE mutation workaround."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_rotary():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/dispatch/backends/flaggems/impl/rotary.py"
    )
    spec = importlib.util.spec_from_file_location("rotary_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rotary_embedding_flaggems


@pytest.mark.parametrize("vendor", ["nvidia", "musa"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("packed_views", [False, True])
def test_provider_routing_preserves_mutation_and_storage(
    monkeypatch, vendor, inplace, interleaved, packed_views
):
    # Real CPU tensors exercise view mutation without loading GPU packages.
    packed = torch.arange(7 * 6 * 160, dtype=torch.float32).reshape(7, 6, 160)
    packed_before = packed.clone()
    query, key = packed[:, :4, :80], packed[:, 4:, :80]
    if not packed_views:
        query, key = query.contiguous(), key.contiguous()
    before = query.clone(), key.clone()
    strides = query.stride(), key.stride()
    cos, sin = torch.ones(16, 40), torch.zeros(16, 40)
    positions = torch.tensor([3, 0, 7, 4, 9, 1, 5])
    calls = []
    produced = []

    def provider(q, k, c, s, *, position_ids, rotary_interleaved, inplace):
        assert q is query and k is key
        assert c is cos and s is sin
        assert position_ids is positions
        assert rotary_interleaved is interleaved
        if vendor == "nvidia":
            assert not inplace, "NVIDIA must not call the unsafe in-place kernel"
        calls.append(inplace)
        result_q, result_k = q + 5, k - 7
        if inplace:
            q.copy_(result_q)
            k.copy_(result_k)
            result_q, result_k = q, k
        produced.append((result_q, result_k))
        return result_q, result_k

    gems = ModuleType("flag_gems")
    gems.__path__ = []
    gems.vendor_name = vendor
    config = ModuleType("flag_gems.config")
    config.use_c_extension = False
    gems.config = config
    provider_module = ModuleType("flaggems_vllm")
    provider_module.apply_rotary_pos_emb = provider
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    monkeypatch.setitem(sys.modules, "flag_gems.config", config)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", provider_module)

    result = _load_rotary()(
        None,
        query,
        key,
        cos,
        sin,
        positions,
        rotary_interleaved=interleaved,
        inplace=inplace,
    )

    assert calls == [False if vendor == "nvidia" else inplace]
    for actual, value, saved, stride, delta in zip(
        result, (query, key), before, strides, (5, -7)
    ):
        torch.testing.assert_close(actual, saved + delta, rtol=0, atol=0)
        assert actual.dtype == value.dtype and actual.device == value.device
        if inplace:
            assert actual is value
            assert actual.stride() == stride
            assert actual.data_ptr() == value.data_ptr()
            torch.testing.assert_close(value, saved + delta, rtol=0, atol=0)
        else:
            assert actual.data_ptr() != value.data_ptr()
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
    if not inplace:
        assert result[0] is produced[0][0] and result[1] is produced[0][1]
    if packed_views and inplace:
        torch.testing.assert_close(packed[:, :, 80:], packed_before[:, :, 80:])
    else:
        torch.testing.assert_close(packed, packed_before)

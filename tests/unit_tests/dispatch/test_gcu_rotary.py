# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU contract checks for the GCU RoPE adapter; no simulated GPU numerics."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_PATH = Path(__file__).parents[3] / "vllm_fl/dispatch/backends/flaggems/impl/rotary.py"
_SPEC = importlib.util.spec_from_file_location("_gcu_rotary_contract", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ROTARY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ROTARY)


class _TensorOn:
    """Expose device metadata while keeping copy/stride checks on real CPU data."""

    def __init__(self, value, device="gcu"):
        self.value = value
        self.device = SimpleNamespace(type=device)
        self.copy_calls = 0

    def copy_(self, source):
        self.copy_calls += 1
        self.value.copy_(source)
        return self


def _install_provider(monkeypatch, provider):
    gems = ModuleType("flag_gems")
    gems.vendor_name = "enflame"
    config = ModuleType("flag_gems.config")
    config.use_c_extension = False
    fused = ModuleType("flaggems_vllm")
    fused.apply_rotary_pos_emb = provider
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    monkeypatch.setitem(sys.modules, "flag_gems.config", config)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", fused)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed_views", [False, True])
def test_gcu_split_half_copyback_preserves_storage_and_arguments(
    monkeypatch, dtype, packed_views
):
    packed = torch.arange(2 * 3 * 8, device="cpu").reshape(2, 3, 8).to(dtype)
    untouched = packed.clone()
    query_value, key_value = packed[:, :2, :4], packed[:, 2:, :4]
    if not packed_views:
        query_value, key_value = query_value.contiguous(), key_value.contiguous()
    query, key = _TensorOn(query_value), _TensorOn(key_value)
    pointers = [value.data_ptr() for value in (query_value, key_value)]
    strides = [value.stride() for value in (query_value, key_value)]
    outputs = query_value.clone() + 1, key_value.clone() - 2
    cos, sin, positions = object(), object(), object()
    calls = []

    def provider(q, k, c, s, **kwargs):
        calls.append((q, k, c, s, kwargs))
        assert q is query and k is key
        assert c is cos and s is sin
        assert kwargs == {
            "position_ids": positions,
            "rotary_interleaved": False,
            "inplace": False,
        }
        return outputs

    _install_provider(monkeypatch, provider)
    actual = _ROTARY.rotary_embedding_flaggems(
        None, query, key, cos, sin, positions, rotary_interleaved=False, inplace=True
    )
    assert actual == (query, key)
    assert len(calls) == 1
    for value, output, pointer, stride in zip((query, key), outputs, pointers, strides):
        assert value.copy_calls == 1
        assert value.value.dtype == dtype
        assert value.value.data_ptr() == pointer
        assert value.value.stride() == stride
        torch.testing.assert_close(value.value, output, rtol=0, atol=0)
    if packed_views:
        # Q and K share a packed allocation but have non-overlapping head views.
        torch.testing.assert_close(
            packed[:, :, 4:], untouched[:, :, 4:], rtol=0, atol=0
        )
    else:
        torch.testing.assert_close(packed, untouched, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device,inplace,interleaved",
    [
        (device, inplace, interleaved)
        for device in ("gcu", "cuda", "npu", "cpu")
        for inplace in (False, True)
        for interleaved in (False, True)
        if not (device == "gcu" and inplace and not interleaved)
    ],
)
def test_other_devices_and_modes_keep_original_provider_call(
    monkeypatch, device, inplace, interleaved
):
    query = _TensorOn(torch.ones((1, 2, 4), device="cpu"), device)
    key = _TensorOn(torch.zeros((1, 1, 4), device="cpu"), device)
    cos, sin, positions = object(), object(), object()
    result = (object(), object())
    calls = []

    def provider(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    _install_provider(monkeypatch, provider)
    actual = _ROTARY.rotary_embedding_flaggems(
        None, query, key, cos, sin, positions, interleaved, inplace
    )
    assert actual is result
    assert calls == [
        (
            (query, key, cos, sin),
            {
                "position_ids": positions,
                "rotary_interleaved": interleaved,
                "inplace": inplace,
            },
        )
    ]
    assert query.copy_calls == key.copy_calls == 0


@pytest.mark.parametrize("missing_key", [False, True])
def test_gcu_provider_errors_propagate_before_mutation(monkeypatch, missing_key):
    query = _TensorOn(torch.ones((1, 2, 4), device="cpu"))
    key = None if missing_key else _TensorOn(torch.zeros((1, 1, 4), device="cpu"))
    error = RuntimeError("provider rejected input")

    def provider(q, k, *args, **kwargs):
        assert q is query and k is key
        raise error

    _install_provider(monkeypatch, provider)
    with pytest.raises(RuntimeError) as caught:
        _ROTARY.rotary_embedding_flaggems(
            None, query, key, object(), object(), object(), inplace=True
        )
    assert caught.value is error
    assert query.copy_calls == 0
    assert key is None or key.copy_calls == 0

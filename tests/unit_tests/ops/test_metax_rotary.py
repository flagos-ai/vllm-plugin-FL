# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU numerics and native ABI recording for MetaX rotary dispatch."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/rotary_embedding.py"


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _oracle(value, cos, sin, positions, interleaved):
    c, s = cos[positions], sin[positions]
    if value.ndim == 4 and positions.ndim == 1:
        c, s = c[None, None], s[None, None]
    else:
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    if cos.shape[-1] * 2 == value.shape[-1]:
        if interleaved:
            c, s = c.repeat_interleave(2, dim=-1), s.repeat_interleave(2, dim=-1)
        else:
            c, s = torch.cat((c, c), dim=-1), torch.cat((s, s), dim=-1)
    if interleaved:
        rotated = torch.stack((-value[..., 1::2], value[..., ::2]), dim=-1).flatten(-2)
    else:
        left, right = value.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
    return value * c + rotated * s


@pytest.fixture
def rotary(monkeypatch):
    calls = []

    def native(positions, query, key, head_size, cache, is_neox):
        assert isinstance(head_size, int) and head_size == query.shape[-1]
        assert query.is_contiguous() and key.is_contiguous()
        assert positions.ndim == 1 and positions.dtype == torch.long
        assert cache.dtype == query.dtype and cache.device == query.device
        cos, sin = cache.chunk(2, dim=-1)
        q_out = _oracle(query, cos, sin, positions, not is_neox)
        k_out = _oracle(key, cos, sin, positions, not is_neox)
        query.copy_(q_out)
        key.copy_(k_out)
        calls.append((positions, head_size, cache, is_neox))

    ops = ModuleType("vllm._custom_ops")
    ops.rotary_embedding = native
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    _load(
        monkeypatch,
        "vllm_fl.dispatch.backends.reference.impl.rotary",
        ROOT / "vllm_fl/dispatch/backends/reference/impl/rotary.py",
    )
    module = _load(monkeypatch, "metax_rotary_under_test", SOURCE)
    return module.rotary_embedding_maca, calls


@pytest.mark.parametrize("layout", ["tokens", "batch_shared", "batch_positions"])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_native_rope_abi_preserves_values_mutation_and_partial_views(
    rotary, layout, interleaved, inplace, partial
):
    apply, calls = rotary
    torch.manual_seed(17)
    head_size = 12 if partial else 8
    if layout == "tokens":
        q_owner, k_owner = torch.randn(5, 3, head_size), torch.randn(5, 2, head_size)
        positions = torch.tensor([3, 1, 7, 0, 5], dtype=torch.int32)
    else:
        q_owner, k_owner = (
            torch.randn(2, 3, 5, head_size),
            torch.randn(2, 2, 5, head_size),
        )
        positions = torch.tensor([3, 1, 7, 0, 5], dtype=torch.int32)
        if layout == "batch_positions":
            positions = torch.stack((positions, positions.flip(0)))
    query, key = q_owner[..., :8], k_owner[..., :8]
    q_before, k_before = query.clone(), key.clone()
    q_tail, k_tail = q_owner[..., 8:].clone(), k_owner[..., 8:].clone()
    cos, sin = torch.randn(12, 4), torch.randn(12, 4)
    q_expected = _oracle(query, cos, sin, positions.long(), interleaved)
    k_expected = _oracle(key, cos, sin, positions.long(), interleaved)
    q_out, k_out = apply(
        object(), query, key, cos, sin, positions, interleaved, inplace
    )
    torch.testing.assert_close(q_out, q_expected)
    torch.testing.assert_close(k_out, k_expected)
    torch.testing.assert_close(q_owner[..., 8:], q_tail)
    torch.testing.assert_close(k_owner[..., 8:], k_tail)
    assert len(calls) == 1
    assert calls[0][1] == 8 and calls[0][3] is (not interleaved)
    torch.testing.assert_close(calls[0][2], torch.cat((cos, sin), dim=-1))
    if inplace:
        assert q_out is query and k_out is key
    else:
        assert (
            q_out.data_ptr() != query.data_ptr() and k_out.data_ptr() != key.data_ptr()
        )
        torch.testing.assert_close(query, q_before)
        torch.testing.assert_close(key, k_before)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_independent_full_width_coefficients_keep_fl_contract(
    rotary, interleaved, inplace
):
    apply, calls = rotary
    query, key = torch.randn(3, 2, 8), torch.randn(3, 1, 8)
    cos, sin = torch.randn(8, 8), torch.randn(8, 8)
    positions = torch.tensor([4, 1, 3])
    q_before, k_before = query.clone(), key.clone()
    q_expected = _oracle(query, cos, sin, positions, interleaved)
    k_expected = _oracle(key, cos, sin, positions, interleaved)
    q_out, k_out = apply(
        object(), query, key, cos, sin, positions, interleaved, inplace
    )
    torch.testing.assert_close(q_out, q_expected)
    torch.testing.assert_close(k_out, k_expected)
    assert not calls  # Native half-width ABI cannot encode arbitrary coefficients.
    if inplace:
        assert q_out is query and k_out is key
    else:
        torch.testing.assert_close(query, q_before)
        torch.testing.assert_close(key, k_before)


def test_native_rope_refuses_non_integer_positions(rotary):
    apply, calls = rotary
    query, key = torch.randn(3, 2, 8), torch.randn(3, 1, 8)
    with pytest.raises(ValueError, match="int32 or int64"):
        apply(object(), query, key, torch.ones(8, 4), torch.zeros(8, 4), torch.ones(3))
    assert not calls

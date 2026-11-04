# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU numerics and isolation contracts for the TXDA MRoPE row lookup."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/txda_mrope.py"


@pytest.fixture
def adapter(monkeypatch):
    logging = ModuleType("vllm.logger")
    logger = Mock()
    logging.init_logger = Mock(return_value=logger)
    monkeypatch.setitem(sys.modules, logging.__name__, logging)
    spec = importlib.util.spec_from_file_location("txda_mrope_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    upstream = ModuleType("vllm.model_executor.layers.rotary_embedding.mrope")

    class MRotaryEmbedding:
        def forward_native(self, positions, query, key=None, offsets=None):
            return query, key, offsets

    def interleave(value, sections):
        channels = torch.arange(value.shape[-1])
        height = (channels % 3 == 1) & (channels < sections[1] * 3)
        width = (channels % 3 == 2) & (channels < sections[2] * 3)
        return torch.where(width, value[2], torch.where(height, value[1], value[0]))

    upstream.MRotaryEmbedding = MRotaryEmbedding
    upstream.apply_interleaved_rope = Mock(side_effect=interleave)
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="txda")
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    return SimpleNamespace(
        module=module,
        logger=logger,
        logging=logging,
        upstream=upstream,
        platforms=platforms,
    )


def positions_for(name):
    if name == "random":
        return torch.tensor([7, 1, 7, 2, 15, 0, 9], dtype=torch.int32)
    if name == "negative":
        return torch.tensor([-16, -1, 0, 15, -3, -3])
    if name == "three_axes":
        return torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    if name == "strided":
        return (torch.arange(36) % 16).view(3, 12)[:, ::2]
    if name == "scalar":
        return torch.tensor(3)
    shape = (0,) if name == "empty1" else (3, 0)
    return torch.empty(shape, dtype=torch.int64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "indices",
    ["random", "negative", "three_axes", "strided", "scalar", "empty1", "empty3"],
)
def test_gather_matches_advanced_index_values_and_preserves_inputs(
    adapter, dtype, strided, indices
):
    raw = torch.arange(400, dtype=torch.float32).to(dtype)
    cache = raw[8:392].view(16, 24)[:, ::2]
    if not strided:
        cache = cache.contiguous()
    positions = positions_for(indices)
    before_cache, before_positions = cache.clone(), positions.clone()
    actual = adapter.module.gather_rows(cache, positions)
    expected = cache[positions]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (*positions.shape, cache.shape[1])
    assert actual.dtype == dtype and actual.device == cache.device
    torch.testing.assert_close(cache, before_cache, rtol=0, atol=0)
    assert torch.equal(positions, before_positions)


@pytest.mark.parametrize(
    "kind", ["high", "low", "float", "bool", "cache_rank", "index_device"]
)
def test_invalid_lookup_is_rejected_before_output_allocation(
    adapter, monkeypatch, kind
):
    cache = torch.arange(48).view(4, 12)
    positions = torch.tensor([0, 1])
    if kind == "high":
        positions[1] = 4
    elif kind == "low":
        positions[1] = -5
    elif kind == "float":
        positions = positions.float()
    elif kind == "bool":
        positions = positions.bool()
    elif kind == "cache_rank":
        cache = cache.unsqueeze(0)
    else:
        positions = torch.empty(2, dtype=torch.int64, device="meta")
    allocate = Mock(side_effect=AssertionError("must validate before allocating"))
    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises((IndexError, TypeError, ValueError)):
        adapter.module.gather_rows(cache, positions)
    allocate.assert_not_called()


@pytest.mark.parametrize("shape", [(0, 12), (4, 0)])
def test_zero_sized_cache_dimensions_have_native_indexing_shape(adapter, shape):
    cache = torch.empty(shape)
    positions = (
        torch.empty(3, 0, dtype=torch.int64) if not shape[0] else torch.tensor([0, 3])
    )
    result = adapter.module.gather_rows(cache, positions)
    assert result.shape == (*positions.shape, shape[1])
    assert result.numel() == 0


def test_lookup_transfers_only_indices_and_never_invokes_advanced_gather(
    adapter, monkeypatch
):
    cache = torch.arange(192).view(16, 12)
    positions = torch.tensor([2, 3, 4, 9, 10, 9])
    expected = cache[positions]
    copied = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copied.append(tensor)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    for name in ("index_select", "gather", "cat"):
        monkeypatch.setattr(torch, name, Mock(side_effect=AssertionError(name)))
    result = adapter.module.gather_rows(cache, positions)
    assert len(copied) == 1 and copied[0] is positions
    assert torch.equal(result, expected)


def apply_rotation(x, cos, sin, neox=True):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if neox:
        first, second = x.chunk(2, dim=-1)
        return torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
    pairs = x.unflatten(-1, (-1, 2))
    first, second = pairs[..., 0], pairs[..., 1]
    return torch.stack(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).flatten(-2)


def rotation_oracle(x, cache, positions, sections, interleaved, neox):
    tokens, width = positions.shape[-1], cache.shape[1] // 2
    cos = torch.empty(tokens, width, dtype=cache.dtype)
    sin = torch.empty_like(cos)
    for token in range(tokens):
        for channel in range(width):
            if positions.ndim == 1:
                position = positions[token]
            else:
                if interleaved:
                    axis = 1 if channel % 3 == 1 and channel < sections[1] * 3 else 0
                    if channel % 3 == 2 and channel < sections[2] * 3:
                        axis = 2
                else:
                    axis = (
                        0
                        if channel < sections[0]
                        else 1
                        if channel < sum(sections[:2])
                        else 2
                    )
                position = positions[axis, token]
            cos[token, channel] = cache[position, channel]
            sin[token, channel] = cache[position, channel + width]
    shaped = x.view(tokens, -1, 16)
    expected = shaped.clone()
    # Independent per-pair scalar layout rather than the production helpers.
    for channel in range(width):
        first, second = (
            (channel, channel + width) if neox else (2 * channel, 2 * channel + 1)
        )
        expected[..., first] = (
            shaped[..., first] * cos[:, channel, None]
            - shaped[..., second] * sin[:, channel, None]
        )
        expected[..., second] = (
            shaped[..., second] * cos[:, channel, None]
            + shaped[..., first] * sin[:, channel, None]
        )
    return expected.reshape(x.shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("axes", [1, 3])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("tokens", [1, 8, 65])
def test_native_rotation_preserves_axis_layout_and_partial_tail(
    adapter, dtype, axes, interleaved, tokens
):
    gen = torch.Generator().manual_seed(20261104)
    cache = torch.randn(128, 12, generator=gen, dtype=dtype)
    query = torch.randn(tokens, 32, generator=gen, dtype=dtype)
    key = torch.randn(tokens, 16, generator=gen, dtype=dtype)
    positions = (
        torch.arange(tokens)
        if axes == 1
        else torch.stack(
            [(torch.arange(tokens) + shift) % 128 for shift in (0, 11, 23)]
        )
    )
    match = Mock(return_value=cache)
    rotate = Mock(side_effect=apply_rotation)
    layer = SimpleNamespace(
        _match_cos_sin_cache_dtype=match,
        head_size=16,
        rotary_dim=12,
        mrope_section=[2, 2, 2],
        mrope_interleaved=interleaved,
        apply_rotary_emb=SimpleNamespace(forward_native=rotate),
    )
    expected = [
        rotation_oracle(x, cache, positions, [2, 2, 2], interleaved, True)
        for x in (query, key)
    ]
    actual = adapter.module._forward_native(
        layer, positions, query, key, offsets=torch.ones(tokens)
    )
    for result, reference, original in zip(actual, expected, (query, key)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert torch.equal(
            result.view(tokens, -1, 16)[..., 12:],
            original.view(tokens, -1, 16)[..., 12:],
        )
    assert rotate.call_count == 2
    match.assert_called_once_with(query)
    assert adapter.upstream.apply_interleaved_rope.call_count == (
        2 if axes == 3 and interleaved else 0
    )
    adapter.module._forward_native(layer, positions, query, key)
    adapter.logger.info.assert_called_once()
    assert adapter.logger.info.call_args.args[0].startswith("TXDA MRoPE lookup active")
    adapter.logging.init_logger.assert_called_once_with("vllm.fl.txda.mrope")


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu", "musa"])
def test_other_platforms_do_not_install_the_override(adapter, device):
    cls = adapter.upstream.MRotaryEmbedding
    original = cls.forward_native
    adapter.platforms.current_platform.device_type = device
    assert not adapter.module.patch_txda_mrope()
    assert cls.forward_native is original


def test_install_is_idempotent_preserves_cpu_path_and_fails_without_fallback(
    adapter, monkeypatch
):
    cls = adapter.upstream.MRotaryEmbedding
    original = cls.forward_native
    assert adapter.module.patch_txda_mrope()
    installed = cls.forward_native
    assert installed.__wrapped__ is original
    assert adapter.module.patch_txda_mrope() and cls.forward_native is installed
    query, key, offsets = torch.ones(1, 16), torch.zeros(1, 16), object()
    result = cls().forward_native(torch.tensor([0]), query, key, offsets)
    assert result == (query, key, offsets)
    failure = Mock(side_effect=RuntimeError("lookup failed"))
    monkeypatch.setattr(adapter.module, "_forward_native", failure)
    txda_query = SimpleNamespace(device=SimpleNamespace(type="txda"))
    with pytest.raises(RuntimeError, match="lookup failed"):
        cls().forward_native(torch.tensor([0]), txda_query, key)
    failure.assert_called_once()

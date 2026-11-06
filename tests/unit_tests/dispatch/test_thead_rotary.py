# Copyright (c) 2026 BAAI. All rights reserved.

"""Contract tests for the T-Head PPU rotary path in the FlagGems backend.

The NVIDIA FlagGems parity suite only runs on NVIDIA, so these cases pin the
T-Head branch of ``rotary_embedding_flaggems``: the vendor gate must compute
both rotations out of place and copy them back, keeping the caller's values,
storage and strides for compact and packed Q/K views, including graph replay.
"""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[3]
_ROTARY_MODULE = "vllm_fl/dispatch/backends/flaggems/impl/rotary.py"


def _load_source_module(name, relative_path, device_type="cuda"):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Load the source contract without importing the plugin package or
    # initializing a real vendor backend during collection.
    platform = SimpleNamespace(device_type=device_type)

    def source_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.platforms" and level == 0:
            assert tuple(fromlist) == ("current_platform",)
            return SimpleNamespace(current_platform=platform)
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = {
        **vars(builtins),
        "__import__": source_import,
    }
    spec.loader.exec_module(module)
    return module


flaggems_rotary = _load_source_module("_test_thead_rotary", _ROTARY_MODULE)


def _rotary_inputs(dtype=torch.float32):
    query = torch.randn(4, 2, 8, dtype=dtype)
    key = torch.randn(4, 2, 8, dtype=dtype)
    positions = torch.arange(4)
    frequencies = torch.outer(
        torch.arange(16, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
    )
    return (
        query,
        key,
        frequencies.cos().to(dtype),
        frequencies.sin().to(dtype),
        positions,
    )


def _expected_rotary(value, cos, sin, positions, interleaved):
    selected_cos = cos[positions].unsqueeze(1)
    selected_sin = sin[positions].unsqueeze(1)
    if selected_cos.shape[-1] != value.shape[-1]:
        if interleaved:
            selected_cos = selected_cos.repeat_interleave(2, dim=-1)
            selected_sin = selected_sin.repeat_interleave(2, dim=-1)
        else:
            selected_cos = torch.cat((selected_cos, selected_cos), dim=-1)
            selected_sin = torch.cat((selected_sin, selected_sin), dim=-1)

    if interleaved:
        first = value[..., ::2]
        second = value[..., 1::2]
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    else:
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    return value * selected_cos + rotated * selected_sin


def _install_fake_flaggems(monkeypatch, common_rotary, vendor_name):
    flag_gems = ModuleType("flag_gems")
    flag_gems.__path__ = []
    flag_gems.vendor_name = vendor_name

    config = ModuleType("flag_gems.config")
    config.use_c_extension = False
    fused = ModuleType("flaggems_vllm")
    fused.apply_rotary_pos_emb = common_rotary

    flag_gems.config = config
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setitem(sys.modules, "flag_gems.config", config)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", fused)


@pytest.mark.parametrize("packed_views", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_thead_rotary_contract_preserves_values_storage_and_strides(
    monkeypatch, inplace, packed_views, interleaved
):
    calls = []

    def common_rotary(
        query, key, cos, sin, position_ids=None, rotary_interleaved=False, inplace=True
    ):
        calls.append(inplace)
        return (
            _expected_rotary(query, cos, sin, position_ids, rotary_interleaved),
            _expected_rotary(key, cos, sin, position_ids, rotary_interleaved),
        )

    _install_fake_flaggems(monkeypatch, common_rotary, "thead")
    packed = torch.randn(4, 3, 2, 16, dtype=torch.bfloat16)
    packed_before = packed.clone()
    if packed_views:
        # Q and K keep the full packed head width as their stride, so the
        # views are not compact and copy_ must honour the gaps.
        query, key = packed[:, 0, :, :8], packed[:, 1, :, :8]
        assert not query.is_contiguous()
        assert not key.is_contiguous()
    else:
        query = packed[:, 0, :, :8].contiguous()
        key = packed[:, 1, :, :8].contiguous()
        packed_before = packed.clone()
    query_before, key_before = query.clone(), key.clone()
    _, _, cos, sin, positions = _rotary_inputs(torch.bfloat16)
    expected_query = _expected_rotary(query_before, cos, sin, positions, interleaved)
    expected_key = _expected_rotary(key_before, cos, sin, positions, interleaved)

    actual_query, actual_key = flaggems_rotary.rotary_embedding_flaggems(
        None,
        query,
        key,
        cos,
        sin,
        positions,
        rotary_interleaved=interleaved,
        inplace=inplace,
    )

    # The T-Head kernel must always be asked for out-of-place results.
    assert calls == [False]
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
    if inplace:
        assert actual_query is query
        assert actual_key is key
        torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
        if packed_views:
            torch.testing.assert_close(
                packed[:, 0, :, 8:], packed_before[:, 0, :, 8:], rtol=0, atol=0
            )
            torch.testing.assert_close(
                packed[:, 1, :, 8:], packed_before[:, 1, :, 8:], rtol=0, atol=0
            )
            torch.testing.assert_close(
                packed[:, 2], packed_before[:, 2], rtol=0, atol=0
            )
    else:
        assert actual_query is not query
        assert actual_key is not key
        torch.testing.assert_close(query, query_before, rtol=0, atol=0)
        torch.testing.assert_close(key, key_before, rtol=0, atol=0)
        if packed_views:
            torch.testing.assert_close(packed, packed_before, rtol=0, atol=0)


@pytest.mark.parametrize("vendor_name", ["musa", "gcu"])
def test_thead_rotary_contract_does_not_change_other_vendors(monkeypatch, vendor_name):
    calls = []

    def common_rotary(
        query, key, cos, sin, position_ids=None, rotary_interleaved=False, inplace=True
    ):
        calls.append(inplace)
        return query, key

    _install_fake_flaggems(monkeypatch, common_rotary, vendor_name)
    query, key, cos, sin, positions = _rotary_inputs(torch.bfloat16)

    result_query, result_key = flaggems_rotary.rotary_embedding_flaggems(
        None, query, key, cos, sin, positions, inplace=True
    )

    # Only the T-Head vendor gate allocates out of place; other vendors keep
    # passing the caller's inplace flag straight through.
    assert calls == [True]
    assert result_query is query
    assert result_key is key


def _require_thead_rotary(backend):
    if backend != "thead":
        pytest.skip("T-Head PPU rotary contract")
    if not torch.cuda.is_available():
        pytest.skip("T-Head PPU is unavailable")
    import flag_gems

    if getattr(flag_gems, "vendor_name", None) != "thead":
        pytest.skip("FlagGems is not configured for T-Head")
    pytest.importorskip("flaggems_vllm")


@pytest.mark.gpu
@pytest.mark.flaggems
@pytest.mark.parametrize("packed_views", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_thead_rotary_packed_partial_views_on_ppu(
    backend, device, inplace, packed_views, interleaved
):
    _require_thead_rotary(backend)
    packed_cpu = torch.randn(4, 3, 2, 16, dtype=torch.float32).to(torch.bfloat16)
    packed = packed_cpu.to(device)
    if packed_views:
        query, key = packed[:, 0, :, :8], packed[:, 1, :, :8]
        assert not query.is_contiguous()
        assert not key.is_contiguous()
        query_cpu, key_cpu = packed_cpu[:, 0, :, :8], packed_cpu[:, 1, :, :8]
    else:
        query = packed[:, 0, :, :8].contiguous()
        key = packed[:, 1, :, :8].contiguous()
        query_cpu = packed_cpu[:, 0, :, :8].contiguous()
        key_cpu = packed_cpu[:, 1, :, :8].contiguous()
    _, _, cos_cpu, sin_cpu, positions_cpu = _rotary_inputs(torch.bfloat16)
    expected_query = _expected_rotary(
        query_cpu.float(), cos_cpu.float(), sin_cpu.float(), positions_cpu, interleaved
    ).to(torch.bfloat16)
    expected_key = _expected_rotary(
        key_cpu.float(), cos_cpu.float(), sin_cpu.float(), positions_cpu, interleaved
    ).to(torch.bfloat16)

    actual_query, actual_key = flaggems_rotary.rotary_embedding_flaggems(
        None,
        query,
        key,
        cos_cpu.to(device),
        sin_cpu.to(device),
        positions_cpu.to(device),
        rotary_interleaved=interleaved,
        inplace=inplace,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_query.cpu(), expected_query, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_key.cpu(), expected_key, rtol=2e-2, atol=2e-2)
    if inplace:
        assert actual_query is query
        assert actual_key is key
        torch.testing.assert_close(
            packed[:, 0, :, 8:].cpu(), packed_cpu[:, 0, :, 8:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed[:, 1, :, 8:].cpu(), packed_cpu[:, 1, :, 8:], rtol=0, atol=0
        )
        torch.testing.assert_close(packed[:, 2].cpu(), packed_cpu[:, 2], rtol=0, atol=0)
    else:
        assert actual_query is not query
        assert actual_key is not key
        torch.testing.assert_close(packed.cpu(), packed_cpu, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.flaggems
def test_thead_rotary_inplace_graph_replays_changed_packed_views(backend, device):
    _require_thead_rotary(backend)
    generator = torch.Generator().manual_seed(29)
    packed = torch.randn((7, 6, 160), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    query, key = packed[:, :4, :80], packed[:, 4:, :80]
    angles = torch.randn((16, 40), generator=generator)
    cos = angles.cos().to(device=device, dtype=torch.bfloat16)
    sin = angles.sin().to(device=device, dtype=torch.bfloat16)
    base_positions = torch.tensor([3, 0, 7, 4, 9, 1, 5])
    positions = base_positions.to(device)

    def rotate():
        return flaggems_rotary.rotary_embedding_flaggems(
            None,
            query,
            key,
            cos,
            sin,
            positions,
            rotary_interleaved=False,
            inplace=True,
        )

    # Warm up outside capture so compilation does not become part of the graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            rotate()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rotate()

    for replay in range(3):
        original = torch.randn((7, 6, 160), generator=generator).to(torch.bfloat16)
        positions_cpu = base_positions + replay
        packed.copy_(original)
        positions.copy_(positions_cpu)
        graph.replay()
        for name, output, value, source in zip(
            ("query", "key"),
            captured,
            (query, key),
            (original[:, :4, :80], original[:, 4:, :80]),
        ):
            expected = _expected_rotary(
                source.float(),
                cos.cpu().float(),
                sin.cpu().float(),
                positions_cpu,
                False,
            ).to(torch.bfloat16)
            assert output.data_ptr() == value.data_ptr()
            torch.testing.assert_close(
                value.cpu(),
                expected,
                rtol=2e-2,
                atol=2e-2,
                msg=lambda m, n=name: f"{n}: {m}",
            )
        torch.testing.assert_close(
            packed[:, :4, 80:].cpu(), original[:, :4, 80:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed[:, 4:, 80:].cpu(), original[:, 4:, 80:], rtol=0, atol=0
        )

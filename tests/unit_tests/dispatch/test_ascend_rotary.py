# Copyright (c) 2026 BAAI. All rights reserved.

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[3]
_REFERENCE_MODULE = "vllm_fl.dispatch.backends.reference.impl.rotary"


def _load_source_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference_rotary = _load_source_module(
    "_test_reference_rotary",
    "vllm_fl/dispatch/backends/reference/impl/rotary.py",
)
ascend_rotary = _load_source_module(
    "_test_ascend_rotary",
    "vllm_fl/dispatch/backends/vendor/ascend/impl/rotary.py",
)
flaggems_rotary = _load_source_module(
    "_test_flaggems_rotary",
    "vllm_fl/dispatch/backends/flaggems/impl/rotary.py",
)


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


def _install_fake_flaggems(monkeypatch, apply_rotary, common_rotary):
    flag_gems = ModuleType("flag_gems")
    flag_gems.__path__ = []
    flag_gems.apply_rotary_pos_emb = apply_rotary

    config = ModuleType("flag_gems.config")
    config.use_c_extension = False
    modules = ModuleType("flag_gems.modules")
    modules.__path__ = []
    rotary = ModuleType("flag_gems.modules.rotary_embedding")
    rotary.gems_rope_forward = common_rotary

    flag_gems.config = config
    flag_gems.modules = modules
    modules.rotary_embedding = rotary
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setitem(sys.modules, "flag_gems.config", config)
    monkeypatch.setitem(sys.modules, "flag_gems.modules", modules)
    monkeypatch.setitem(sys.modules, "flag_gems.modules.rotary_embedding", rotary)


def test_reference_interleaved_rotary_repeats_adjacent_cache_values():
    query, key, cos, sin, positions = _rotary_inputs()
    actual_query, actual_key = reference_rotary.rotary_embedding_torch(
        None,
        query,
        key,
        cos,
        sin,
        positions,
        rotary_interleaved=True,
        inplace=False,
    )

    torch.testing.assert_close(
        actual_query, _expected_rotary(query, cos, sin, positions, True)
    )
    torch.testing.assert_close(
        actual_key, _expected_rotary(key, cos, sin, positions, True)
    )

    half_query = query.to(torch.float16)
    half_key = key.to(torch.float16)
    half_query_result, half_key_result = reference_rotary.rotary_embedding_torch(
        None,
        half_query,
        half_key,
        cos,
        sin,
        positions,
        rotary_interleaved=True,
        inplace=False,
    )
    assert half_query_result.dtype == torch.float16
    assert half_key_result.dtype == torch.float16
    torch.testing.assert_close(
        half_query_result,
        _expected_rotary(
            half_query, cos.to(torch.float16), sin.to(torch.float16), positions, True
        ),
    )
    torch.testing.assert_close(
        half_key_result,
        _expected_rotary(
            half_key, cos.to(torch.float16), sin.to(torch.float16), positions, True
        ),
    )

    with pytest.raises(ValueError, match="equal token counts"):
        reference_rotary.rotary_embedding_torch(
            None, query, key[:1], cos, sin, positions, inplace=False
        )
    with pytest.raises(ValueError, match="dtype int32 or int64"):
        reference_rotary.rotary_embedding_torch(
            None, query, key, cos, sin, positions.float(), inplace=False
        )


def test_ascend_fp32_rotary_fallback_preserves_inplace_contract(monkeypatch):
    original_import = builtins.__import__

    def reject_torch_npu(name, *args, **kwargs):
        if name == _REFERENCE_MODULE:
            return reference_rotary
        if name == "torch_npu":
            raise AssertionError("unsupported inputs must not import torch_npu")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch_npu)
    base_query, base_key, cos, sin, positions = _rotary_inputs()

    for inplace in (False, True):
        query = base_query.clone()
        key = base_key.clone()
        query_before = query.clone()
        key_before = key.clone()
        expected_query = _expected_rotary(query_before, cos, sin, positions, False)
        expected_key = _expected_rotary(key_before, cos, sin, positions, False)

        actual_query, actual_key = ascend_rotary.rotary_embedding_ascend(
            None, query, key, cos, sin, positions, inplace=inplace
        )

        torch.testing.assert_close(actual_query, expected_query)
        torch.testing.assert_close(actual_key, expected_key)
        if inplace:
            assert actual_query is query
            assert actual_key is key
            torch.testing.assert_close(query, expected_query)
            torch.testing.assert_close(key, expected_key)
        else:
            assert actual_query is not query
            assert actual_key is not key
            torch.testing.assert_close(query, query_before)
            torch.testing.assert_close(key, key_before)

    npu_device = SimpleNamespace(type="npu")
    query_meta = SimpleNamespace(
        device=npu_device, dtype=torch.float16, shape=(4, 2, 8), dim=lambda: 3
    )
    key_meta = SimpleNamespace(
        device=npu_device, dtype=torch.float16, shape=(4, 1, 8), dim=lambda: 3
    )
    cache_meta = SimpleNamespace(
        device=npu_device, dtype=torch.float16, shape=(16, 4), dim=lambda: 2
    )
    int64_positions = SimpleNamespace(
        device=npu_device,
        dtype=torch.int64,
        dim=lambda: 1,
        numel=lambda: 4,
        is_contiguous=lambda: True,
    )
    int32_positions = SimpleNamespace(
        device=npu_device,
        dtype=torch.int32,
        dim=lambda: 1,
        numel=lambda: 4,
        is_contiguous=lambda: True,
    )
    assert ascend_rotary._can_use_ascend_rotary_kernel(
        query_meta, key_meta, cache_meta, cache_meta, int64_positions
    )
    assert not ascend_rotary._can_use_ascend_rotary_kernel(
        query_meta, key_meta, cache_meta, cache_meta, int32_positions
    )
    empty_query_meta = SimpleNamespace(
        device=npu_device, dtype=torch.float16, shape=(0, 2, 8), dim=lambda: 3
    )
    empty_positions = SimpleNamespace(
        device=npu_device,
        dtype=torch.int64,
        dim=lambda: 1,
        numel=lambda: 0,
        is_contiguous=lambda: True,
    )
    assert not ascend_rotary._can_use_ascend_rotary_kernel(
        empty_query_meta,
        empty_query_meta,
        cache_meta,
        cache_meta,
        empty_positions,
    )


def test_ascend_rotary_fast_path_preserves_inplace_and_layout(monkeypatch):
    torch_npu = ModuleType("torch_npu")
    calls = []

    def fake_rotary(position_ids, query, key, head_size, cos_sin_cache, is_neox_style):
        calls.append(
            {
                "position_ids": position_ids,
                "query_shape": query.shape,
                "key_shape": key.shape,
                "head_size": head_size,
                "cos_sin_cache": cos_sin_cache.clone(),
                "is_neox_style": is_neox_style,
                "query_contiguous": query.is_contiguous(),
                "key_contiguous": key.is_contiguous(),
            }
        )
        query.add_(2)
        key.sub_(2)

    torch_npu._npu_rotary_embedding = fake_rotary
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setattr(
        ascend_rotary, "_can_use_ascend_rotary_kernel", lambda *args: True
    )
    _, _, cos, sin, positions = _rotary_inputs(torch.float16)

    for contiguous in (False, True):
        for inplace in (False, True):
            query = torch.randn(4, 2, 8, dtype=torch.float16)
            key = torch.randn(4, 2, 8, dtype=torch.float16)
            if not contiguous:
                query = query.transpose(1, 2).contiguous().transpose(1, 2)
                key = key.transpose(1, 2).contiguous().transpose(1, 2)
            query_before = query.clone()
            key_before = key.clone()

            actual_query, actual_key = ascend_rotary.rotary_embedding_ascend(
                None, query, key, cos, sin, positions, inplace=inplace
            )

            torch.testing.assert_close(actual_query, query_before + 2)
            torch.testing.assert_close(actual_key, key_before - 2)
            if inplace:
                assert actual_query is query
                assert actual_key is key
            else:
                assert actual_query is not query
                assert actual_key is not key
                torch.testing.assert_close(query, query_before)
                torch.testing.assert_close(key, key_before)

    assert len(calls) == 4
    for call in calls:
        assert call["position_ids"] is positions
        assert call["query_shape"] == (4, 16)
        assert call["key_shape"] == (4, 16)
        assert call["head_size"] == 8
        torch.testing.assert_close(call["cos_sin_cache"], torch.cat((cos, sin), -1))
        assert call["is_neox_style"] is True
        assert call["query_contiguous"] is True
        assert call["key_contiguous"] is True


def test_flaggems_six_argument_rotary_preserves_inplace_contract(monkeypatch):
    calls = []

    def ascend_apply_rotary_pos_emb(
        query, key, cos, sin, position_ids=None, rotary_interleaved=False
    ):
        calls.append((position_ids, rotary_interleaved))
        return query + 1, key - 1

    def reject_common_wrapper(*args, **kwargs):
        raise AssertionError("the seven-argument wrapper must not run")

    _install_fake_flaggems(
        monkeypatch, ascend_apply_rotary_pos_emb, reject_common_wrapper
    )
    flaggems_rotary._supports_inplace_argument.cache_clear()
    base_query, base_key, cos, sin, positions = _rotary_inputs(torch.bfloat16)

    for inplace in (False, True):
        query = base_query.clone()
        key = base_key.clone()
        actual_query, actual_key = flaggems_rotary.rotary_embedding_flaggems(
            SimpleNamespace(),
            query,
            key,
            cos,
            sin,
            positions,
            rotary_interleaved=False,
            inplace=inplace,
        )

        torch.testing.assert_close(actual_query, base_query + 1)
        torch.testing.assert_close(actual_key, base_key - 1)
        if inplace:
            assert actual_query is query
            assert actual_key is key
        else:
            assert actual_query is not query
            assert actual_key is not key
            torch.testing.assert_close(query, base_query)
            torch.testing.assert_close(key, base_key)

    flaggems_rotary._supports_inplace_argument.cache_clear()
    monkeypatch.setattr(
        flaggems_rotary,
        "inspect",
        SimpleNamespace(
            signature=lambda fn: (_ for _ in ()).throw(ValueError("opaque callable"))
        ),
    )
    flaggems_rotary.rotary_embedding_flaggems(
        SimpleNamespace(), base_query, base_key, cos, sin, positions, inplace=False
    )
    assert len(calls) == 3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU launch contracts for the Sunrise stride-aware cache writer."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def cache_writer(monkeypatch):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    triton = ModuleType("triton")
    triton.jit = lambda fn: Kernel()
    triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    language = ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/dispatch/backends/vendor/sunrise/impl/cache_update.py"
    )
    spec = importlib.util.spec_from_file_location("_sunrise_cache_write_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, calls=calls)


def test_writer_passes_real_combined_cache_and_input_strides(cache_writer):
    key = torch.empty(3, 2, 8)
    value = torch.empty(3, 8, 2).transpose(1, 2)
    logical = torch.empty(2, 2, 16, 16)
    key_cache = logical[..., :8].transpose(1, 2)
    value_cache = logical[..., 8:].transpose(1, 2)
    slots = torch.tensor([3, -1, 17])
    cache_writer.module.write_kv_cache(key, value, key_cache, value_cache, slots)
    grid, args, kwargs = cache_writer.calls[0]
    assert grid == (3, 2)
    assert args[:5] == (key, value, key_cache, value_cache, slots)
    assert args[5:] == (
        *key.stride(),
        *value.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
    )
    assert kwargs == {"HEAD_SIZE": 8, "BLOCK_SIZE": 16, "WIDTH": 8, "num_warps": 4}


def test_empty_slot_mapping_does_not_launch(cache_writer):
    cache_writer.module.write_kv_cache(None, None, None, None, torch.empty(0))
    assert not cache_writer.calls

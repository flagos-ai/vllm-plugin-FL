# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_BRIDGE_PATH = (
    Path(__file__).parents[3] / "vllm_fl" / "patches" / "flaggems_aten_plan_cache.py"
)
_SPEC = spec_from_file_location("vllm_fl_flaggems_aten_plan_cache_test", _BRIDGE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bridge = module_from_spec(_SPEC)
_SPEC.loader.exec_module(bridge)


def _fake_flaggems(monkeypatch, **attributes):
    module = types.ModuleType("flag_gems")
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    return module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLAGGEMS_ATEN_PLAN_CACHE", raising=False)


def test_bridge_is_opt_in(monkeypatch):
    calls = []
    _fake_flaggems(monkeypatch, enable_aten_plan_cache=lambda: calls.append(1))
    assert bridge.apply_flaggems_aten_plan_cache() is False
    assert calls == []


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_disabled_setting_does_not_enable_cache(monkeypatch, value):
    monkeypatch.setenv("FLAGGEMS_ATEN_PLAN_CACHE", value)
    _fake_flaggems(monkeypatch, enable_aten_plan_cache=lambda: pytest.fail("disabled"))
    assert bridge.apply_flaggems_aten_plan_cache() is False


def test_flaggems_setting_uses_public_api(monkeypatch):
    calls = []
    monkeypatch.setenv("FLAGGEMS_ATEN_PLAN_CACHE", "1")
    _fake_flaggems(monkeypatch, enable_aten_plan_cache=lambda: calls.append(1) or True)
    assert bridge.apply_flaggems_aten_plan_cache() is True
    assert calls == [1]


def test_module_enable_api_is_used_when_public_export_is_absent(monkeypatch):
    monkeypatch.setenv("FLAGGEMS_ATEN_PLAN_CACHE", "1")
    _fake_flaggems(monkeypatch)
    module = types.ModuleType("flag_gems.utils.aten_plan_cache")
    module.enable = lambda: True
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert bridge.apply_flaggems_aten_plan_cache() is True


def test_unavailable_api_keeps_normal_routing(monkeypatch):
    monkeypatch.setenv("FLAGGEMS_ATEN_PLAN_CACHE", "1")
    _fake_flaggems(monkeypatch)
    assert bridge.apply_flaggems_aten_plan_cache() is False


def test_enable_failure_propagates(monkeypatch):
    monkeypatch.setenv("FLAGGEMS_ATEN_PLAN_CACHE", "1")

    def fail():
        raise RuntimeError("cache initialization failed")

    _fake_flaggems(monkeypatch, enable_aten_plan_cache=fail)
    with pytest.raises(RuntimeError, match="cache initialization failed"):
        bridge.apply_flaggems_aten_plan_cache()

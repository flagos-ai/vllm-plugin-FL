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
    monkeypatch.delenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", raising=False)
    monkeypatch.delenv("FLAGGEMS_ATEN_PLAN_CACHE", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE", raising=False)


def test_bridge_is_opt_in(monkeypatch):
    calls = []
    _fake_flaggems(monkeypatch, enable_aten_plan_cache=lambda: calls.append(1))

    assert bridge.apply_flaggems_aten_plan_cache() is False
    assert calls == []


def test_plugin_switch_uses_public_api(monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", "1")
    _fake_flaggems(
        monkeypatch,
        enable_aten_plan_cache=lambda: calls.append(1) or True,
        aten_plan_cache_stats=lambda: {
            "enabled": bool(calls),
            "installed": bool(calls),
        },
    )

    assert bridge.apply_flaggems_aten_plan_cache() is True
    assert calls == [1]


def test_plugin_switch_overrides_flaggems_switch(monkeypatch):
    calls = []
    monkeypatch.setenv("FLAGGEMS_ATEN_PLAN_CACHE", "1")
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", "0")
    _fake_flaggems(monkeypatch, enable_aten_plan_cache=lambda: calls.append(1))

    assert bridge.apply_flaggems_aten_plan_cache() is False
    assert calls == []


def test_already_enabled_does_not_reconfigure(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", "1")

    def unexpected_enable():
        raise AssertionError("already-enabled cache must not be cleared")

    _fake_flaggems(
        monkeypatch,
        enable_aten_plan_cache=unexpected_enable,
        aten_plan_cache_stats=lambda: {"enabled": True, "installed": True},
    )

    assert bridge.apply_flaggems_aten_plan_cache() is True


def test_stats_uses_public_api(monkeypatch):
    expected = {"enabled": True, "hits": 7, "misses": 1}
    _fake_flaggems(monkeypatch, aten_plan_cache_stats=lambda: expected)

    assert bridge.get_flaggems_aten_plan_cache_stats() == expected


def test_successful_noop_is_not_reported_active(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", "1")
    _fake_flaggems(
        monkeypatch,
        enable_aten_plan_cache=lambda: True,
        aten_plan_cache_stats=lambda: {"hits": 0, "misses": 0},
    )
    assert bridge.apply_flaggems_aten_plan_cache() is False


def test_required_cache_rejects_disabled_state(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", "1")
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE", "1")
    _fake_flaggems(
        monkeypatch,
        enable_aten_plan_cache=lambda: True,
        aten_plan_cache_stats=lambda: {"enabled": False, "installed": True},
    )
    with pytest.raises(RuntimeError, match="not verified active"):
        bridge.apply_flaggems_aten_plan_cache()


def test_required_cache_rejects_no_warmup_reuse(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", "1")
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE", "1")
    _fake_flaggems(
        monkeypatch,
        aten_plan_cache_stats=lambda: {
            "enabled": True, "installed": True, "hits": 0, "misses": 1,
        },
    )
    with pytest.raises(RuntimeError, match="no demonstrated cache reuse"):
        bridge.log_flaggems_aten_plan_cache_stats("post-warmup")


def test_required_cache_accepts_real_warmup_reuse(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE", "1")
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE", "1")
    _fake_flaggems(
        monkeypatch,
        aten_plan_cache_stats=lambda: {
            "enabled": True, "installed": True, "hits": 3, "misses": 1,
        },
    )
    bridge.log_flaggems_aten_plan_cache_stats("post-warmup")


def test_required_cache_rejects_disabled_request(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE", "1")
    with pytest.raises(RuntimeError, match="required but disabled"):
        bridge.apply_flaggems_aten_plan_cache()

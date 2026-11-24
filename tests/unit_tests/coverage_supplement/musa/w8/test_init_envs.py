# Copyright (c) 2026 FlagOS Contributors
"""Unit tests for ``vllm_fl.envs`` lazy environment accessors."""

from __future__ import annotations

import pytest

import vllm_fl.envs as envs


def test_dir_lists_registered_variables():
    names = dir(envs)
    assert set(names) == {
        "VLLM_FL_PREFER_ENABLED",
        "FLAGGEMS_ENABLE_OPLIST_PATH",
        "USE_FLAGGEMS",
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("True", True),
        ("true", True),
        ("1", True),
        ("False", False),
        ("0", False),
        ("maybe", False),
    ],
)
def test_prefer_enabled(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_FL_PREFER_ENABLED", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", value)
    assert envs.VLLM_FL_PREFER_ENABLED is expected


def test_oplist_path_default_and_override(monkeypatch):
    monkeypatch.delenv("FLAGGEMS_ENABLE_OPLIST_PATH", raising=False)
    assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/tmp/flaggems_enable_oplist.txt"
    monkeypatch.setenv("FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/custom.txt")
    assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/tmp/custom.txt"


def test_use_flaggems_delegates_to_utils(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(
        envs.fl_vllm_environment_variables, "USE_FLAGGEMS", lambda: sentinel
    )
    assert envs.USE_FLAGGEMS is sentinel


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(envs, "NO_SUCH_VARIABLE")


def test_is_set_known_variable(monkeypatch):
    monkeypatch.setenv("VLLM_FL_PREFER_ENABLED", "1")
    assert envs.is_set("VLLM_FL_PREFER_ENABLED") is True
    monkeypatch.delenv("VLLM_FL_PREFER_ENABLED", raising=False)
    assert envs.is_set("VLLM_FL_PREFER_ENABLED") is False


def test_is_set_unknown_variable_raises():
    with pytest.raises(AttributeError):
        envs.is_set("NO_SUCH_VARIABLE")

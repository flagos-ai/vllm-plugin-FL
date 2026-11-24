# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual branch coverage for ``vllm_fl.dispatch.policy``.

Complements w9b ``test_mgr_policy.py`` with config-file string/list
coercion branches and empty platform/default overrides.
"""

import os
import tempfile

import pytest

from vllm_fl.dispatch import config as cfg_mod
from vllm_fl.dispatch import policy as pol
from vllm_fl.dispatch.policy import (
    PREFER_VENDOR,
    policy_from_config,
    policy_from_env,
    reset_global_policy,
)

ENV_KEYS = (
    "VLLM_FL_CONFIG",
    "VLLM_FL_PREFER",
    "VLLM_FL_STRICT",
    "VLLM_FL_DENY_VENDORS",
    "VLLM_FL_ALLOW_VENDORS",
    "VLLM_FL_PER_OP",
)


@pytest.fixture(autouse=True)
def _clean_policy(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    reset_global_policy()
    yield
    reset_global_policy()


def _write(text):
    fh = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    fh.write(text)
    fh.close()
    return fh.name


def test_config_allow_string_and_non_dict_op_backends():
    path = _write(
        'allow_vendors: "CUDA, AMD"\n'
        "deny_vendors:\n"
        "  - AMD\n"
        "  - ''\n"
        'op_backends: "silu=vendor|flagos"\n'
    )
    try:
        p = policy_from_config(path)
    finally:
        os.unlink(path)

    assert p.allow_vendors == frozenset({"CUDA", "AMD"})
    assert p.deny_vendors == frozenset({"AMD"})
    # A truthy-but-not-dict ``op_backends`` is ignored entirely.
    assert p.per_op_order == ()


def test_config_op_backends_list_coerces_and_filters():
    path = _write(
        "op_backends:\n"
        "  coerced:\n"
        "    - 1\n"
        "    - null\n"
        '    - " flagos "\n'
    )
    try:
        p = policy_from_config(path)
    finally:
        os.unlink(path)

    assert p.get_per_op_order("coerced") == ["1", "flagos"]
    assert p.get_per_op_order("other") is None


def test_env_empty_platform_lists_and_falsy_op_config(monkeypatch):
    path = _write(
        "prefer: vendor\n"
        "strict: false\n"
        "deny_vendors: []\n"
        "allow_vendors: []\n"
        "op_backends: {}\n"
    )
    monkeypatch.setattr(cfg_mod, "get_config_path", lambda: path)
    monkeypatch.setattr(pol, "get_op_config", lambda: {})
    try:
        p = policy_from_env()
        assert p.prefer == PREFER_VENDOR
        # Empty platform collections fall back to "unrestricted".
        assert p.deny_vendors == frozenset()
        assert p.allow_vendors is None
        assert p.per_op_order == ()

        # Empty ``op_config`` is falsy, so VLLM_FL_PER_OP wins.
        monkeypatch.setenv("VLLM_FL_PER_OP", "from_env=vendor|reference")
        p = policy_from_env()
        assert p.get_per_op_order("from_env") == ["vendor", "reference"]
    finally:
        os.unlink(path)


def test_env_platform_without_allow_deny_uses_env_overrides(monkeypatch):
    path = _write("prefer: reference\n")
    monkeypatch.setattr(cfg_mod, "get_config_path", lambda: path)
    monkeypatch.setattr(pol, "get_op_config", lambda: None)
    try:
        p = policy_from_env()
        assert p.prefer == "reference"
        assert p.deny_vendors == frozenset()

        monkeypatch.setenv("VLLM_FL_ALLOW_VENDORS", " CUDA , AMD ")
        monkeypatch.setenv("VLLM_FL_DENY_VENDORS", "Intel")
        p = policy_from_env()
        assert p.allow_vendors == frozenset({"CUDA", "AMD"})
        assert p.deny_vendors == frozenset({"Intel"})
    finally:
        os.unlink(path)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""vllm_fl._metax_runtime_env: the MACA dispatch-mode policy (R6/I1).

The module is loaded from its file, so these tests need neither FlagGems nor
a device.
"""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "vllm_fl" / "_metax_runtime_env.py"


def _load():
    spec = importlib.util.spec_from_file_location("_metax_runtime_env_ut", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _maca():
    return "3.7.0.7"


def _not_maca():
    return None


def _must_not_probe():
    raise AssertionError("torch probed although the policy already decided")


def test_default_is_off_and_does_not_probe_torch():
    m = _load()
    env = {}
    assert m.POLICY_DEFAULT == "0"
    assert m.apply(env, _must_not_probe).startswith("off")
    assert env == {} and not m.APPLIED


@pytest.mark.parametrize("value", ["0", "off", "false", "", " 0 "])
def test_explicit_off(value):
    m = _load()
    env = {m.POLICY_ENV: value}
    m.apply(env, _must_not_probe)
    assert m.RUNTIME_ENV not in env and not m.APPLIED


@pytest.mark.parametrize("value", ["1", "on", "true", "YES"])
def test_on_maca_sets_direct_dispatch(value):
    m = _load()
    env = {m.POLICY_ENV: value}
    state = m.apply(env, _maca)
    assert env[m.RUNTIME_ENV] == "1" and m.APPLIED
    assert "3.7.0.7" in state


def test_on_but_not_maca_keeps_env():
    m = _load()
    env = {m.POLICY_ENV: "1"}
    assert m.apply(env, _not_maca) == "off (not a MACA build of PyTorch)"
    assert m.RUNTIME_ENV not in env and not m.APPLIED


@pytest.mark.parametrize("key", ["MACA_DIRECT_DISPATCH", "MACA_EXT_DIRECT_DISPATCH"])
def test_explicit_runtime_setting_wins(key):
    m = _load()
    env = {m.POLICY_ENV: "1", key: "0"}
    assert m.apply(env, _must_not_probe) == f"kept explicit {key}=0"
    assert env[key] == "0" and not m.APPLIED
    if key != m.RUNTIME_ENV:
        assert m.RUNTIME_ENV not in env


@pytest.mark.parametrize(
    "vendor,applied", [("nvidia", False), ("iluvatar", False), ("metax", True), ("MACA", True)]
)
def test_flaggems_vendor_override(vendor, applied):
    m = _load()
    env = {m.POLICY_ENV: "1", "GEMS_VENDOR": vendor}
    m.apply(env, _maca)
    assert (env.get(m.RUNTIME_ENV) == "1") is applied


def test_unknown_policy_value_is_off():
    m = _load()
    env = {m.POLICY_ENV: "maybe"}
    assert m.apply(env, _must_not_probe).startswith("off (unknown")
    assert m.RUNTIME_ENV not in env


def test_probe_error_never_raises():
    m = _load()

    def boom():
        raise RuntimeError("no torch")

    env = {m.POLICY_ENV: "1"}
    assert m.apply(env, boom).startswith("off (error")
    assert m.RUNTIME_ENV not in env and not m.APPLIED

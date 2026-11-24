# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for vllm_fl.worker.runtime FlagGems wiring."""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE_REL = "vllm_fl/worker/runtime.py"


def _source_root():
    candidates = []
    env = os.environ.get("VLLM_FL_SOURCE_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend([Path.cwd(), Path("/tmp/flsrc"), Path(__file__).resolve().parents[2]])
    for root in candidates:
        if (root / SOURCE_REL).is_file():
            return root
    pytest.skip("vllm_fl sources are required", allow_module_level=True)


ROOT = _source_root()
_TREE = ast.parse((ROOT / SOURCE_REL).read_text(encoding="utf-8"))
_FUTURE = ast.ImportFrom(
    module="__future__", names=[ast.alias(name="annotations")], level=0
)
NAMES = {"_initialize_fl_runtime", "_enable_except"}


def _exec_module(namespace):
    selected = [n for n in _TREE.body if getattr(n, "name", None) in NAMES]
    assert {n.name for n in selected} == NAMES
    module = ast.fix_missing_locations(ast.Module(body=[_FUTURE, *selected], type_ignores=[]))
    exec(compile(module, SOURCE_REL, "exec"), namespace)


class _Logger:
    def __init__(self):
        self.messages = []

    def _log(self, level, msg, *args):
        self.messages.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._log("info", msg, *args)

    def error(self, msg, *args):
        self._log("error", msg, *args)

    def warning(self, msg, *args):
        self._log("warning", msg, *args)


class _FlagGems:
    def __init__(self, full_config=None):
        self.calls = []
        self._FULL_CONFIG = full_config if full_config is not None else []

    def only_enable(self, **kwargs):
        self.calls.append(("only_enable", kwargs))

    def enable(self, **kwargs):
        self.calls.append(("enable", kwargs))


class _FlagGemsWithoutConfig(_FlagGems):
    @property
    def _FULL_CONFIG(self):
        raise RuntimeError("no config")

    @_FULL_CONFIG.setter
    def _FULL_CONFIG(self, value):
        pass


@pytest.fixture
def env(monkeypatch):
    namespace = {
        "logger": _Logger(),
        "fl_envs": SimpleNamespace(
            USE_FLAGGEMS=True, FLAGGEMS_ENABLE_OPLIST_PATH="/tmp/oplist"
        ),
        "patch_mm_encoder_attention": lambda: None,
        "get_flag_gems_whitelist_blacklist": lambda: (None, None),
    }
    _exec_module(namespace)

    ops_module = SimpleNamespace(
        register_oot_ops=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "vllm_fl.ops.custom_ops", ops_module)
    patched = []
    namespace["patch_mm_encoder_attention"] = lambda: patched.append("mm")
    namespace["patched"] = patched

    class _Env:
        def __init__(self, backing):
            object.__setattr__(self, "_ns", backing)

        def __getattr__(self, name):
            try:
                return self._ns[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self._ns[name] = value

    return _Env(namespace)


def _install_flag_gems(monkeypatch, flag_gems):
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)


def _install_platform(monkeypatch, vendor):
    module = SimpleNamespace(current_platform=SimpleNamespace(vendor_name=vendor))
    monkeypatch.setitem(sys.modules, "vllm.platforms", module)


def test_disabled_flaggems_only_installs_hooks(env, monkeypatch):
    env.fl_envs = SimpleNamespace(
        USE_FLAGGEMS=False, FLAGGEMS_ENABLE_OPLIST_PATH="/tmp/oplist"
    )
    flag_gems = _FlagGems()
    _install_flag_gems(monkeypatch, flag_gems)
    env._initialize_fl_runtime(0)
    assert env.patched == ["mm"]
    assert flag_gems.calls == []


def test_whitelist_uses_only_enable(env, monkeypatch):
    flag_gems = _FlagGems()
    _install_flag_gems(monkeypatch, flag_gems)
    env.get_flag_gems_whitelist_blacklist = lambda: (["mm", "add"], None)
    env._initialize_fl_runtime(rank=1)
    assert flag_gems.calls == [
        (
            "only_enable",
            {
                "include": ["mm", "add"],
                "record": False,
                "once": True,
                "path": "/tmp/oplist",
            },
        )
    ]


def test_blacklist_uses_enable_unused_for_other_vendors(env, monkeypatch):
    flag_gems = _FlagGems()
    _install_flag_gems(monkeypatch, flag_gems)
    _install_platform(monkeypatch, "npu")
    env.get_flag_gems_whitelist_blacklist = lambda: (None, ["pow.Scalar"])
    env._initialize_fl_runtime(rank=0)
    assert flag_gems.calls == [
        (
            "enable",
            {
                "unused": ["pow.Scalar"],
                "record": True,
                "once": True,
                "path": "/tmp/oplist",
            },
        )
    ]


def test_blacklist_on_amd_derives_explicit_whitelist(env, monkeypatch):
    flag_gems = _FlagGems(
        full_config=[("mm", {}), ("pow.Scalar", {}), ("add", {})]
    )
    _install_flag_gems(monkeypatch, flag_gems)
    _install_platform(monkeypatch, "amd")
    env.get_flag_gems_whitelist_blacklist = lambda: (None, ["pow.Scalar", "missing.op"])
    env._initialize_fl_runtime(rank=0)
    assert flag_gems.calls == [
        (
            "only_enable",
            {
                "include": ["mm", "add"],
                "record": True,
                "once": True,
                "path": "/tmp/oplist",
            },
        )
    ]
    assert any("not present in the operator list" in m for _, m in env.logger.messages)
    assert any("Enabling 2 of 3 ops" in m for _, m in env.logger.messages)


def test_amd_blacklist_falls_back_when_operator_list_unavailable(env, monkeypatch):
    flag_gems = _FlagGemsWithoutConfig()
    _install_flag_gems(monkeypatch, flag_gems)
    _install_platform(monkeypatch, "amd")
    env.get_flag_gems_whitelist_blacklist = lambda: (None, ["pow.Scalar"])
    env._initialize_fl_runtime(rank=0)
    assert flag_gems.calls == [
        (
            "enable",
            {
                "unused": ["pow.Scalar"],
                "record": True,
                "once": True,
                "path": "/tmp/oplist",
            },
        )
    ]
    assert any(level == "error" for level, _ in env.logger.messages)


def test_enable_all_ops(env, monkeypatch):
    flag_gems = _FlagGems()
    _install_flag_gems(monkeypatch, flag_gems)
    env._initialize_fl_runtime(rank=3)
    assert flag_gems.calls == [
        ("enable", {"record": False, "once": True, "path": "/tmp/oplist"})
    ]


def test_enable_except_handles_missing_blacklisted_ops(env):
    flag_gems = _FlagGems(full_config=[("mm", {}), ("add", {})])
    common = {"record": True, "once": True, "path": "/p"}
    env._enable_except(flag_gems, ["bogus"], common)
    assert flag_gems.calls == [("only_enable", dict(include=["mm", "add"], **common))]
    assert any("Blacklisted ops not present" in m for _, m in env.logger.messages)

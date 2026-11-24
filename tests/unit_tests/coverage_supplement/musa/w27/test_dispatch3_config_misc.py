# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual coverage for config load errors, vendor-dir mapping and
``dispatch.__init__`` AMD warm-up gating."""

from types import SimpleNamespace

import pytest

from vllm_fl.dispatch import builtin_ops as b
from vllm_fl.dispatch import config as config_pkg
from vllm_fl.dispatch.config import utils as u
import vllm_fl.dispatch as dp


def test_load_platform_config_unreadable_path_returns_none(monkeypatch, tmp_path):
    directory = tmp_path / "config_dir"
    directory.mkdir()
    monkeypatch.setattr(u, "get_config_path", lambda platform=None: directory)
    # ``open(directory)`` raises IsADirectoryError, which must be swallowed.
    assert u.load_platform_config() is None


def test_effective_config_user_file_open_error_falls_back(monkeypatch):
    monkeypatch.setenv("VLLM_FL_CONFIG", "/some/user/config.yaml")
    monkeypatch.setattr(u.os.path, "isfile", lambda path: True)

    def _deny(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("builtins.open", _deny)
    monkeypatch.setattr(
        u, "load_platform_config", lambda platform=None: {"fallback": True}
    )
    assert u.get_effective_config() == {"fallback": True}


def test_config_package_exports_vendor_map():
    assert config_pkg.get_vendor_device_map is u.get_vendor_device_map


def test_find_vendor_backend_dir_uses_device_name_and_type(monkeypatch):
    monkeypatch.setattr(
        b,
        "get_vendor_device_map",
        lambda: {
            "weird": {"device_type": "tp", "device_name": "nm"},
            "noname": {"device_type": "tp2"},
        },
    )
    assert b._find_vendor_backend_dir("weird", {"weird"}) == "weird"
    assert b._find_vendor_backend_dir("weird", {"nm"}) == "nm"
    assert b._find_vendor_backend_dir("weird", {"tp"}) == "tp"
    # ``device_name`` missing -> skipped, ``device_type`` still resolves.
    assert b._find_vendor_backend_dir("noname", {"tp2"}) == "tp2"
    assert b._find_vendor_backend_dir("noname", {"nope"}) is None


def test_register_vendor_backends_filters_hidden_dirs_and_files(monkeypatch):
    monkeypatch.setattr(
        b.os.path,
        "isdir",
        lambda path: path.endswith(("vendor", "cuda", "_hidden")),
    )
    monkeypatch.setattr(
        b.os, "listdir", lambda path: ["_hidden", "cuda", "README.txt"]
    )
    monkeypatch.setattr(b.os.path, "isfile", lambda path: True)

    captured = {}

    def _current(available_dirs):
        captured["available"] = available_dirs
        return "cuda"

    monkeypatch.setattr(b, "_get_current_vendor_backend_dirs", _current)
    imported = []
    monkeypatch.setattr(
        b.importlib,
        "import_module",
        lambda name, package=None: SimpleNamespace(
            register_builtins=lambda registry: imported.append(name)
        ),
    )

    b._register_vendor_backends(object())

    assert captured["available"] == {"cuda"}
    assert imported == [".backends.vendor.cuda.register_ops"]


class _FakeImpl:
    def __init__(self):
        self.fn = lambda *a, **k: None
        self.impl_id = "reference.fake"


class _FakeMgr:
    def __init__(self):
        self.policy_epoch = 0
        self.initialized = False
        self.recorded = []

    def ensure_initialized(self):
        self.initialized = True

    def _resolve_impl(self, op_name):
        return _FakeImpl()

    def _record_first_use(self, op_name, impl):
        self.recorded.append((op_name, impl))


@pytest.mark.parametrize(
    "fast_path,dump_enabled,expected",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_amd_compile_ready_requires_fast_path_and_no_dump(
    monkeypatch, fast_path, dump_enabled, expected
):
    registry = set()
    monkeypatch.setattr(dp, "_CACHED_OPS", registry)
    op = dp.CachedOp("w27_amd")
    assert op in registry
    monkeypatch.setattr(dp, "_OP_FAST_PATH_ENABLED", fast_path)
    monkeypatch.setattr(dp, "is_dump_enabled", lambda: dump_enabled)

    mgr = _FakeMgr()
    dp.warmup_cached_ops(mgr, amd_compile=True)

    assert mgr.initialized is True
    assert op._impl is not None
    assert op._amd_compile_ready is expected
    assert mgr.recorded and mgr.recorded[0][0] == "w27_amd"


def test_warmup_without_amd_compile_leaves_flag_false(monkeypatch):
    registry = set()
    monkeypatch.setattr(dp, "_CACHED_OPS", registry)
    op = dp.CachedOp("w27_plain")
    monkeypatch.setattr(dp, "_OP_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(dp, "is_dump_enabled", lambda: False)

    dp.warmup_cached_ops(_FakeMgr())

    assert op._impl is not None
    assert op._amd_compile_ready is False

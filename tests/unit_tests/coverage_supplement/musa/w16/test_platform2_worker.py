# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual CPU contracts for worker/runtime.py and worker/nvidia_worker.py."""

import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_fl.worker.nvidia_worker as nw
import vllm_fl.worker.runtime as runtime


def _install(monkeypatch, full, **attrs):
    parts = full.split(".")
    for i in range(1, len(parts)):
        pname = ".".join(parts[:i])
        if pname in sys.modules:
            continue
        pkg = ModuleType(pname)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, pname, pkg)
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            monkeypatch.setattr(parent, parts[i - 1], pkg, raising=False)
    module = ModuleType(full)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, full, module)
    parent_name, _, child = full.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, module, raising=False)
    return module


def _fake_flag_gems(monkeypatch, full_config=None):
    module = ModuleType("flag_gems")
    module.only_enable = Mock()
    module.enable = Mock()
    if full_config is not None:
        module._FULL_CONFIG = full_config
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    return module


def _fake_register(monkeypatch):
    reg = Mock()
    _install(monkeypatch, "vllm_fl.ops.custom_ops", register_oot_ops=reg)
    return reg


def _set_env(monkeypatch, *, use_flaggems, path="/tmp/flaggems_oplist.txt"):
    monkeypatch.setattr(
        runtime,
        "fl_envs",
        SimpleNamespace(
            USE_FLAGGEMS=use_flaggems,
            FLAGGEMS_ENABLE_OPLIST_PATH=path,
        ),
    )


# --------------------------------------------------------------------------- #
# runtime._initialize_fl_runtime
# --------------------------------------------------------------------------- #


def test_initialize_runtime_returns_early_without_flaggems(monkeypatch):
    patch_mm = Mock()
    monkeypatch.setattr(runtime, "patch_mm_encoder_attention", patch_mm)
    reg = _fake_register(monkeypatch)
    _fake_flag_gems(monkeypatch)
    _set_env(monkeypatch, use_flaggems=False)
    runtime._initialize_fl_runtime(3)
    patch_mm.assert_called_once_with()
    reg.assert_called_once_with()


def test_initialize_runtime_whitelist_uses_only_enable(monkeypatch):
    monkeypatch.setattr(runtime, "patch_mm_encoder_attention", Mock())
    _fake_register(monkeypatch)
    gems = _fake_flag_gems(monkeypatch)
    monkeypatch.setattr(
        runtime, "get_flag_gems_whitelist_blacklist", lambda: (["rms_norm"], None)
    )
    _set_env(monkeypatch, use_flaggems=True, path="/tmp/x.txt")
    runtime._initialize_fl_runtime(1)
    gems.only_enable.assert_called_once_with(
        include=["rms_norm"], record=False, once=True, path="/tmp/x.txt"
    )
    gems.enable.assert_not_called()


def test_initialize_runtime_blacklist_non_amd_uses_unused(monkeypatch):
    monkeypatch.setattr(runtime, "patch_mm_encoder_attention", Mock())
    _fake_register(monkeypatch)
    gems = _fake_flag_gems(monkeypatch)
    monkeypatch.setattr(
        runtime, "get_flag_gems_whitelist_blacklist", lambda: (None, ["pow"])
    )
    import vllm.platforms as vp

    monkeypatch.setattr(
        vp, "current_platform", SimpleNamespace(vendor_name="nvidia")
    )
    _set_env(monkeypatch, use_flaggems=True)
    runtime._initialize_fl_runtime(0)
    gems.enable.assert_called_once_with(
        unused=["pow"], record=True, once=True, path="/tmp/flaggems_oplist.txt"
    )


def test_initialize_runtime_blacklist_amd_derives_whitelist(monkeypatch):
    monkeypatch.setattr(runtime, "patch_mm_encoder_attention", Mock())
    _fake_register(monkeypatch)
    gems = _fake_flag_gems(
        monkeypatch, full_config=(("mm", 1), ("pow", 2), ("rms_norm", 3))
    )
    monkeypatch.setattr(
        runtime, "get_flag_gems_whitelist_blacklist", lambda: (None, ["pow", "zzz"])
    )
    import vllm.platforms as vp

    monkeypatch.setattr(vp, "current_platform", SimpleNamespace(vendor_name="amd"))
    _set_env(monkeypatch, use_flaggems=True)
    runtime._initialize_fl_runtime(0)
    gems.only_enable.assert_called_once()
    include = gems.only_enable.call_args.kwargs["include"]
    assert "pow" not in include
    assert include == ["mm", "rms_norm"]
    gems.enable.assert_not_called()


def test_initialize_runtime_enables_all_when_no_restrictions(monkeypatch):
    monkeypatch.setattr(runtime, "patch_mm_encoder_attention", Mock())
    _fake_register(monkeypatch)
    gems = _fake_flag_gems(monkeypatch)
    monkeypatch.setattr(
        runtime, "get_flag_gems_whitelist_blacklist", lambda: (None, None)
    )
    _set_env(monkeypatch, use_flaggems=True)
    runtime._initialize_fl_runtime(0)
    gems.enable.assert_called_once_with(
        record=True, once=True, path="/tmp/flaggems_oplist.txt"
    )


def test_enable_except_falls_back_when_full_config_missing(monkeypatch):
    gems = _fake_flag_gems(monkeypatch)
    runtime._enable_except(
        gems, ["pow"], {"record": True, "once": True, "path": "/tmp/x"}
    )
    gems.enable.assert_called_once_with(
        unused=["pow"], record=True, once=True, path="/tmp/x"
    )
    gems.only_enable.assert_not_called()


# --------------------------------------------------------------------------- #
# nvidia_worker helpers
# --------------------------------------------------------------------------- #


def test_nvidia_worker_init_installs_runtime(monkeypatch):
    def fake_init(self, *args, **kwargs):
        self.rank = 9

    monkeypatch.setattr(nw.NativeGPUWorker, "__init__", fake_init)
    install = Mock()
    monkeypatch.setattr(nw, "_initialize_fl_runtime", install)
    worker = nw.NvidiaWorkerFL()
    assert worker.rank == 9
    assert worker._fl_io_dump_initialized is False
    install.assert_called_once_with(9)


def test_install_native_runner_methods_skips_missing_and_managed():
    runner = SimpleNamespace()

    def managed(self):
        return "x"

    managed._vllm_fl_managed_io = True
    runner.execute_model = MethodType(managed, runner)
    # sample_tokens / pool are absent: skipped without error.
    nw._install_native_runner_io_methods(runner)
    assert runner.execute_model() == "x"


def test_install_native_runner_methods_raises_without_wrapper():
    class Runner:
        def execute_model(self):
            return "plain"

    with pytest.raises(RuntimeError, match="no longer exposes"):
        nw._install_native_runner_io_methods(Runner())


def test_call_native_worker_requires_wrapper():
    with pytest.raises(RuntimeError, match="no longer exposes"):
        nw._call_native_worker_with_managed_inference(lambda self: None, object())


def test_call_native_worker_runs_unwrapped_impl():
    def wrapper(self, value):
        return value * 2

    wrapper.__wrapped__ = lambda self, value: value * 2
    assert nw._call_native_worker_with_managed_inference(wrapper, object(), 4) == 8


def _patch_native_method(monkeypatch, name, raw):
    def fn(self, arg):
        return raw(self, arg)

    fn.__wrapped__ = raw
    monkeypatch.setattr(nw.NativeGPUWorker, name, fn, raising=False)


def test_sample_tokens_advances_io_step_on_output(monkeypatch):
    advance = Mock()
    _install(monkeypatch, "vllm_fl.dispatch.io_dumper", advance_io_step=advance)
    _patch_native_method(monkeypatch, "sample_tokens", lambda self, g: "out")
    worker = object.__new__(nw.NvidiaWorkerFL)
    worker._fl_io_dump_initialized = True
    assert worker.sample_tokens("g") == "out"
    advance.assert_called_once_with()


def test_execute_model_skips_advance_on_none(monkeypatch):
    advance = Mock()
    _install(monkeypatch, "vllm_fl.dispatch.io_dumper", advance_io_step=advance)
    _patch_native_method(monkeypatch, "execute_model", lambda self, s: None)
    worker = object.__new__(nw.NvidiaWorkerFL)
    worker._fl_io_dump_initialized = True
    assert worker.execute_model("s") is None
    advance.assert_not_called()

# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tail coverage for io_dumper config/OSError/hook branches + GDN patch.

CPU-only, no kernels, no distributed.  ``gdn_packed_decode`` is imported with
minimal stubbed ``vllm`` modules when the real package is unavailable so that
its module body and pure helper functions can be exercised.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from vllm_fl.dispatch import io_common, io_dumper  # noqa: E402


# --------------------------------------------------------------------------- #
# Isolation: keep global dumper state and env from leaking across test files
# --------------------------------------------------------------------------- #

_ENV_KEYS_PREFIX = "VLLM_FL_IO_DUMP"

_GLOBAL_DEFAULTS = {
    "_enabled": False,
    "_dump_dir": "",
    "_match_all": False,
    "_op_filter": set(),
    "_module_filter": set(),
    "_layer_filter": set(),
    "_max_calls": 0,
    "_step_range": None,
    "_with_values": False,
    "_with_metas": False,
    "_print_enabled": False,
    "_torch_funcs_enabled": False,
    "_torch_func_filter": set(),
    "_rank_filter": None,
    "_bg_workers": 8,
    "_hooks_activated": False,
    "_call_counters": {},
    "_op_summary": {},
    "_io_executor": None,
    "_open_file_handles": {},
    "_pending_futures": [],
    "_module_hook_handles": [],
}


def _close_resources():
    d = io_dumper
    try:
        d._close_open_files()
    except Exception:
        pass
    exec_ = d._io_executor
    d._io_executor = None
    if exec_ is not None and not getattr(exec_, "_shutdown", False):
        try:
            exec_.shutdown(wait=False)
        except Exception:
            pass
    with d._pending_lock:
        d._pending_futures.clear()


def _env_snapshot():
    return {
        k: v
        for k, v in os.environ.items()
        if k.startswith(_ENV_KEYS_PREFIX) or k == "VLLM_FL_CONFIG"
    }


def _env_clear_and_restore(saved):
    for k in list(os.environ):
        if k.startswith(_ENV_KEYS_PREFIX) or k == "VLLM_FL_CONFIG":
            del os.environ[k]
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _isolate_dumper(monkeypatch):
    saved_globals = {
        k: (set(v) if isinstance(v, set) else dict(v) if isinstance(v, dict) else v)
        for k, v in ((k, getattr(io_dumper, k)) for k in _GLOBAL_DEFAULTS)
    }
    saved_active = io_common.is_io_active()
    saved_env = _env_snapshot()

    _close_resources()
    for k, v in _GLOBAL_DEFAULTS.items():
        setattr(io_dumper, k, set(v) if isinstance(v, set) else
                dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
    _env_clear_and_restore({})

    try:
        yield
    finally:
        _close_resources()
        for k, v in saved_globals.items():
            setattr(io_dumper, k, v)
        io_common.set_io_active(saved_active)
        _env_clear_and_restore(saved_env)


def _ns(**kw):
    return types.SimpleNamespace(**kw)


# =========================================================================== #
# io_dumper._init_from_env  (config combinations + OSError)
# =========================================================================== #

def test_init_from_env_early_return_when_enabled(monkeypatch):
    monkeypatch.setattr(io_dumper, "_enabled", True)
    monkeypatch.setattr(io_dumper, "_dump_dir", "/keep")
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    io_dumper._init_from_env()
    assert io_dumper._enabled is True
    assert io_dumper._dump_dir == "/keep"


def test_init_from_env_disabled_without_config(monkeypatch):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    io_dumper._init_from_env()
    assert io_dumper._enabled is False
    assert io_dumper._dump_dir == ""


def test_init_from_env_explicit_zero_resets(monkeypatch):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setenv("VLLM_FL_IO_DUMP", "0")
    io_dumper._init_from_env()
    assert io_dumper._enabled is False
    assert io_dumper._dump_dir == ""


def test_init_from_env_auto_enable_via_filters(monkeypatch, tmp_path):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setattr(io_dumper.os, "getcwd", lambda: str(tmp_path))
    monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "0-1")
    io_dumper._init_from_env()
    assert io_dumper._enabled is True
    assert io_dumper._step_range == (0, 2)
    assert io_dumper._dump_dir == os.path.join(str(tmp_path), "io_dump")


def test_init_from_env_yaml_all_sections(monkeypatch, tmp_path):
    yaml = pytest.importorskip("yaml")  # noqa: F841
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    ydir = tmp_path / "yd"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "io_dump:\n"
        f"  dir: '{ydir}'\n"
        "  ops: [rms_norm, silu_and_mul]\n"
        "  modules: [Linear]\n"
        "  layers: [0, 1-2]\n"
        "  max_calls: 7\n"
        "  step_range: '3-5'\n"
        "  with_torch_funcs: [matmul, softmax]\n"
        "  ranks: [0, 2]\n"
        "  with_values: true\n"
        "  with_metas: true\n"
        "  with_print: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
    io_dumper._init_from_env()
    assert io_dumper._enabled is True
    assert io_dumper._dump_dir == str(ydir)
    assert io_dumper._op_filter == {"rms_norm", "silu_and_mul"}
    assert io_dumper._module_filter == {"Linear"}
    assert "model.layers.0" in io_dumper._layer_filter
    assert "model.layers.1" in io_dumper._layer_filter
    assert io_dumper._max_calls == 7
    assert io_dumper._step_range == (3, 6)
    assert io_dumper._torch_funcs_enabled is True
    assert io_dumper._torch_func_filter == {"matmul", "softmax"}
    assert io_dumper._rank_filter == {0, 2}
    assert io_dumper._with_values is True
    assert io_dumper._with_metas is True
    assert io_dumper._print_enabled is True
    assert (ydir).is_dir()


def test_init_from_env_print_only(monkeypatch):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")
    io_dumper._init_from_env()
    assert io_dumper._enabled is True
    assert io_dumper._print_enabled is True
    assert io_dumper._dump_dir == ""


def test_init_from_env_print_disabled_literal(monkeypatch, tmp_path):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "d"))
    monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "0")
    io_dumper._init_from_env()
    assert io_dumper._enabled is True
    assert io_dumper._print_enabled is False


def test_init_from_env_all_env_vars(monkeypatch, tmp_path):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "envdump"))
    monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "a,b")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_MODULES", "Linear")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_LAYERS", "0")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "4")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "0-2")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS", "matmul")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_RANK", "0,1")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_VALUES", "1")
    monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_METAS", "true")
    io_dumper._init_from_env()
    assert io_dumper._op_filter == {"a", "b"}
    assert io_dumper._module_filter == {"Linear"}
    assert "model.layers.0" in io_dumper._layer_filter
    assert io_dumper._max_calls == 4
    assert io_dumper._step_range == (0, 3)
    assert io_dumper._torch_funcs_enabled is True
    assert io_dumper._torch_func_filter == {"matmul"}
    assert io_dumper._rank_filter == {0, 1}
    assert io_dumper._with_values is True
    assert io_dumper._with_metas is True


def test_init_from_env_bad_int_env_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "d"))
    monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "not-int")
    io_dumper._init_from_env()
    assert io_dumper._max_calls == 0


def test_init_from_env_makedirs_oserror_no_print(monkeypatch, tmp_path):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "boom"))

    def _raise(*a, **k):
        raise OSError("no space")

    monkeypatch.setattr(io_dumper.os, "makedirs", _raise)
    io_dumper._init_from_env()
    assert io_dumper._enabled is False
    assert io_dumper._dump_dir == ""


def test_init_from_env_makedirs_oserror_print_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(io_dumper, "_deactivate_hooks", lambda: None)
    monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "boom"))
    monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "1")

    def _raise(*a, **k):
        raise OSError("no space")

    monkeypatch.setattr(io_dumper.os, "makedirs", _raise)
    io_dumper._init_from_env()
    assert io_dumper._enabled is True
    assert io_dumper._dump_dir == ""
    assert io_dumper._print_enabled is True


# =========================================================================== #
# Exception / OSError in helpers
# =========================================================================== #

class _BadFh:
    def flush(self):
        raise OSError("flush failed")

    def close(self):
        raise OSError("close failed")


def test_close_open_files_swallows_oserror():
    io_dumper._open_file_handles["/nope.json"] = io_dumper._WipHandle(_BadFh())
    io_dumper._close_open_files()
    assert io_dumper._open_file_handles == {}


def test_reset_state_closes_bad_handle_and_shuts_executor():
    class _Exec:
        _shutdown = False

        def shutdown(self, wait=False):
            self._shutdown = True

    exec_ = _Exec()
    io_dumper._io_executor = exec_
    io_dumper._open_file_handles["/bad.json"] = io_dumper._WipHandle(_BadFh())
    io_dumper._reset_state()
    assert exec_._shutdown is True
    assert io_dumper._io_executor is None
    assert io_dumper._open_file_handles == {}
    assert io_dumper._pending_futures == []


def test_dump_output_exception_is_swallowed(monkeypatch):
    def _raise(_name):
        raise RuntimeError("pop failed")

    monkeypatch.setattr(io_dumper, "_pop_pairing", _raise)
    io_dumper._dump_output("op", torch.ones(1))  # must not raise


def test_record_device_event_event_raises_and_missing(monkeypatch):
    class _Ev:
        def record(self):
            raise RuntimeError("no stream")

    class _ModWithEvent:
        @staticmethod
        def Event():
            return _Ev()

    monkeypatch.setattr(torch, "_covaccel_a", _ModWithEvent(), raising=False)
    t1 = types.SimpleNamespace(device=types.SimpleNamespace(type="_covaccel_a"))
    assert io_dumper._record_device_event({"t": t1}) is None

    monkeypatch.setattr(torch, "_covaccel_b", types.SimpleNamespace(), raising=False)
    t2 = types.SimpleNamespace(device=types.SimpleNamespace(type="_covaccel_b"))
    assert io_dumper._record_device_event({"t": t2}) is None


def test_is_flaggems_op_no_called_op(monkeypatch):
    from vllm_fl.dispatch import manager as manager_mod

    fake = types.SimpleNamespace(_called_ops={}, _registry=None)
    monkeypatch.setattr(manager_mod, "get_default_manager", lambda: fake)
    assert io_dumper._is_flaggems_op("op", "") is False


# =========================================================================== #
# Hook lifecycle boundaries
# =========================================================================== #

def _fake_mode_mgr(entered, events=None):
    events = events if events is not None else []

    class _M:
        def is_entered(self, name):
            return entered

        def enter(self, name, inst):
            events.append(("enter", name))

        def request_exit(self, name):
            events.append(("exit", name))

    return _M(), events


def test_on_step_advance_lazy_activation_boundaries(monkeypatch):
    entered = []
    monkeypatch.setattr(io_dumper, "_enter_dispatch_modes", lambda: entered.append(1))
    monkeypatch.setattr(io_dumper, "_step_range", (2, 5))
    monkeypatch.setattr(io_dumper, "_dump_dir", "")
    monkeypatch.setattr(io_dumper, "_print_enabled", False)
    mgr, _ = _fake_mode_mgr(False)
    monkeypatch.setattr(io_dumper, "dispatch_mode_mgr", mgr)

    # next_step (1) below start -> no enter
    io_dumper._on_step_advance(0, set(), set())
    # next_step (6) at/above end -> no enter
    io_dumper._on_step_advance(5, set(), set())
    assert entered == []

    # already entered -> no enter
    mgr2, _ = _fake_mode_mgr(True)
    monkeypatch.setattr(io_dumper, "dispatch_mode_mgr", mgr2)
    io_dumper._on_step_advance(2, set(), set())
    assert entered == []


def test_activate_hooks_immediate_with_zero_start(monkeypatch):
    entered = []
    monkeypatch.setattr(io_dumper, "_enter_dispatch_modes", lambda: entered.append(1))
    monkeypatch.setattr(io_dumper, "register_step_callback", lambda cb: None)
    monkeypatch.setattr(io_dumper, "unregister_step_callback", lambda cb: None)
    monkeypatch.setattr(io_dumper, "_step_range", (0, 3))
    io_dumper._hooks_activated = False
    io_dumper._activate_hooks()
    assert entered == [1]
    assert io_dumper._hooks_activated is True
    io_dumper._deactivate_hooks()
    assert io_dumper._hooks_activated is False


def test_resume_dispatch_modes_skips_when_entered(monkeypatch):
    entered = []
    mgr, _ = _fake_mode_mgr(True)
    monkeypatch.setattr(io_dumper, "dispatch_mode_mgr", mgr)
    monkeypatch.setattr(io_dumper, "_enabled", True)
    monkeypatch.setattr(io_dumper, "_hooks_activated", True)
    monkeypatch.setattr(io_dumper, "_enter_dispatch_modes", lambda: entered.append(1))
    io_dumper.resume_dispatch_modes()
    assert entered == []


def test_enter_dispatch_modes_respects_availability_flags(monkeypatch):
    if getattr(io_dumper, "_DumpDispatchMode", None) is None:
        pytest.skip("TorchDispatchMode class unavailable")
    if getattr(io_dumper, "_DumpTorchFuncMode", None) is None:
        pytest.skip("TorchFunctionMode class unavailable")
    events = []
    fake, _ = _fake_mode_mgr(False, events)
    monkeypatch.setattr(io_dumper, "dispatch_mode_mgr", fake)
    monkeypatch.setattr(io_dumper, "func_mode_mgr", fake)

    # no dispatch mode, funcs disabled
    monkeypatch.setattr(io_dumper, "HAS_TORCH_DISPATCH_MODE", False)
    monkeypatch.setattr(io_dumper, "HAS_TORCH_FUNC_MODE", True)
    monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", False)
    io_dumper._enter_dispatch_modes()
    assert events == []

    # no dispatch mode, funcs enabled -> only func mode
    monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
    io_dumper._enter_dispatch_modes()
    assert events == [("enter", "dump")]

    events.clear()
    # dispatch available, func mode unavailable
    monkeypatch.setattr(io_dumper, "HAS_TORCH_DISPATCH_MODE", True)
    monkeypatch.setattr(io_dumper, "HAS_TORCH_FUNC_MODE", False)
    io_dumper._enter_dispatch_modes()
    assert events == [("enter", "dump")]


def test_should_dump_torch_func_module_match_true(monkeypatch):
    monkeypatch.setattr(io_dumper, "_step_range", None)
    monkeypatch.setattr(io_dumper, "_layer_filter", set())
    monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
    monkeypatch.setattr(io_dumper, "_torch_func_filter", set())
    monkeypatch.setattr(io_dumper, "_match_all", False)
    monkeypatch.setattr(io_dumper, "_op_filter", set())
    monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
    monkeypatch.setattr(io_dumper, "should_inspect_torch_func", lambda *a, **k: True)
    monkeypatch.setattr(io_dumper, "module_context_matches", lambda s: True)
    assert io_dumper._should_dump_torch_func("matmul") is True


def test_register_module_hooks_idempotent(monkeypatch):
    monkeypatch.setattr(io_dumper, "_enabled", True)
    monkeypatch.setattr(io_dumper, "register_module_paths", lambda model: None)
    model = torch.nn.Linear(2, 2)
    io_dumper.register_io_module_hooks(model)
    first = list(io_dumper._module_hook_handles)
    assert first
    io_dumper.register_io_module_hooks(model)  # removes previous first
    assert len(io_dumper._module_hook_handles) == len(first)
    io_dumper._remove_module_hooks()
    assert io_dumper._module_hook_handles == []


def test_get_or_open_handle_reuses(monkeypatch, tmp_path):
    path = str(tmp_path / "reuse.json")
    h1 = io_dumper._get_or_open_handle(path)
    h2 = io_dumper._get_or_open_handle(path)
    assert h1 is h2
    io_dumper._close_open_files()


# =========================================================================== #
# enable_io_dump tail config branches
# =========================================================================== #

def _no_activate(monkeypatch):
    monkeypatch.setattr(io_dumper, "_activate_hooks", lambda: None)


def test_enable_layers_string(monkeypatch, tmp_path):
    _no_activate(monkeypatch)
    io_dumper.enable_io_dump(str(tmp_path / "d"), layers="0")
    assert "model.layers.0" in io_dumper._layer_filter


def test_enable_step_range_tuple(monkeypatch, tmp_path):
    _no_activate(monkeypatch)
    io_dumper.enable_io_dump(str(tmp_path / "d"), step_range=(1, 3))
    assert io_dumper._step_range == (1, 3)


def test_enable_env_dir_literal_one(monkeypatch, tmp_path):
    _no_activate(monkeypatch)
    monkeypatch.setattr(io_dumper.os, "getcwd", lambda: str(tmp_path))
    monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
    io_dumper.enable_io_dump()
    assert io_dumper._dump_dir == os.path.join(str(tmp_path), "io_dump")


def test_enable_env_print_false_literal(monkeypatch, tmp_path):
    _no_activate(monkeypatch)
    monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "d"))
    monkeypatch.setenv("VLLM_FL_IO_DUMP_WITH_PRINT", "0")
    io_dumper.enable_io_dump()
    assert io_dumper._print_enabled is False


def test_enable_empty_dump_dir_uses_cwd(monkeypatch, tmp_path):
    _no_activate(monkeypatch)
    monkeypatch.setattr(io_dumper.os, "getcwd", lambda: str(tmp_path))
    io_dumper.enable_io_dump(dump_dir="")
    assert io_dumper._dump_dir == os.path.join(str(tmp_path), "io_dump")


def test_enable_yaml_torch_funcs_ranks_flags(monkeypatch, tmp_path):
    yaml = pytest.importorskip("yaml")  # noqa: F841
    _no_activate(monkeypatch)
    ydir = tmp_path / "yd"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "io_dump:\n"
        f"  dir: '{ydir}'\n"
        "  with_torch_funcs: [matmul, softmax]\n"
        "  ranks: [0, 2]\n"
        "  with_values: false\n"
        "  with_metas: true\n"
        "  with_print: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
    io_dumper.enable_io_dump()
    assert io_dumper._torch_funcs_enabled is True
    assert io_dumper._torch_func_filter == {"matmul", "softmax"}
    assert io_dumper._rank_filter == {0, 2}
    assert io_dumper._with_values is False
    assert io_dumper._with_metas is True
    assert io_dumper._print_enabled is False


# =========================================================================== #
# Dispatch / function mode extra branches
# =========================================================================== #

_DISP_AVAIL = getattr(io_dumper, "HAS_TORCH_DISPATCH_MODE", False)
_FUNC_AVAIL = getattr(io_dumper, "HAS_TORCH_FUNC_MODE", False)


def _make_dispatch_op(name, result):
    def fn(*a, **k):
        return result

    fn.name = lambda: name
    fn.namespace = "aten"
    return fn


@pytest.mark.skipif(not _DISP_AVAIL, reason="TorchDispatchMode unavailable")
def test_dispatch_mode_module_filter_rejects(monkeypatch):
    mode = io_dumper._DumpDispatchMode()
    io_dumper.set_guard(False)
    monkeypatch.setattr(io_dumper, "_enabled", True)
    monkeypatch.setattr(io_dumper, "_rank_filter", None)
    monkeypatch.setattr(io_dumper, "_match_all", False)
    monkeypatch.setattr(io_dumper, "_op_filter", set())
    monkeypatch.setattr(io_dumper, "_with_metas", True)
    monkeypatch.setattr(io_dumper, "_step_range", None)
    monkeypatch.setattr(io_dumper, "_layer_filter", set())
    monkeypatch.setattr(io_dumper, "_module_filter", {"ZZZ"})
    monkeypatch.setattr(io_dumper, "_dump_dir", "")
    monkeypatch.setattr(io_dumper, "should_inspect_dispatch_op", lambda *a, **k: True)
    monkeypatch.setattr(io_dumper, "get_dispatch_op_name", lambda f: "cov_op")
    monkeypatch.setattr(io_dumper, "get_dispatch_op_namespace", lambda f: "aten")
    monkeypatch.setattr(io_dumper, "module_context_matches", lambda s: False)
    result = mode.__torch_dispatch__(_make_dispatch_op("cov_op", "R"), (), (), {})
    assert result == "R"


@pytest.mark.skipif(not _DISP_AVAIL, reason="TorchDispatchMode unavailable")
def test_dispatch_mode_check_limits_rejects(monkeypatch):
    mode = io_dumper._DumpDispatchMode()
    io_dumper.set_guard(False)
    monkeypatch.setattr(io_dumper, "_enabled", True)
    monkeypatch.setattr(io_dumper, "_rank_filter", None)
    monkeypatch.setattr(io_dumper, "_match_all", True)
    monkeypatch.setattr(io_dumper, "_op_filter", set())
    monkeypatch.setattr(io_dumper, "_with_metas", True)
    monkeypatch.setattr(io_dumper, "_step_range", None)
    monkeypatch.setattr(io_dumper, "_layer_filter", set())
    monkeypatch.setattr(io_dumper, "_module_filter", set())
    monkeypatch.setattr(io_dumper, "_dump_dir", "")
    monkeypatch.setattr(io_dumper, "should_inspect_dispatch_op", lambda *a, **k: True)
    monkeypatch.setattr(io_dumper, "get_dispatch_op_name", lambda f: "cov2")
    monkeypatch.setattr(io_dumper, "get_dispatch_op_namespace", lambda f: "aten")
    monkeypatch.setattr(io_dumper, "_check_limits", lambda name: False)
    result = mode.__torch_dispatch__(_make_dispatch_op("cov2", "R2"), (), (), {})
    assert result == "R2"


@pytest.mark.skipif(not _FUNC_AVAIL, reason="TorchFunctionMode unavailable")
def test_torch_func_mode_compiling_short_circuit(monkeypatch):
    mode = io_dumper._DumpTorchFuncMode()
    monkeypatch.setattr(
        torch, "compiler", types.SimpleNamespace(is_compiling=lambda: True), raising=False
    )

    def fn(*a, **k):
        return "C"

    assert mode.__torch_function__(fn, (), (), {}) == "C"


# =========================================================================== #
# GDN packed_decode patch
# =========================================================================== #

_GDN_NAME = "vllm_fl.patches.gdn_packed_decode"
_GDN = None


def _build_stub_vllm():
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    third = types.ModuleType("vllm.third_party")
    third.__path__ = []
    fla = types.ModuleType("vllm.third_party.flash_linear_attention")
    fla.__path__ = []
    ops = types.ModuleType("vllm.third_party.flash_linear_attention.ops")
    ops.__path__ = []
    op = types.ModuleType("vllm.third_party.flash_linear_attention.ops.op")
    op.exp = lambda x: x

    def _jit(fn):
        return fn

    tl = types.SimpleNamespace(constexpr=object)
    triton = types.SimpleNamespace(jit=_jit)
    tu = types.ModuleType("vllm.triton_utils")
    tu.tl = tl
    tu.triton = triton

    vllm.third_party = third
    third.flash_linear_attention = fla
    fla.ops = ops
    ops.op = op

    return {
        "vllm": vllm,
        "vllm.third_party": third,
        "vllm.third_party.flash_linear_attention": fla,
        "vllm.third_party.flash_linear_attention.ops": ops,
        "vllm.third_party.flash_linear_attention.ops.op": op,
        "vllm.triton_utils": tu,
    }


def _load_gdn():
    global _GDN
    if _GDN is not None:
        return _GDN
    try:
        _GDN = importlib.import_module(_GDN_NAME)
        return _GDN
    except Exception:
        pass

    stubs = _build_stub_vllm()
    sentinel = object()
    saved = {name: sys.modules.get(name, sentinel) for name in stubs}
    try:
        sys.modules.update(stubs)
        _GDN = importlib.import_module(_GDN_NAME)
    except Exception:
        _GDN = None
    finally:
        for name, old in saved.items():
            if old is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return _GDN


@pytest.fixture
def gdn():
    mod = _load_gdn()
    if mod is None:
        pytest.skip("gdn_packed_decode not importable in this environment")
    return mod


def test_gdn_import_and_marker(gdn):
    replacement = gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    assert getattr(replacement, "_fl_fp32_beta", False) is True
    assert "tl.sigmoid(b_val)" in gdn._VULNERABLE_BETA_EXPRESSION


def test_gdn_kernel_needs_patch_already_marked(gdn):
    assert gdn._kernel_needs_beta_patch(_ns(_fl_fp32_beta=True)) is False


def test_gdn_kernel_needs_patch_vulnerable_with_fn(gdn, monkeypatch):
    monkeypatch.setattr(
        gdn.inspect, "getsource", lambda obj: gdn._VULNERABLE_BETA_EXPRESSION
    )
    assert gdn._kernel_needs_beta_patch(_ns(fn=lambda: None)) is True


def test_gdn_kernel_needs_patch_vulnerable_plain_object(gdn, monkeypatch):
    monkeypatch.setattr(
        gdn.inspect, "getsource", lambda obj: "beta = " + gdn._VULNERABLE_BETA_EXPRESSION
    )
    assert gdn._kernel_needs_beta_patch(object()) is True


def test_gdn_kernel_needs_patch_fixed_source(gdn, monkeypatch):
    monkeypatch.setattr(gdn.inspect, "getsource", lambda obj: "beta = tl.sigmoid(b_val)")
    assert gdn._kernel_needs_beta_patch(_ns(fn=lambda: None)) is False


@pytest.mark.parametrize("exc", [OSError("missing"), TypeError("bad")])
def test_gdn_kernel_needs_patch_inspection_failure(gdn, monkeypatch, exc):
    def _raise(obj):
        raise exc

    monkeypatch.setattr(gdn.inspect, "getsource", _raise)
    assert gdn._kernel_needs_beta_patch(_ns(fn=lambda: None)) is True


def test_gdn_patch_applied_then_idempotent(gdn, monkeypatch):
    target = _ns(**{gdn._TARGET_NAME: lambda *a, **k: None})
    monkeypatch.setattr(gdn.importlib, "import_module", lambda name: target)
    monkeypatch.setattr(
        gdn.inspect, "getsource", lambda obj: gdn._VULNERABLE_BETA_EXPRESSION
    )
    first = gdn.patch_vllm_packed_gdn_beta()
    assert first is True
    replacement = getattr(target, gdn._TARGET_NAME)
    assert replacement is gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    # Now the symbol already carries the marker -> no-op.
    assert gdn.patch_vllm_packed_gdn_beta() is False


def test_gdn_patch_import_error(gdn, monkeypatch):
    def _boom(name):
        raise ImportError("no fla")

    monkeypatch.setattr(gdn.importlib, "import_module", _boom)
    assert gdn.patch_vllm_packed_gdn_beta() is False


def test_gdn_patch_missing_symbol(gdn, monkeypatch):
    monkeypatch.setattr(gdn.importlib, "import_module", lambda name: _ns())
    assert gdn.patch_vllm_packed_gdn_beta() is False


def test_gdn_patch_not_needed(gdn, monkeypatch):
    target = _ns(**{gdn._TARGET_NAME: lambda *a, **k: None})
    monkeypatch.setattr(gdn.importlib, "import_module", lambda name: target)
    monkeypatch.setattr(gdn.inspect, "getsource", lambda obj: "beta = tl.sigmoid(b_val)")
    assert gdn.patch_vllm_packed_gdn_beta() is False

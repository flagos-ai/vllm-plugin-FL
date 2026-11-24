# Copyright (c) 2026 BAAI. All rights reserved.

"""Residual branch coverage for ``vllm_fl.dispatch.io_dumper``.

Targets the public entry points that w1/w2b do not exercise:
``init_io_dump_from_env``, ``dump_before``, ``dump_after`` and the
``__torch_dispatch__`` / ``__torch_function__`` handlers of the two
interception modes.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch import io_common as c
from vllm_fl.dispatch import io_dumper as i

_IO_ENV_KEYS = [
    "VLLM_FL_IO_DUMP",
    "VLLM_FL_IO_DUMP_OPS",
    "VLLM_FL_IO_DUMP_MODULES",
    "VLLM_FL_IO_DUMP_LAYERS",
    "VLLM_FL_IO_DUMP_MAX_CALLS",
    "VLLM_FL_IO_DUMP_STEP_RANGE",
    "VLLM_FL_IO_DUMP_RANK",
    "VLLM_FL_IO_DUMP_WITH_TORCH_FUNCS",
    "VLLM_FL_IO_DUMP_WITH_VALUES",
    "VLLM_FL_IO_DUMP_WITH_METAS",
    "VLLM_FL_IO_DUMP_WITH_PRINT",
    "VLLM_FL_IO_DUMP_BG_WORKERS",
    "VLLM_FL_CONFIG",
]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for key in _IO_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    c.reset_step()
    c.reset_exec_order()
    c.reset_rank()

    for tls in (i._dump_pairing, i._print_pairing):
        try:
            delattr(tls, "stack")
        except AttributeError:
            pass

    i._enabled = False
    i._dump_dir = ""
    i._match_all = False
    i._op_filter = set()
    i._module_filter = set()
    i._layer_filter = set()
    i._max_calls = 0
    i._step_range = None
    i._with_values = False
    i._with_metas = False
    i._print_enabled = False
    i._torch_funcs_enabled = False
    i._torch_func_filter = set()
    i._rank_filter = None
    i._call_counters = {}
    i._op_summary = {}
    i._hooks_activated = False
    i.set_guard(False)

    yield

    i._enabled = False
    i._dump_dir = ""
    i._with_values = False
    i._with_metas = False
    i._print_enabled = False
    i._call_counters = {}
    i._op_summary = {}
    i.set_guard(False)
    c.set_io_active(False)
    c.reset_step()
    c.reset_exec_order()
    c.reset_rank()
    for tls in (i._dump_pairing, i._print_pairing):
        try:
            delattr(tls, "stack")
        except AttributeError:
            pass


class _Func:
    """Callable stand-in for a torch operator/function."""

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


# 鈹€鈹€ init_io_dump_from_env 鈹€鈹€


class TestInitIoDumpFromEnv:
    def test_already_enabled_is_noop(self, monkeypatch):
        i._enabled = True
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        calls = []
        monkeypatch.setattr(i, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(i, "_activate_hooks", lambda: calls.append("hooks"))
        i.init_io_dump_from_env(True)
        assert calls == []
        assert i._enabled is True

    def test_not_eager_is_noop(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        calls = []
        monkeypatch.setattr(i, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(i, "_activate_hooks", lambda: calls.append("hooks"))
        i.init_io_dump_from_env(False)
        assert calls == []
        assert i._enabled is False

    def test_no_config_is_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(i, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(i, "_activate_hooks", lambda: calls.append("hooks"))
        i.init_io_dump_from_env(True)
        assert calls == []

    def test_env_dump_dir_triggers_init(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        calls = []
        monkeypatch.setattr(i, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(i, "_activate_hooks", lambda: calls.append("hooks"))
        i.init_io_dump_from_env(True)
        assert calls == ["init", "hooks"]

    def test_filter_env_only_triggers_init(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_IO_DUMP_STEP_RANGE", "0-1")
        calls = []
        monkeypatch.setattr(i, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(i, "_activate_hooks", lambda: calls.append("hooks"))
        i.init_io_dump_from_env(True)
        assert calls == ["init", "hooks"]

    def test_yaml_config_triggers_init(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_CONFIG", "cfg.yaml")
        calls = []
        monkeypatch.setattr(i, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(i, "_activate_hooks", lambda: calls.append("hooks"))
        i.init_io_dump_from_env(True)
        assert calls == ["init", "hooks"]

    def test_blank_config_env_is_noop(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_CONFIG", "   ")
        calls = []
        monkeypatch.setattr(i, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(i, "_activate_hooks", lambda: calls.append("hooks"))
        i.init_io_dump_from_env(True)
        assert calls == []


# 鈹€鈹€ dump_before 鈹€鈹€


class TestDumpBefore:
    def test_guard_active_short_circuits(self, monkeypatch):
        monkeypatch.setattr(i, "guard_active", lambda: True)
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: pytest.fail("no"))
        i.dump_before("mm", (), {})
        assert True

    def test_rank_filter_short_circuits(self, monkeypatch):
        monkeypatch.setattr(i, "_rank_ok", lambda: False)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: pytest.fail("no"))
        i.dump_before("mm", (), {})

    def test_should_dump_false_short_circuits(self, monkeypatch):
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: False)
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: pytest.fail("no"))
        i.dump_before("mm", (), {})

    def test_summary_only_records_and_returns(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = False
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: calls.setdefault("seen", a))
        monkeypatch.setattr(
            i, "_record_op_summary", lambda *a, **k: calls.setdefault("summary", a)
        )
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: pytest.fail("no file dump")
        )
        i.dump_before("mm", (torch.ones(2),), {})
        assert calls["seen"][0] == "mm"
        assert calls["summary"] == ("mm", None)

    def test_full_path_uses_preallocated_order(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = True
        i._dump_dir = "/d"
        i._print_enabled = True
        monkeypatch.setattr(i, "get_module_class_name", lambda args: None)
        monkeypatch.setattr(i, "get_current_module", lambda: "Linear")
        monkeypatch.setattr(i, "get_current_module_path", lambda: "blk")
        monkeypatch.setattr(i, "make_label", lambda op, *a, **k: "LBL")
        monkeypatch.setattr(i, "make_op_tag", lambda op: "[op=1,1]")
        monkeypatch.setattr(
            i, "next_exec_order", lambda: pytest.fail("must reuse exec_order")
        )
        monkeypatch.setattr(
            i, "_format_inputs", lambda args, kwargs, **k: ["  arg[0]"]
        )
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: rec.setdefault("in", (a, k))
        )
        monkeypatch.setattr(
            i, "_push_print_pairing", lambda *a, **k: rec.setdefault("pair", (a, k))
        )
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: rec.setdefault("seen", a))

        args = (torch.ones(2),)
        i.dump_before("mm", args, {"eps": 1e-6}, exec_order=42)

        in_args, in_kwargs = rec["in"]
        assert in_args[0] == "mm"
        assert in_kwargs["exec_order"] == 42
        assert in_kwargs["label"] == "LBL"
        assert in_kwargs["module_tag"] == "Linear:blk"
        assert in_kwargs["op_tag"] == "[op=1,1]"
        pair_args = rec["pair"][0]
        assert pair_args[0] == "mm"
        assert pair_args[1] == "LBL"
        assert pair_args[2] == 42
        assert pair_args[3] == ["  arg[0]"]
        assert pair_args[4] == "[op=1,1]"

    def test_full_path_generates_order_and_skips_print(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = True
        i._dump_dir = "/d"
        i._print_enabled = False
        monkeypatch.setattr(i, "get_module_class_name", lambda args: None)
        monkeypatch.setattr(i, "get_current_module", lambda: None)
        monkeypatch.setattr(i, "get_current_module_path", lambda: "")
        monkeypatch.setattr(i, "make_label", lambda op, *a, **k: "L")
        monkeypatch.setattr(i, "next_exec_order", lambda: 9)
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: rec.setdefault("in", (a, k))
        )
        monkeypatch.setattr(
            i, "_push_print_pairing", lambda *a, **k: pytest.fail("no print")
        )
        i.dump_before("mm", (torch.ones(1),), {}, op_tag="[op=2,2]")
        assert rec["in"][1]["exec_order"] == 9
        assert rec["in"][1]["op_tag"] == "[op=2,2]"
        assert rec["in"][1]["module_tag"] == ""

    def test_print_only_without_dump_dir(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = True
        i._dump_dir = ""
        i._print_enabled = True
        monkeypatch.setattr(i, "get_module_class_name", lambda args: "Linear")
        monkeypatch.setattr(i, "get_current_module", lambda: None)
        monkeypatch.setattr(i, "get_current_module_path", lambda: "")
        monkeypatch.setattr(i, "make_label", lambda op, *a, **k: "L")
        monkeypatch.setattr(i, "make_op_tag", lambda op: "T")
        monkeypatch.setattr(i, "next_exec_order", lambda: 1)
        monkeypatch.setattr(i, "_format_inputs", lambda a, k, **kw: [])
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: pytest.fail("must skip file dump")
        )
        monkeypatch.setattr(
            i, "_push_print_pairing", lambda *a, **k: rec.setdefault("pair", a)
        )
        i.dump_before("mm", (torch.ones(1),), {})
        assert rec["pair"][0] == "mm"

    def test_guard_is_released_when_dump_raises(self, monkeypatch):
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = True
        i._dump_dir = "/d"
        monkeypatch.setattr(i, "get_module_class_name", lambda args: None)
        monkeypatch.setattr(i, "get_current_module", lambda: None)
        monkeypatch.setattr(i, "get_current_module_path", lambda: "")
        monkeypatch.setattr(i, "make_label", lambda op, *a, **k: "L")
        monkeypatch.setattr(i, "make_op_tag", lambda op: "T")
        monkeypatch.setattr(i, "next_exec_order", lambda: 1)

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(i, "_dump_input", _boom)
        with pytest.raises(RuntimeError, match="boom"):
            i.dump_before("mm", (), {})
        assert i.guard_active() is False


# 鈹€鈹€ dump_after 鈹€鈹€


class TestDumpAfter:
    def test_guard_active_short_circuits(self, monkeypatch):
        monkeypatch.setattr(i, "guard_active", lambda: True)
        monkeypatch.setattr(i, "_dump_output", lambda *a: pytest.fail("no"))
        i.dump_after("mm", (), 1)

    def test_rank_filter_short_circuits(self, monkeypatch):
        monkeypatch.setattr(i, "_rank_ok", lambda: False)
        monkeypatch.setattr(i, "_dump_output", lambda *a: pytest.fail("no"))
        i.dump_after("mm", (), 1)

    def test_should_dump_false_short_circuits(self, monkeypatch):
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: False)
        monkeypatch.setattr(i, "_dump_output", lambda *a: pytest.fail("no"))
        i.dump_after("mm", (), 1)

    def test_summary_only_returns(self, monkeypatch):
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = False
        monkeypatch.setattr(i, "_dump_output", lambda *a: pytest.fail("no"))
        i.dump_after("mm", (), 1)

    def test_file_and_print_with_pairing(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = True
        i._dump_dir = "/d"
        i._print_enabled = True
        monkeypatch.setattr(
            i, "_dump_output", lambda *a: rec.setdefault("out", a)
        )
        monkeypatch.setattr(
            i,
            "_pop_print_pairing",
            lambda op: ("LBL", 5, ["  arg[0]"], "[op=1,1]"),
        )
        monkeypatch.setattr(
            i, "_log_combined", lambda *a, **k: rec.setdefault("log", (a, k))
        )
        i.dump_after("mm", (), torch.ones(2))

        assert rec["out"][0] == "mm"
        log_args, log_kwargs = rec["log"]
        assert log_args[0] == "LBL"
        assert log_args[1] == ["  arg[0]"]
        assert log_kwargs["op_tag"] == "[op=1,1]"
        assert log_kwargs["exec_order"] == 5

    def test_print_without_pairing_uses_outputs_only(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = True
        i._dump_dir = ""
        i._print_enabled = True
        monkeypatch.setattr(i, "_pop_print_pairing", lambda op: None)
        monkeypatch.setattr(i, "make_op_tag", lambda op: "[op=7,7]")
        monkeypatch.setattr(
            i, "_log_outputs_only", lambda *a, **k: rec.setdefault("log", (a, k))
        )
        monkeypatch.setattr(i, "_log_combined", lambda *a, **k: pytest.fail("no"))

        i.dump_after("mm", (), 3)
        log_args, log_kwargs = rec["log"]
        assert log_args[0] == "Op 'mm' (module=)"
        assert log_args[1] == 3
        assert log_kwargs["op_tag"] == "[op=7,7]"

    def test_guard_is_released_when_output_raises(self, monkeypatch):
        monkeypatch.setattr(i, "_rank_ok", lambda: True)
        monkeypatch.setattr(i, "_should_dump", lambda op, args: True)
        i._with_metas = True
        i._dump_dir = "/d"

        def _boom(*a):
            raise RuntimeError("out boom")

        monkeypatch.setattr(i, "_dump_output", _boom)
        with pytest.raises(RuntimeError, match="out boom"):
            i.dump_after("mm", (), 1)
        assert i.guard_active() is False


# 鈹€鈹€ _DumpDispatchMode.__torch_dispatch__ 鈹€鈹€


def _dispatch_mode():
    return i._DumpDispatchMode()


def _patch_dispatch_common(monkeypatch, op="mm", ns="aten"):
    monkeypatch.setattr(i, "get_dispatch_op_name", lambda func: op)
    monkeypatch.setattr(i, "get_dispatch_op_namespace", lambda func: ns)
    monkeypatch.setattr(
        i,
        "get_dispatch_keys",
        lambda func, args, kwargs: [("CUDA", "FlagGems", False)],
    )
    monkeypatch.setattr(i, "should_inspect_dispatch_op", lambda n, m, f: True)
    monkeypatch.setattr(i, "_rank_ok", lambda: True)


class TestDumpDispatchMode:
    def test_disabled_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        monkeypatch.setattr(i, "should_inspect_dispatch_op", lambda n, m, f: False)
        func = _Func("ret")
        i._enabled = False
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"
        assert func.calls == 1

    def test_guard_active_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        monkeypatch.setattr(i, "guard_active", lambda: True)
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), None) == "ret"

    def test_rank_mismatch_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        monkeypatch.setattr(i, "_rank_ok", lambda: False)
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"
        assert func.calls == 1

    def test_op_not_inspected_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        monkeypatch.setattr(i, "should_inspect_dispatch_op", lambda n, m, f: False)
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"

    def test_step_out_of_range_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._step_range = (5, 6)
        monkeypatch.setattr(i, "get_step", lambda: 0)
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"

    def test_summary_only_records_and_passes(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._with_metas = False
        rec = {}
        monkeypatch.setattr(
            i, "_record_op_summary", lambda *a: rec.setdefault("summary", a)
        )
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: rec.setdefault("seen", a))
        func = _Func("ret")
        out = _dispatch_mode().__torch_dispatch__(func, {}, (1,), {})
        assert out == "ret"
        assert rec["summary"][0] == "aten.mm"
        assert rec["summary"][1] == [("CUDA", "FlagGems", False)]
        assert rec["seen"][0] == "aten.mm"

    def test_layer_filter_mismatch_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._layer_filter = {"model.layers.0"}
        monkeypatch.setattr(i, "layer_path_matches", lambda flt: False)
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"

    def test_module_filter_mismatch_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._module_filter = {"Linear"}
        monkeypatch.setattr(i, "module_context_matches", lambda flt: False)
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"

    def test_limits_failure_passthrough(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        monkeypatch.setattr(i, "_check_limits", lambda name: False)
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"

    def test_full_dump_success(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._dump_dir = "/d"
        i._print_enabled = True
        rec = {}
        monkeypatch.setattr(i, "_check_limits", lambda name: True)
        monkeypatch.setattr(i, "get_current_module", lambda: "Linear")
        monkeypatch.setattr(i, "get_current_module_path", lambda: "blk")
        monkeypatch.setattr(i, "make_label", lambda name, **k: "L:" + name)
        monkeypatch.setattr(i, "next_exec_order", lambda: 3)
        monkeypatch.setattr(
            i, "make_module_tag_from_ctx", lambda m, p, for_json=False: f"{m}:{p}"
        )
        monkeypatch.setattr(i, "make_op_tag", lambda name: "[op=1,1]")
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: rec.setdefault("seen", a))
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: rec.setdefault("in", (a, k))
        )
        monkeypatch.setattr(
            i, "_dump_output", lambda *a, **k: rec.setdefault("out", (a, k))
        )
        monkeypatch.setattr(i, "_format_inputs", lambda a, k: ["  arg[0]"])
        monkeypatch.setattr(
            i, "_log_combined", lambda *a, **k: rec.setdefault("log", (a, k))
        )
        func = _Func("ret")

        out = _dispatch_mode().__torch_dispatch__(func, {}, (torch.ones(2),), {})

        assert out == "ret"
        assert func.calls == 1
        assert rec["in"][0][0] == "aten.mm"
        assert rec["in"][1]["exec_order"] == 3
        assert rec["in"][1]["module_tag"] == "Linear:blk"
        assert rec["in"][1]["op_tag"] == "[op=1,1]"
        assert rec["in"][1]["dispatch_keys"] == [("CUDA", "FlagGems", False)]
        assert rec["out"][0][0] == "aten.mm"
        assert rec["log"][0][0] == "L:aten.mm"
        assert rec["log"][1]["exec_order"] == 3
        # order: input dump -> op call -> output dump/log
        assert rec["seen"][0] == "aten.mm"

    def test_print_without_file_dump(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._dump_dir = ""
        i._print_enabled = True
        rec = {}
        monkeypatch.setattr(i, "_check_limits", lambda name: True)
        monkeypatch.setattr(i, "get_current_module", lambda: None)
        monkeypatch.setattr(i, "get_current_module_path", lambda: "")
        monkeypatch.setattr(i, "make_label", lambda name, **k: "L")
        monkeypatch.setattr(i, "next_exec_order", lambda: 1)
        monkeypatch.setattr(
            i, "make_module_tag_from_ctx", lambda m, p, for_json=False: ""
        )
        monkeypatch.setattr(i, "make_op_tag", lambda name: "T")
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: pytest.fail("no file dump")
        )
        monkeypatch.setattr(
            i, "_dump_output", lambda *a, **k: pytest.fail("no file dump")
        )
        monkeypatch.setattr(i, "_format_inputs", lambda a, k: ["x"])
        monkeypatch.setattr(
            i, "_log_combined", lambda *a, **k: rec.setdefault("log", a)
        )
        func = _Func("ret")
        assert _dispatch_mode().__torch_dispatch__(func, {}, (1,), {}) == "ret"
        assert rec["log"][1] == ["x"]

    def test_func_exception_pops_pairing_and_reraises(self, monkeypatch):
        _patch_dispatch_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._dump_dir = ""
        rec = {}
        monkeypatch.setattr(i, "_check_limits", lambda name: True)
        monkeypatch.setattr(i, "get_current_module", lambda: None)
        monkeypatch.setattr(i, "get_current_module_path", lambda: "")
        monkeypatch.setattr(i, "make_label", lambda name, **k: "L")
        monkeypatch.setattr(i, "next_exec_order", lambda: 1)
        monkeypatch.setattr(
            i, "make_module_tag_from_ctx", lambda m, p, for_json=False: ""
        )
        monkeypatch.setattr(i, "make_op_tag", lambda name: "T")
        monkeypatch.setattr(
            i, "_pop_pairing", lambda name: rec.setdefault("pop", name)
        )
        func = _Func(exc=RuntimeError("kernel failed"))
        with pytest.raises(RuntimeError, match="kernel failed"):
            _dispatch_mode().__torch_dispatch__(func, {}, (1,), {})
        assert rec["pop"] == "aten.mm"
        assert i.guard_active() is False


# 鈹€鈹€ _DumpTorchFuncMode.__torch_function__ 鈹€鈹€


def _torch_func_mode():
    return i._DumpTorchFuncMode()


def _patch_torch_func_common(monkeypatch, name="matmul"):
    monkeypatch.setattr(
        i.torch, "compiler", SimpleNamespace(is_compiling=lambda: False), raising=False
    )
    monkeypatch.setattr(i, "get_torch_func_name", lambda func: name)
    monkeypatch.setattr(i, "_rank_ok", lambda: True)


class TestDumpTorchFuncMode:
    def test_compiling_passthrough(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        monkeypatch.setattr(
            i.torch, "compiler", SimpleNamespace(is_compiling=lambda: True)
        )
        i._enabled = True
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"

    def test_disabled_passthrough(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = False
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"

    def test_guard_active_passthrough(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        monkeypatch.setattr(i, "guard_active", lambda: True)
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"

    def test_rank_mismatch_passthrough(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        monkeypatch.setattr(i, "_rank_ok", lambda: False)
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"

    def test_summary_only_records(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        i._with_metas = False
        rec = {}
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: rec.setdefault("seen", a))
        monkeypatch.setattr(
            i, "release_torch_func_tags", lambda: pytest.fail("no tags acquired")
        )
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"
        assert rec["seen"][0] == "torch.matmul"

    def test_should_dump_false_passthrough(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        monkeypatch.setattr(i, "_should_dump_torch_func", lambda name: False)
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"

    def test_limits_failure_passthrough(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        monkeypatch.setattr(i, "_should_dump_torch_func", lambda name: True)
        monkeypatch.setattr(i, "_check_limits", lambda name: False)
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"

    def test_full_dump_success(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._dump_dir = "/d"
        i._print_enabled = True
        rec = {}
        monkeypatch.setattr(i, "_should_dump_torch_func", lambda name: True)
        monkeypatch.setattr(i, "_check_limits", lambda name: True)
        monkeypatch.setattr(i, "get_current_module", lambda: "Linear")
        monkeypatch.setattr(i, "get_current_module_path", lambda: "blk")
        monkeypatch.setattr(i, "make_label", lambda name, **k: "L:" + name)
        monkeypatch.setattr(
            i, "acquire_torch_func_tags", lambda name: ("[m]", "[op]", 5)
        )
        monkeypatch.setattr(
            i, "release_torch_func_tags", lambda: rec.setdefault("release", True)
        )
        monkeypatch.setattr(i, "record_seen", lambda *a, **k: rec.setdefault("seen", a))
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: rec.setdefault("in", (a, k))
        )
        monkeypatch.setattr(
            i, "_dump_output", lambda *a, **k: rec.setdefault("out", (a, k))
        )
        monkeypatch.setattr(i, "_format_inputs", lambda a, k: ["x"])
        monkeypatch.setattr(
            i, "_log_combined", lambda *a, **k: rec.setdefault("log", (a, k))
        )
        func = _Func("ret")

        out = _torch_func_mode().__torch_function__(func, {}, (torch.ones(2),), {})

        assert out == "ret"
        assert func.calls == 1
        assert rec["in"][0][0] == "torch.matmul"
        assert rec["in"][1]["exec_order"] == 5
        assert rec["in"][1]["op_tag"] == "[op]"
        assert rec["in"][1]["module_tag"] == "Linear:blk"
        assert rec["out"][0][0] == "torch.matmul"
        assert rec["log"][1]["exec_order"] == 5
        assert rec["release"] is True

    def test_print_without_file_dump(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._dump_dir = ""
        i._print_enabled = True
        rec = {}
        monkeypatch.setattr(i, "_should_dump_torch_func", lambda name: True)
        monkeypatch.setattr(i, "_check_limits", lambda name: True)
        monkeypatch.setattr(i, "get_current_module", lambda: None)
        monkeypatch.setattr(i, "get_current_module_path", lambda: "")
        monkeypatch.setattr(i, "make_label", lambda name, **k: "L")
        monkeypatch.setattr(i, "acquire_torch_func_tags", lambda name: ("", "T", 2))
        monkeypatch.setattr(
            i, "release_torch_func_tags", lambda: rec.setdefault("release", True)
        )
        monkeypatch.setattr(
            i, "_dump_input", lambda *a, **k: pytest.fail("no file dump")
        )
        monkeypatch.setattr(
            i, "_dump_output", lambda *a, **k: pytest.fail("no file dump")
        )
        monkeypatch.setattr(i, "_format_inputs", lambda a, k: ["x"])
        monkeypatch.setattr(
            i, "_log_combined", lambda *a, **k: rec.setdefault("log", a)
        )
        func = _Func("ret")
        assert _torch_func_mode().__torch_function__(func, {}, (1,), {}) == "ret"
        assert rec["log"][1] == ["x"]
        assert rec["release"] is True

    def test_func_exception_releases_tags_and_reraises(self, monkeypatch):
        _patch_torch_func_common(monkeypatch)
        i._enabled = True
        i._with_metas = True
        i._dump_dir = ""
        rec = {}
        monkeypatch.setattr(i, "_should_dump_torch_func", lambda name: True)
        monkeypatch.setattr(i, "_check_limits", lambda name: True)
        monkeypatch.setattr(i, "get_current_module", lambda: None)
        monkeypatch.setattr(i, "get_current_module_path", lambda: "")
        monkeypatch.setattr(i, "make_label", lambda name, **k: "L")
        monkeypatch.setattr(i, "acquire_torch_func_tags", lambda name: ("", "T", 2))
        monkeypatch.setattr(
            i, "release_torch_func_tags", lambda: rec.setdefault("release", True)
        )
        monkeypatch.setattr(
            i, "_pop_pairing", lambda name: rec.setdefault("pop", name)
        )
        func = _Func(exc=RuntimeError("func failed"))
        with pytest.raises(RuntimeError, match="func failed"):
            _torch_func_mode().__torch_function__(func, {}, (1,), {})
        assert rec["pop"] == "torch.matmul"
        assert rec["release"] is True
        assert i.guard_active() is False


# 鈹€鈹€ _dump_input / _dump_output real bodies 鈹€鈹€


class TestDumpInputOutputBodies:
    def _setup(self, monkeypatch, tmp_path, with_values):
        i._enabled = True
        i._dump_dir = str(tmp_path)
        i._with_values = with_values
        i._call_counters = {}
        i._op_summary = {}
        submitted = []
        debug = []
        monkeypatch.setattr(
            i, "_submit_bg", lambda *a, **k: submitted.append((a, k))
        )
        monkeypatch.setattr(i, "_record_device_event", lambda refs: None)
        monkeypatch.setattr(
            i,
            "logger",
            SimpleNamespace(
                debug=lambda *a, **k: debug.append(a),
                warning=lambda *a, **k: None,
            ),
        )
        return submitted, debug

    def test_input_with_values(self, tmp_path, monkeypatch):
        submitted, debug = self._setup(monkeypatch, tmp_path, True)
        args = (torch.ones(2), "not-a-tensor")
        kwargs = {"eps": 1e-6, "t": torch.zeros(3)}
        i._dump_input(
            "mm",
            args,
            kwargs,
            exec_order=4,
            label="L",
            module_tag="M",
            op_tag="T",
            dispatch_keys=[("CUDA", "FlagGems", False)],
        )

        assert i._call_counters["mm"] == 1
        assert i._op_summary["mm"]["call_count"] == 1
        assert i._op_summary["mm"]["dispatch_keys"] == "[(CUDA, FlagGems, False)]"

        pairing = i._pop_pairing("mm")
        assert pairing[0] == 1
        assert pairing[1] == 4
        assert pairing[3] == "L"
        assert pairing[4] == "M"
        assert pairing[5] == "T"
        assert os.path.isdir(pairing[2])

        assert len(submitted) == 2
        first_a, first_k = submitted[0]
        assert first_a[0] is i._bg_compute_and_write
        assert first_a[1].endswith("input.json")
        assert first_a[2] == 1
        skeleton = first_a[3]
        assert skeleton["op_name"] == "mm"
        assert skeleton["exec_order"] == 4
        assert skeleton["call_num"] == 1
        assert skeleton["module_tag"] == "M"
        assert skeleton["op_tag"] == "T"
        assert set(first_a[4]) == {"arg_0", "kwarg_t"}

        second_a, second_k = submitted[1]
        assert second_a[0] is i._bg_save_pt
        assert second_a[1] == args
        assert second_a[2] == kwargs
        assert second_k["is_output"] is False
        assert second_k["pt_path"].endswith("call_1_input.pt")
        assert debug

    def test_input_without_values(self, tmp_path, monkeypatch):
        submitted, _ = self._setup(monkeypatch, tmp_path, False)
        i._dump_input("add", (torch.ones(1),), {}, exec_order=0)
        assert len(submitted) == 1
        assert submitted[0][0][0] is i._bg_compute_and_write
        # default label/module/op tags fall back to empty strings
        skeleton = submitted[0][0][3]
        assert skeleton["module_tag"] == ""
        assert skeleton["op_tag"] == ""

    def test_output_with_values(self, tmp_path, monkeypatch):
        submitted, debug = self._setup(monkeypatch, tmp_path, True)
        op_dir = str(tmp_path / "op")
        os.makedirs(op_dir, exist_ok=True)
        i._push_pairing("mm", 2, 9, op_dir, label="L", module_tag="M", op_tag="T")
        result = (torch.zeros(1), torch.ones(2))
        i._dump_output("mm", result)

        assert i._pop_pairing("mm") is None
        assert len(submitted) == 2
        first_a, _ = submitted[0]
        assert first_a[0] is i._bg_compute_and_write
        assert first_a[1] == os.path.join(op_dir, "output.json")
        assert first_a[2] == 2
        skeleton = first_a[3]
        assert skeleton["call_num"] == 2
        assert skeleton["exec_order"] == 9
        assert skeleton["module_tag"] == "M"
        assert set(first_a[4]) == {"result_0", "result_1"}

        second_a, second_k = submitted[1]
        assert second_a[0] is i._bg_save_pt
        assert second_a[1] == (result,)
        assert second_k["is_output"] is True
        assert second_k["pt_path"].endswith("call_2_output.pt")
        assert debug

    def test_output_without_values(self, tmp_path, monkeypatch):
        submitted, _ = self._setup(monkeypatch, tmp_path, False)
        op_dir = str(tmp_path / "op2")
        os.makedirs(op_dir, exist_ok=True)
        i._push_pairing("mm", 1, 0, op_dir)
        i._dump_output("mm", torch.ones(2))
        assert len(submitted) == 1
        assert submitted[0][0][1].endswith("output.json")

    def test_output_scalar_result(self, tmp_path, monkeypatch):
        submitted, _ = self._setup(monkeypatch, tmp_path, False)
        op_dir = str(tmp_path / "op3")
        os.makedirs(op_dir, exist_ok=True)
        i._push_pairing("mm", 1, 0, op_dir)
        i._dump_output("mm", 42)
        refs = submitted[0][0][4]
        assert refs == {}


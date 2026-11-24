# Copyright (c) 2026 BAAI. All rights reserved.
"""Coverage-focused unit tests for the vllm_fl.dispatch IO / policy spine.

Targets (read-only source, commit 030b452f):
  - vllm_fl/dispatch/io_dumper.py
  - vllm_fl/dispatch/io_common.py
  - vllm_fl/dispatch/manager.py
  - vllm_fl/dispatch/policy.py
  - vllm_fl/dispatch/discovery.py
  - vllm_fl/dispatch/config/utils.py
  - vllm_fl/dispatch/builtin_ops.py
  - vllm_fl/dispatch/__init__.py
  - vllm_fl/dispatch/ops.py

Self-contained: pytest + stdlib + unittest.mock only. torch is required by the
modules under test; there is no accelerator work. File I/O uses tmp_path.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import types
import weakref
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

import vllm_fl.dispatch as dispatch_init  # noqa: E402
from vllm_fl.dispatch import (  # noqa: E402
    builtin_ops,
    discovery,
    io_common,
    io_dumper,
    manager as manager_mod,
    policy as policy_mod,
)
from vllm_fl.dispatch.config import utils as cfg_utils  # noqa: E402
from vllm_fl.dispatch.manager import (  # noqa: E402
    OpManager,
    _OpManagerState,
    _can_retry_failed_op,
    _record_default_flagos_op,
    get_default_manager,
    reset_default_manager,
)
from vllm_fl.dispatch.ops import VLLMFLBackendBase  # noqa: E402
from vllm_fl.dispatch.policy import (  # noqa: E402
    PREFER_DEFAULT,
    PREFER_REFERENCE,
    PREFER_VENDOR,
    PolicyManager,
    SelectionPolicy,
    _PolicyContext,
    bump_policy_epoch,
    get_policy,
    get_policy_epoch,
    policy_context,
    policy_from_config,
    policy_from_env,
    reset_global_policy,
    set_global_policy,
    with_allowed_vendors,
    with_denied_vendors,
    with_preference,
    with_strict_mode,
)
from vllm_fl.dispatch.registry import OpRegistry  # noqa: E402
from vllm_fl.dispatch.types import (  # noqa: E402
    BackendImplKind,
    OpImpl,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #

def _install_fake_platform(monkeypatch, platform=None, *, missing_current=False):
    """Install a fake ``vllm.platforms`` module for the duration of a test."""
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    platforms = types.ModuleType("vllm.platforms")
    platforms.__path__ = []
    if not missing_current:
        platforms.current_platform = platform
    vllm_mod.platforms = platforms
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    return platforms


def _reg_impl(registry, op_name, impl_id, kind, fn, priority=0, vendor=None):
    impl = OpImpl(
        op_name=op_name,
        impl_id=impl_id,
        kind=kind,
        fn=fn,
        priority=priority,
        vendor=vendor,
    )
    registry.register_impl(impl)
    return impl


def _make_manager(registry=None, *, initialized=True):
    mgr = OpManager(registry=registry or OpRegistry())
    if initialized:
        mgr._state.initialized = True
        mgr._state.init_pid = os.getpid()
    return mgr


def _clear_module_ctx():
    stack = getattr(io_common._module_context, "stack", None)
    if stack:
        stack.clear()


def _safe_reset_dumper():
    d = io_dumper
    try:
        d._wait_and_flush()
    except Exception:
        pass
    try:
        d._enabled = False
        d._deactivate_hooks()
    except Exception:
        pass
    try:
        d._hooks_activated = False
    except Exception:
        pass
    try:
        d.set_guard(False)
    except Exception:
        pass
    try:
        d._reset_state()
    except Exception:
        pass


def _read_jsonl(path):
    data = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data.update(json.loads(line))
    return data


@pytest.fixture(autouse=True)
def _isolate_global_state(monkeypatch):
    """Keep every test independent of module-level mutable state."""
    env_snapshot = {k: v for k, v in os.environ.items() if k.startswith("VLLM_FL_")}
    for key in [k for k in os.environ if k.startswith("VLLM_FL_")]:
        del os.environ[key]
    for key in ("RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)

    saved_stats = list(io_common._TENSOR_STAT_REGISTRY)
    saved_table_cache = dict(io_common._dispatch_table_cache)
    saved_path_map = dict(io_common._module_path_map)
    saved_discovered = list(discovery._discovered_plugins)
    saved_callbacks = list(io_common._step_callbacks)

    # Deterministic policy base: ignore the host platform YAML / op config
    # unless a test explicitly re-patches these.
    monkeypatch.setattr(
        "vllm_fl.dispatch.config.get_config_path", lambda *a, **k: None
    )
    monkeypatch.setattr(policy_mod, "get_op_config", lambda: None)

    io_common.reset_rank()
    io_common.reset_exec_order()
    io_common.reset_step()
    io_common.set_io_active(False)
    _clear_module_ctx()
    manager_mod._RECORDED_FLAGOS_OPS.clear()
    _safe_reset_dumper()
    reset_global_policy()
    reset_default_manager()

    try:
        yield
    finally:
        _safe_reset_dumper()
        io_common.set_io_active(False)
        _clear_module_ctx()
        io_common._TENSOR_STAT_REGISTRY[:] = saved_stats
        io_common._dispatch_table_cache.clear()
        io_common._dispatch_table_cache.update(saved_table_cache)
        io_common._module_path_map.clear()
        io_common._module_path_map.update(saved_path_map)
        discovery._discovered_plugins[:] = saved_discovered
        io_common._step_callbacks[:] = saved_callbacks
        io_common.reset_rank()
        io_common.reset_exec_order()
        io_common.reset_step()
        manager_mod._RECORDED_FLAGOS_OPS.clear()
        reset_global_policy()
        reset_default_manager()
        for key in [k for k in os.environ if k.startswith("VLLM_FL_")]:
            del os.environ[key]
        os.environ.update(env_snapshot)


# =========================================================================== #
# io_common.py
# =========================================================================== #

class TestRankHelpers:
    def test_reset_and_default(self):
        io_common.reset_rank()
        assert io_common.get_rank() == 0

    def test_env_rank_cached(self, monkeypatch):
        io_common.reset_rank()
        monkeypatch.setenv("RANK", "3")
        assert io_common.get_rank() == 3
        monkeypatch.setenv("RANK", "9")
        assert io_common.get_rank() == 3

    def test_env_local_rank(self, monkeypatch):
        io_common.reset_rank()
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert io_common.get_rank() == 2

    def test_env_invalid_value_skipped(self, monkeypatch):
        io_common.reset_rank()
        monkeypatch.setenv("RANK", "not-int")
        monkeypatch.setenv("LOCAL_RANK", "4")
        assert io_common.get_rank() == 4

    def test_distributed_rank(self, monkeypatch):
        import torch.distributed as dist

        io_common.reset_rank()
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 7)
        assert io_common.get_rank() == 7

    def test_distributed_probe_exception(self, monkeypatch):
        import torch.distributed as dist

        io_common.reset_rank()

        def _boom():
            raise RuntimeError("no dist")

        monkeypatch.setattr(dist, "is_initialized", _boom)
        monkeypatch.setenv("RANK", "1")
        assert io_common.get_rank() == 1

    def test_parse_rank_filter(self):
        assert io_common.parse_rank_filter("") is None
        assert io_common.parse_rank_filter("all") is None
        assert io_common.parse_rank_filter("0,2,4") == {0, 2, 4}
        assert io_common.parse_rank_filter("x,1") == {1}
        assert io_common.parse_rank_filter("x") is None


class TestIoActiveAndGuard:
    def test_io_active_flag(self):
        io_common.set_io_active(True)
        assert io_common.is_io_active() is True
        io_common.set_io_active(False)
        assert io_common.is_io_active() is False

    def test_make_guard_isolated(self):
        g1_active, g1_set = io_common.make_guard()
        g2_active, _ = io_common.make_guard()
        g1_set(True)
        assert g1_active() is True
        assert g2_active() is False
        g1_set(False)
        assert g1_active() is False

    def test_managed_inference_mode_no_grad_branch(self):
        io_common.set_io_active(True)

        @io_common.managed_inference_mode()
        def fn(x):
            return x * 2

        assert torch.equal(fn(torch.ones(2)), torch.full((2,), 2.0))

    def test_managed_inference_mode_inference_branch(self):
        io_common.set_io_active(False)

        @io_common.managed_inference_mode()
        def fn(x):
            return x + 1

        assert torch.equal(fn(torch.zeros(2)), torch.ones(2))


class TestModeManager:
    class _Mode:
        def __init__(self):
            self.entered = 0
            self.exited = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *exc):
            self.exited += 1

    def test_enter_idempotent_and_lifo_drain(self):
        mgr = io_common.ModeManager()
        a, b = self._Mode(), self._Mode()
        mgr.enter("a", a)
        mgr.enter("a", a)
        assert a.entered == 1
        assert mgr.is_entered("a")
        mgr.enter("b", b)
        assert mgr.is_entered("b")

        mgr.request_exit("a")
        assert a.exited == 0
        mgr.request_exit("b")
        assert b.exited == 1
        assert a.exited == 1
        assert not mgr.is_entered("a")
        assert not mgr.is_entered("b")

    def test_request_exit_unknown_is_noop(self):
        mgr = io_common.ModeManager()
        mgr.request_exit("missing")

    def test_exit_all(self):
        mgr = io_common.ModeManager()
        a, b = self._Mode(), self._Mode()
        mgr.enter("a", a)
        mgr.enter("b", b)
        mgr.exit_all()
        assert a.exited == 1 and b.exited == 1


class TestExecOrderAndCounters:
    def test_exec_order(self):
        io_common.reset_exec_order()
        assert io_common.get_exec_order() == 0
        assert io_common.next_exec_order() == 1
        assert io_common.next_exec_order() == 2
        assert io_common.get_exec_order() == 2

    def test_module_and_op_counters(self):
        io_common.reset_step()
        assert io_common.next_module_counter("Linear") == (0, 1)
        assert io_common.next_module_counter("Linear") == (0, 2)
        assert io_common.next_module_counter("MLP") == (1, 1)
        assert io_common.next_op_counter("mm") == (0, 1)
        assert io_common.next_op_counter("mm") == (0, 2)
        assert io_common.next_op_counter("add") == (1, 1)

    def test_step_callbacks_receive_summary(self):
        io_common.reset_step()
        calls = []

        def cb(step, seen_modules, seen_ops):
            calls.append((step, set(seen_modules), set(seen_ops)))

        io_common.register_step_callback(cb)
        try:
            io_common.record_seen("my_op", module_name="MyModule")
            new_step = io_common.advance_step()
        finally:
            io_common.unregister_step_callback(cb)

        assert new_step == 1
        assert calls and calls[0][0] == 0
        assert "my_op" in calls[0][2]
        assert "MyModule" in calls[0][1]
        io_common.unregister_step_callback(cb)

    def test_advance_step_callback_error_swallowed(self):
        io_common.reset_step()

        def bad_cb(step, sm, so):
            raise RuntimeError("bad callback")

        io_common.register_step_callback(bad_cb)
        try:
            io_common.advance_step()
        finally:
            io_common.unregister_step_callback(bad_cb)


class TestStepRangeParsing:
    def test_parse_step_range(self):
        assert io_common.parse_step_range(None) is None
        assert io_common.parse_step_range(5) is None
        assert io_common.parse_step_range("") is None
        assert io_common.parse_step_range("0-2") == (0, 3)
        assert io_common.parse_step_range("5") == (5, 6)
        assert io_common.parse_step_range("a-b") is None
        assert io_common.parse_step_range("xyz") is None

    def test_parse_step_range_env(self, monkeypatch):
        monkeypatch.delenv("A", raising=False)
        monkeypatch.setenv("B", "3-5")
        assert io_common.parse_step_range_env("A", "B") == (3, 6)
        monkeypatch.setenv("C", "  ")
        assert io_common.parse_step_range_env("C") is None


class TestModuleContext:
    def test_register_and_lookup_path(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        io_common.register_module_paths(model)
        assert io_common.get_module_path(model[0]) == "0"
        unknown = torch.nn.Linear(1, 1)
        assert io_common.get_module_path(unknown) == ""

    def test_push_pop_context(self):
        _clear_module_ctx()
        io_common.push_module_context("Linear")
        io_common.push_module_context("Attn")
        assert io_common.get_current_module() == "Attn"
        idx, count = io_common.get_current_module_counter()
        assert idx >= 0 and count >= 1
        assert io_common.get_current_module_path() == ""
        io_common.pop_module_context()
        assert io_common.get_current_module() == "Linear"
        io_common.pop_module_context()
        assert io_common.get_current_module() is None
        assert io_common.get_current_module_counter() is None
        io_common.pop_module_context()

    def test_module_context_matches(self):
        _clear_module_ctx()
        assert io_common.module_context_matches({"Linear"}) is False
        io_common.push_module_context("Linear")
        try:
            assert io_common.module_context_matches({"Linear"}) is True
            assert io_common.module_context_matches({"Other"}) is False
        finally:
            io_common.pop_module_context()

    def test_make_module_tag(self):
        _clear_module_ctx()
        assert io_common.make_module_tag() == ""
        io_common.push_module_context("Linear")
        try:
            assert io_common.make_module_tag().startswith("[module=")
        finally:
            io_common.pop_module_context()


class TestLayerSpecs:
    def test_expand_layer_specs(self):
        out = io_common.expand_layer_specs(
            {"0", "2-4", "model.layers.*.self_attn", "", "model.layers.1"}
        )
        assert "model.layers.0" in out
        assert {"model.layers.2", "model.layers.3", "model.layers.4"}.issubset(out)
        assert "model.layers.*.self_attn" in out
        assert "model.layers.1" in out
        assert "" not in out

    def test_expand_custom_prefix(self):
        assert io_common.expand_layer_specs({"1"}, prefix="blocks.") == {"blocks.1"}

    def test_list_model_layers(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        all_paths = io_common.list_model_layers(model)
        assert any(p.startswith("0 (Linear)") for p in all_paths)
        assert io_common.list_model_layers(model, max_depth=1) == [
            "0 (Linear)",
            "1 (Linear)",
        ]

    def test_parse_layers_env(self, monkeypatch):
        monkeypatch.setenv("LAYERS_VAR", "0-2,model.x")
        out = io_common.parse_layers_env("MISSING_VAR", "LAYERS_VAR")
        assert out == {
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.x",
        }
        assert io_common.parse_layers_env("NOPE") == set()


class TestLayerPathMatching:
    def test_empty_stack(self):
        _clear_module_ctx()
        assert io_common.layer_path_matches({"model.layers.0"}) is False

    def test_exact_prefix_glob(self):
        _clear_module_ctx()
        model = torch.nn.Module()
        model.add_module("layers", torch.nn.Module())
        model.layers.add_module("0", torch.nn.Linear(2, 2))
        io_common.register_module_paths(model)
        io_common.push_module_context("Linear", getattr(model.layers, "0"))
        try:
            assert io_common.layer_path_matches({"layers.0"}) is True
            assert io_common.layer_path_matches({"layers"}) is True
            assert io_common.layer_path_matches({"layers.*"}) is True
            assert io_common.layer_path_matches({"other"}) is False
        finally:
            io_common.pop_module_context()

    def test_stack_based_matching(self):
        ctx = [("Linear", "model.layers.0.self_attn")]
        assert io_common.layer_path_matches_from_stack({"model.layers.0"}, ctx)
        assert io_common.layer_path_matches_from_stack({"model.layers.*"}, ctx)
        assert not io_common.layer_path_matches_from_stack({"other"}, ctx)
        assert not io_common.layer_path_matches_from_stack({"x"}, [("A", "")])

    def test_module_context_from_stack(self):
        class _Probe(torch.nn.Module):
            def forward(self, x):
                return io_common.get_module_context_from_stack()

        ctx = _Probe()(torch.zeros(1))
        assert any(name == "_Probe" for name, _ in ctx)

    def test_module_context_matches_from_stack(self):
        ctx = [("Linear", "p"), ("Attn", "q")]
        assert io_common.module_context_matches_from_stack({"Attn"}, ctx)
        assert not io_common.module_context_matches_from_stack({"Other"}, ctx)


class TestTorchFuncHelpers:
    def test_get_torch_func_name(self):
        def matmul():
            pass

        assert io_common.get_torch_func_name(matmul) == "matmul"

        class _NoName:
            def __str__(self):
                return "custom"

        assert io_common.get_torch_func_name(_NoName()) == "custom"

    def test_parse_torch_funcs_config(self):
        assert io_common.parse_torch_funcs_config("") == (False, set())
        assert io_common.parse_torch_funcs_config("0") == (False, set())
        assert io_common.parse_torch_funcs_config("1") == (True, set())
        enabled, funcs = io_common.parse_torch_funcs_config("matmul, softmax")
        assert enabled is True and funcs == {"matmul", "softmax"}

    def test_should_inspect_torch_func(self):
        assert io_common.should_inspect_torch_func("mm", False, set(), True, set()) is False
        assert io_common.should_inspect_torch_func("mm", True, {"mm"}, True, set()) is True
        assert io_common.should_inspect_torch_func("add", True, {"mm"}, True, set()) is False
        assert io_common.should_inspect_torch_func("_private", True, set(), True, set()) is False
        assert io_common.should_inspect_torch_func("size", True, set(), True, set()) is False
        assert io_common.should_inspect_torch_func("mm", True, set(), False, {"add"}) is False
        assert io_common.should_inspect_torch_func("add", True, set(), False, {"add"}) is True


class TestDispatchOpHelpers:
    def test_get_dispatch_op_name(self):
        def f():
            pass

        f.name = lambda: "aten::mm"
        assert io_common.get_dispatch_op_name(f) == "mm"
        g = lambda: None  # noqa: E731
        g.name = lambda: "single"
        assert io_common.get_dispatch_op_name(g) == "single"
        assert io_common.get_dispatch_op_name(object()).startswith("<")

    def test_get_dispatch_op_namespace(self):
        obj = types.SimpleNamespace(namespace="vllm")
        assert io_common.get_dispatch_op_namespace(obj) == "vllm"
        assert io_common.get_dispatch_op_namespace(object()) == "aten"

    def test_infer_backend_from_path(self):
        inf = io_common._infer_backend_from_path
        assert inf("/pytorch/build/aten/src/ATen/RegisterCPU_0.cpp:10") == "CPU"
        assert inf("/pytorch/build/aten/src/ATen/RegisterCUDA_0.cpp:20") == "CUDA"
        assert inf("/x/flag_gems/__init__.py:5") == "FlagGems"
        assert inf("/x/triton/ops/matmul.py:5") == "Triton"
        assert inf("/x/torch/_meta_registrations.py:5") == "Meta"
        assert inf("/x/TraceType_3.cpp") == "unknown"

    def test_parse_dispatch_table_and_cache(self, monkeypatch):
        io_common._dispatch_table_cache.clear()
        table = "\n".join(
            [
                "CPU: registered at /b/RegisterCPU_0.cpp:10 [kernel]",
                "CUDA: registered at /x/flag_gems/__init__.py:20 [kernel]",
                "Meta: registered at /x/torch/_meta_registrations.py:5 "
                "[default backend kernel]",
                "Sparse: registered at /x/foo.cpp:1 [kernel] (fallthrough)",
                "noise line without marker",
                "CPU: [kernel] no registered-at",
            ]
        )
        monkeypatch.setattr(
            torch._C, "_dispatch_dump_table", lambda qn: table, raising=False
        )

        def f():
            pass

        f.name = lambda: "aten::__cov_fl_op__"
        entries = io_common._parse_dispatch_table(f)
        assert entries == [
            ("CPU", "CPU", False),
            ("CUDA", "FlagGems", False),
            ("Meta", "Meta", True),
        ]
        assert io_common._parse_dispatch_table(f) is entries
        assert io_common.get_dispatch_keys(f) == entries

    def test_parse_dispatch_table_exception(self, monkeypatch):
        io_common._dispatch_table_cache.clear()

        def _raise(qn):
            raise RuntimeError("no table")

        monkeypatch.setattr(torch._C, "_dispatch_dump_table", _raise, raising=False)

        def f():
            pass

        f.name = lambda: "aten::__cov_fl_missing__"
        assert io_common._parse_dispatch_table(f) == []
        assert io_common.get_dispatch_keys(f) == [("unknown", "unknown", True)]

    def test_parse_dispatch_table_no_name(self):
        assert io_common._parse_dispatch_table(object()) == []

    def test_should_inspect_dispatch_op(self):
        assert io_common.should_inspect_dispatch_op("mm", True, set()) is True
        assert io_common.should_inspect_dispatch_op("mm", True, {"mm"}) is True
        assert io_common.should_inspect_dispatch_op("add", True, {"mm"}) is False


class TestTensorStatsAndFormatting:
    def test_register_stat_validation_and_replace(self):
        before = len(io_common._TENSOR_STAT_REGISTRY)
        with pytest.raises(ValueError):
            io_common.register_tensor_stat("   ", lambda t: 0)
        io_common.register_tensor_stat("min", lambda t: 42.0, float_only=False)
        assert len(io_common._TENSOR_STAT_REGISTRY) == before
        io_common.register_tensor_stat("l2", lambda t: float(t.norm()), float_only=True)
        assert len(io_common._TENSOR_STAT_REGISTRY) == before + 1

    def test_tensor_stats_float_int_complex_empty(self):
        fstats = io_common.tensor_stats(torch.tensor([1.0, 2.0, 3.0]))
        assert "min" in fstats and "mean" in fstats
        istats = io_common.tensor_stats(torch.tensor([1, 2, 3]))
        assert "min" in istats and "mean" not in istats
        cstats = io_common.tensor_stats(torch.tensor([1 + 2j]))
        assert "min" not in cstats
        estats = io_common.tensor_stats(torch.zeros(0))
        assert "min" not in estats

    def test_tensor_stats_swallows_bad_stat(self):
        io_common.register_tensor_stat(
            "boom",
            lambda t: (_ for _ in ()).throw(RuntimeError("x")),
            float_only=False,
        )
        stats = io_common.tensor_stats(torch.ones(2))
        assert "boom" not in stats

    def test_format_value_variants(self):
        assert io_common.format_value(None) == "None"
        assert io_common.format_value(True) == "True"
        assert io_common.format_value(3) == "3"
        assert "Tensor(" in io_common.format_value(torch.ones(2))
        assert io_common.format_value([1, 2]).startswith("list([")
        assert "len=10" in io_common.format_value(list(range(10)))
        assert io_common.format_value((1, 2)).startswith("tuple([")
        assert io_common.format_value(object()).endswith("(...)")

    def test_format_stat_list_truncation(self):
        io_common.register_tensor_stat(
            "topk", lambda t: [1.0, 2.0, 3.0, 4.0, 5.0], float_only=True
        )
        rendered = io_common.format_value(torch.ones(2))
        assert "topk=" in rendered and "..." in rendered

    def test_format_result(self):
        assert io_common.format_result(torch.ones(2)).startswith("  result:")
        multi = io_common.format_result((torch.ones(1), torch.zeros(1)))
        assert "result[0]:" in multi and "result[1]:" in multi


class TestLabelsAndTags:
    def test_get_module_class_name(self):
        assert io_common.get_module_class_name((torch.nn.Linear(2, 2),)) == "Linear"
        assert io_common.get_module_class_name((torch.zeros(2),)) is None
        assert io_common.get_module_class_name(()) is None

    def test_make_module_tag_from_ctx(self):
        assert io_common.make_module_tag_from_ctx("", "p", for_json=True) == ""
        assert io_common.make_module_tag_from_ctx("Linear", "", for_json=True) == "Linear"
        assert (
            io_common.make_module_tag_from_ctx("Linear", "a.b", for_json=True)
            == "Linear:a.b"
        )
        assert io_common.make_module_tag_from_ctx("Linear", "a.b") == "[Linear:a.b]"

    def test_make_op_tag_and_label(self):
        io_common.reset_step()
        assert io_common.make_op_tag("mm").startswith("[op=")
        label = io_common.make_label(
            "mm",
            module_name="Linear",
            layer_path="model.layers.0",
            dispatch_keys=[("CPU", "CPU", False), ("CUDA", "FlagGems", False)],
        )
        assert "module=Linear" in label
        assert "layer=model.layers.0" in label
        assert "dispatch_keys=[(CPU, CPU), (CUDA, FlagGems)]" in label
        default_label = io_common.make_label("mm", args=(torch.nn.Linear(2, 2),))
        assert "module=Linear" in default_label

    def test_record_seen(self):
        io_common.reset_step()
        io_common.record_seen("op_a", module_name="Linear")
        io_common.record_seen("op_b", args=(torch.nn.Linear(2, 2),))
        io_common.record_seen("op_c")
        assert "op_a" in io_common._seen_ops
        assert "Linear" in io_common._seen_modules

    def test_torch_func_tags_nesting(self):
        io_common.reset_exec_order()
        m1, o1, order1 = io_common.acquire_torch_func_tags("mm")
        m2, o2, order2 = io_common.acquire_torch_func_tags("mm")
        assert (m1, o1, order1) == (m2, o2, order2)
        io_common.release_torch_func_tags()
        io_common.release_torch_func_tags()
        m3, o3, order3 = io_common.acquire_torch_func_tags("mm")
        assert order3 != order1
        io_common.release_torch_func_tags()


class TestYamlParsing:
    def test_parse_io_config_from_yaml_missing_and_invalid(self, tmp_path):
        assert io_common.parse_io_config_from_yaml(str(tmp_path / "nope.yaml")) == {}
        bad = tmp_path / "bad.yaml"
        bad.write_text("io_dump: [unclosed", encoding="utf-8")
        assert io_common.parse_io_config_from_yaml(str(bad)) == {}
        plain = tmp_path / "plain.yaml"
        plain.write_text("other: 1\n", encoding="utf-8")
        assert io_common.parse_io_config_from_yaml(str(plain)) == {}

    def test_parse_io_config_from_yaml_full(self, tmp_path):
        cfg = tmp_path / "io.yaml"
        cfg.write_text(
            "io_dump:\n"
            "  dir: /tmp/dump\n"
            "  with_print: true\n"
            "  ops: [rms_norm, silu_and_mul]\n"
            "  modules:\n"
            "    - Linear\n"
            "  layers: [0, 1-2]\n"
            "  max_calls: 10\n"
            "  step_range: '1-3'\n"
            "  with_torch_funcs: true\n"
            "  with_metas: true\n"
            "  with_values: true\n"
            "  ranks: [0, 1]\n",
            encoding="utf-8",
        )
        parsed = io_common.parse_io_config_from_yaml(str(cfg))["io_dump"]
        assert parsed["dir"] == "/tmp/dump"
        assert parsed["ops"] == {"rms_norm", "silu_and_mul"}
        assert parsed["modules"] == {"Linear"}
        assert {"model.layers.0", "model.layers.1", "model.layers.2"}.issubset(
            parsed["layers"]
        )
        assert parsed["max_calls"] == 10
        assert parsed["step_range"] == (1, 4)
        assert parsed["with_torch_funcs"] == (True, set())
        assert parsed["with_metas"] is True
        assert parsed["with_values"] is True
        assert parsed["ranks"] == {0, 1}
        assert parsed["with_print"] is True

    def test_parse_step_range_yaml_variants(self):
        p = io_common._parse_step_range_yaml
        assert p({}) is None
        assert p({"step_range": 3}) == (3, 4)
        assert p({"step_range": [1, 3]}) == (1, 4)
        assert p({"step_range": [1, 2, 3]}) is None
        assert p({"step_range": "2-4"}) == (2, 5)
        assert p({"step_range": "bad"}) is None

    def test_parse_dump_section_defaults(self):
        parsed = io_common._parse_dump_section({})
        assert parsed["dir"] == ""
        assert parsed["ops"] == set()
        assert parsed["max_calls"] == 0
        assert parsed["step_range"] is None
        assert "with_torch_funcs" not in parsed
        assert "with_metas" not in parsed

    def test_parse_dump_section_bad_max_calls(self):
        assert io_common._parse_dump_section({"max_calls": "abc"})["max_calls"] == 0

    def test_parse_string_list(self):
        assert io_common._parse_string_list(None) == set()
        assert io_common._parse_string_list(["a", " b "]) == {"a", "b"}
        assert io_common._parse_string_list("a,b") == {"a", "b"}
        assert io_common._parse_string_list(123) == set()

    def test_parse_torch_funcs_yaml(self):
        p = io_common._parse_torch_funcs_yaml
        assert p(None) == (False, set())
        assert p(False) == (False, set())
        assert p(True) == (True, set())
        assert p(["matmul", "add"]) == (True, {"matmul", "add"})
        assert p("1") == (True, set())
        assert p(123) == (False, set())

    def test_parse_ranks_yaml(self):
        p = io_common._parse_ranks_yaml
        assert p(None) is None
        assert p([0, 1, "x"]) == {0, 1}
        assert p(["x"]) is None
        assert p(2) == {2}
        assert p("0,2") == {0, 2}
        assert p("all") is None
        assert p(3.5) is None


# =========================================================================== #
# io_dumper.py
# =========================================================================== #

class TestSerializeAndBuild:
    def test_serialize_value(self):
        s = io_dumper._serialize_value
        t = torch.ones(2)
        assert isinstance(s(t), torch.Tensor)
        assert s(torch.nn.Linear(2, 2)).startswith("<module:")
        assert s({"a": t})["a"].shape == (2,)
        assert s((t, 1))[0].shape == (2,)
        assert s([t])[0].shape == (2,)
        assert s(3) == 3
        assert s(3.5) == 3.5
        assert s("x") == "x"
        assert s(None) is None
        assert s(object()).startswith("<")

    def test_build_data_input_and_output(self):
        t = torch.ones(2)
        data = io_dumper._build_data((t,), {"k": 1})
        assert data["arg_0"].shape == (2,)
        assert data["kwarg_k"] == 1
        assert "result_0" in io_dumper._build_data(((t, t),), {}, is_output=True)
        assert "result" in io_dumper._build_data((t,), {}, is_output=True)
        assert io_dumper._build_data((), {}, is_output=True)["result"] is None

    def test_sanitize_path_component(self):
        f = io_dumper._sanitize_path_component
        assert f("") == "_unnamed_"
        assert f("   ") == "_unnamed_"
        assert f("name") == "name"
        assert os.sep not in f("a" + os.sep + "b")
        assert ".." not in f("..secret..")

    def test_extract_tensor_refs(self):
        t = torch.ones(2)
        assert set(io_dumper._extract_tensor_refs((t, 3), {"k": t})) == {
            "arg_0",
            "kwarg_k",
        }
        assert set(io_dumper._extract_tensor_refs(((t, 3),), {}, is_output=True)) == {
            "result_0"
        }
        assert set(io_dumper._extract_tensor_refs((t,), {}, is_output=True)) == {
            "result"
        }
        assert io_dumper._extract_tensor_refs((3,), {}, is_output=True) == {}

    def test_record_device_event_cpu_and_meta(self):
        assert io_dumper._record_device_event({"a": torch.ones(2)}) is None
        meta = types.SimpleNamespace(device=types.SimpleNamespace(type="meta"))
        assert io_dumper._record_device_event({"a": meta}) is None

    def test_format_inputs(self):
        mod = torch.nn.Linear(2, 2)
        lines = io_dumper._format_inputs(
            (mod, torch.ones(2)), {"k": 1}, skip_module_arg=True
        )
        assert len(lines) == 2
        assert lines[0].strip().startswith("arg[1]:")
        assert len(io_dumper._format_inputs((mod,), {})) == 1


class TestPairingAndDirs:
    def test_dump_pairing_push_pop(self):
        io_dumper._push_pairing("op", 1, 5, "/tmp/dir", label="L", module_tag="m", op_tag="o")
        assert io_dumper._pop_pairing("op") == (1, 5, "/tmp/dir", "L", "m", "o")
        assert io_dumper._pop_pairing("op") is None
        assert io_dumper._pop_pairing("never") is None

    def test_print_pairing_push_pop(self):
        io_dumper._push_print_pairing("op", "label", 3, ["line"], "[op=0,1]")
        assert io_dumper._pop_print_pairing("op") == ("label", 3, ["line"], "[op=0,1]")
        assert io_dumper._pop_print_pairing("op") is None

    def test_get_op_dir_and_next_call_num(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        monkeypatch.setattr(io_dumper, "get_step", lambda: 2)
        op_dir = io_dumper._get_op_dir("aten.mm")
        assert op_dir.endswith(os.path.join("rank_0000", "step_0002", "aten.mm"))
        with io_dumper._lock:
            io_dumper._call_counters.clear()
        assert io_dumper._next_call_num("mm") == 1
        assert io_dumper._next_call_num("mm") == 2

    def test_dump_cleanup_pops_pairings(self):
        io_dumper._push_pairing("op", 1, 1, "/tmp", label="L")
        io_dumper._push_print_pairing("op", "label", 1, [], "")
        io_dumper.dump_cleanup("op")
        assert io_dumper._pop_pairing("op") is None
        assert io_dumper._pop_print_pairing("op") is None


class TestAsyncWriteHelpers:
    def test_get_executor_reuse_and_recreate(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_bg_workers", 1)
        monkeypatch.setattr(io_dumper, "_io_executor", None)
        exc1 = io_dumper._get_executor()
        assert io_dumper._get_executor() is exc1
        exc1.shutdown(wait=True)
        exc2 = io_dumper._get_executor()
        assert exc2 is not exc1
        exc2.shutdown(wait=True)
        monkeypatch.setattr(io_dumper, "_io_executor", None)

    def test_submit_bg_disabled_and_enabled(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", False)
        before = len(io_dumper._pending_futures)
        io_dumper._submit_bg(lambda: None)
        assert len(io_dumper._pending_futures) == before
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_bg_workers", 1)
        io_dumper._submit_bg(lambda: None)
        assert len(io_dumper._pending_futures) >= before + 1
        io_dumper._wait_and_flush()

    def test_bg_compute_and_write_reorder_and_bad_tensor(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", True)
        json_path = str(tmp_path / "input.json")
        io_dumper._bg_compute_and_write(
            json_path, 2, {"op_name": "op"}, {"arg_0": torch.ones(2)}, None
        )
        io_dumper._bg_compute_and_write(
            json_path, 1, {"op_name": "op"}, {"arg_0": object()}, None
        )
        io_dumper._close_open_files()
        with open(json_path, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert "call_1" in lines[0]
        assert "call_2" in lines[1]
        assert lines[0]["call_1"]["tensors"] == {}

    def test_bg_compute_and_write_with_event(self, tmp_path):
        events = []

        class _Event:
            def synchronize(self):
                events.append("synced")

        io_dumper._bg_compute_and_write(
            str(tmp_path / "e.json"),
            1,
            {"op_name": "op"},
            {"arg_0": torch.ones(1)},
            _Event(),
        )
        io_dumper._close_open_files()
        assert events == ["synced"]

    def test_bg_save_pt_input_and_output(self, tmp_path):
        t = torch.ones(2)
        in_path = str(tmp_path / "in.pt")
        io_dumper._bg_save_pt((t,), {"k": 1}, is_output=False, pt_path=in_path, device_event=None)
        loaded = torch.load(in_path, weights_only=False)
        assert loaded["arg_0"].shape == (2,)
        assert loaded["kwarg_k"] == 1
        out_path = str(tmp_path / "out.pt")
        io_dumper._bg_save_pt(((t, t),), {}, is_output=True, pt_path=out_path, device_event=None)
        loaded_out = torch.load(out_path, weights_only=False)
        assert "result_0" in loaded_out and "result_1" in loaded_out

    def test_wait_and_flush_logs_exception(self, monkeypatch):
        def _boom():
            raise RuntimeError("bg failure")

        monkeypatch.setattr(io_dumper, "_enabled", True)
        io_dumper._submit_bg(_boom)
        io_dumper._wait_and_flush()

    def test_close_open_files(self, tmp_path):
        handle = io_dumper._get_or_open_handle(str(tmp_path / "h.json"))
        assert handle.fh.closed is False
        io_dumper._close_open_files()
        assert handle.fh.closed is True
        assert io_dumper._open_file_handles == {}


class TestDumpInputOutput:
    def test_dump_input_output_writes_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "_with_values", True)
        monkeypatch.setattr(io_dumper, "_with_metas", True)
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        monkeypatch.setattr(io_dumper, "get_step", lambda: 0)
        monkeypatch.setattr(io_dumper, "_bg_workers", 1)
        with io_dumper._lock:
            io_dumper._call_counters.clear()
        t = torch.zeros(2, 3)
        io_dumper._dump_input("op", (t,), {"k": 1}, exec_order=1)
        io_dumper._dump_output("op", torch.ones(2, 3))
        io_dumper._wait_and_flush()
        op_dir = tmp_path / "rank_0000" / "step_0000" / "op"
        assert (op_dir / "input.json").is_file()
        assert (op_dir / "output.json").is_file()
        assert (op_dir / "call_1_input.pt").is_file()
        assert (op_dir / "call_1_output.pt").is_file()
        assert "call_1" in _read_jsonl(str(op_dir / "input.json"))

    def test_dump_output_without_pairing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "_with_values", False)
        monkeypatch.setattr(io_dumper, "_with_metas", True)
        io_dumper._dump_output("lonely", torch.ones(1))
        io_dumper._wait_and_flush()

    def test_dump_input_exception_swallowed(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_dump_dir", "/some/dir")

        def _raise(_name):
            raise RuntimeError("boom")

        monkeypatch.setattr(io_dumper, "_get_op_dir", _raise)
        io_dumper._dump_input("op", (torch.ones(1),), {}, exec_order=1)


class TestDumpPublicApi:
    def test_is_dump_enabled_and_advance(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", False)
        assert io_dumper.is_dump_enabled() is False
        calls = []
        monkeypatch.setattr(io_dumper, "advance_step", lambda: calls.append(1))
        io_dumper.advance_io_step()
        assert calls == []
        monkeypatch.setattr(io_dumper, "_enabled", True)
        io_dumper.advance_io_step()
        assert calls == [1]
        monkeypatch.setattr(io_dumper, "advance_step", io_common.advance_step)
        assert isinstance(io_dumper.io_dump_step(), int)

    def test_init_io_dump_from_env_guards(self, monkeypatch):
        calls = []
        monkeypatch.setattr(io_dumper, "_enabled", False)
        monkeypatch.setattr(io_dumper, "_init_from_env", lambda: calls.append("init"))
        monkeypatch.setattr(io_dumper, "_activate_hooks", lambda: calls.append("act"))
        io_dumper.init_io_dump_from_env(False)
        assert calls == []
        monkeypatch.setattr(io_dumper, "_enabled", True)
        io_dumper.init_io_dump_from_env(True)
        assert calls == []
        monkeypatch.setattr(io_dumper, "_enabled", False)
        io_dumper.init_io_dump_from_env(True)
        assert calls == []
        monkeypatch.setenv("VLLM_FL_IO_DUMP", "1")
        io_dumper.init_io_dump_from_env(True)
        assert calls == ["init", "act"]

    def test_rank_ok(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_rank_filter", None)
        assert io_dumper._rank_ok() is True
        monkeypatch.setattr(io_dumper, "_rank_filter", {1})
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 1)
        assert io_dumper._rank_ok() is True
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        assert io_dumper._rank_ok() is False

    def test_check_limits(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", (2, 4))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 1)
        assert io_dumper._check_limits("op") is False
        monkeypatch.setattr(io_dumper, "get_step", lambda: 3)
        assert io_dumper._check_limits("op") is True
        monkeypatch.setattr(io_dumper, "_step_range", None)
        monkeypatch.setattr(io_dumper, "_max_calls", 1)
        with io_dumper._lock:
            io_dumper._call_counters.clear()
            io_dumper._call_counters["op"] = 1
        assert io_dumper._check_limits("op") is False
        assert io_dumper._check_limits("other") is True


class TestShouldDump:
    def test_should_dump_filters(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", None)
        monkeypatch.setattr(io_dumper, "_max_calls", 0)
        monkeypatch.setattr(io_dumper, "_layer_filter", set())
        monkeypatch.setattr(io_dumper, "_match_all", True)
        monkeypatch.setattr(io_dumper, "_module_filter", set())
        monkeypatch.setattr(io_dumper, "_op_filter", set())
        assert io_dumper._should_dump("op", ()) is True

        monkeypatch.setattr(io_dumper, "_layer_filter", {"model.layers.0"})
        _clear_module_ctx()
        assert io_dumper._should_dump("op", ()) is False
        monkeypatch.setattr(io_dumper, "_layer_filter", set())

        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        _clear_module_ctx()
        assert io_dumper._should_dump("op", (torch.nn.Linear(2, 2),)) is True
        assert io_dumper._should_dump("op", (torch.zeros(1),)) is False

        monkeypatch.setattr(io_dumper, "_module_filter", set())
        monkeypatch.setattr(io_dumper, "_op_filter", {"op"})
        assert io_dumper._should_dump("op", ()) is True
        assert io_dumper._should_dump("other", ()) is False

        monkeypatch.setattr(io_dumper, "_op_filter", set())
        assert io_dumper._should_dump("op", ()) is False

    def test_should_dump_torch_func(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_step_range", None)
        monkeypatch.setattr(io_dumper, "_layer_filter", set())
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_torch_func_filter", set())
        monkeypatch.setattr(io_dumper, "_match_all", True)
        monkeypatch.setattr(io_dumper, "_op_filter", set())
        monkeypatch.setattr(io_dumper, "_module_filter", set())
        _clear_module_ctx()
        assert io_dumper._should_dump_torch_func("matmul") is True

        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", False)
        assert io_dumper._should_dump_torch_func("matmul") is False

        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_step_range", (5, 6))
        monkeypatch.setattr(io_dumper, "get_step", lambda: 0)
        assert io_dumper._should_dump_torch_func("matmul") is False
        monkeypatch.setattr(io_dumper, "_step_range", None)

        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_module_filter", {"Linear"})
        assert io_dumper._should_dump_torch_func("matmul") is False


class TestDumpBeforeAfter:
    def _prime(self, monkeypatch, tmp_path, **extra):
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "_rank_filter", None)
        monkeypatch.setattr(io_dumper, "_step_range", None)
        monkeypatch.setattr(io_dumper, "_max_calls", 0)
        monkeypatch.setattr(io_dumper, "_layer_filter", set())
        monkeypatch.setattr(io_dumper, "_match_all", True)
        monkeypatch.setattr(io_dumper, "_module_filter", set())
        monkeypatch.setattr(io_dumper, "_op_filter", set())
        monkeypatch.setattr(io_dumper, "_with_metas", True)
        monkeypatch.setattr(io_dumper, "_with_values", False)
        monkeypatch.setattr(io_dumper, "_print_enabled", False)
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        monkeypatch.setattr(io_dumper, "get_step", lambda: 0)
        monkeypatch.setattr(io_dumper, "_bg_workers", 1)
        with io_dumper._lock:
            io_dumper._call_counters.clear()
        for key, value in extra.items():
            monkeypatch.setattr(io_dumper, key, value)

    def test_summary_only_branch(self, monkeypatch, tmp_path):
        self._prime(monkeypatch, tmp_path, _with_metas=False)
        io_dumper.dump_before("op", (torch.ones(1),), {})
        assert "op" in io_dumper._op_summary
        io_dumper.dump_after("op", (torch.ones(1),), torch.ones(1))

    def test_guard_short_circuit(self, monkeypatch, tmp_path):
        self._prime(monkeypatch, tmp_path)
        io_dumper.set_guard(True)
        try:
            io_dumper.dump_before("op", (torch.ones(1),), {})
            io_dumper.dump_after("op", (torch.ones(1),), torch.ones(1))
        finally:
            io_dumper.set_guard(False)
        assert "op" not in io_dumper._op_summary

    def test_rank_filter_short_circuit(self, monkeypatch, tmp_path):
        self._prime(monkeypatch, tmp_path, _rank_filter={2})
        io_dumper.dump_before("op", (torch.ones(1),), {})
        assert "op" not in io_dumper._op_summary

    def test_full_dump_and_print(self, monkeypatch, tmp_path):
        self._prime(monkeypatch, tmp_path, _print_enabled=True)
        t = torch.ones(2)
        io_dumper.dump_before("op", (t,), {"k": 1})
        io_dumper.dump_after("op", (t,), torch.zeros(2))
        io_dumper._wait_and_flush()
        op_dir = tmp_path / "rank_0000" / "step_0000" / "op"
        assert (op_dir / "input.json").is_file()
        assert (op_dir / "output.json").is_file()

    def test_print_outputs_only_fallback(self, monkeypatch, tmp_path):
        self._prime(monkeypatch, tmp_path, _print_enabled=True)
        io_dumper.dump_after("op", (torch.ones(1),), torch.zeros(1))
        io_dumper._wait_and_flush()

    def test_filtered_out(self, monkeypatch, tmp_path):
        self._prime(monkeypatch, tmp_path, _match_all=False, _op_filter={"other"})
        io_dumper.dump_before("op", (torch.ones(1),), {})
        assert "op" not in io_dumper._op_summary


class TestEnableDisableAndEnv:
    def _patch_activate(self, monkeypatch):
        seen = []
        monkeypatch.setattr(io_dumper, "_activate_hooks", lambda: seen.append("a"))
        return seen

    def test_enable_basic(self, monkeypatch, tmp_path):
        seen = self._patch_activate(monkeypatch)
        io_dumper.enable_io_dump(
            str(tmp_path / "dump"),
            ops={"op_a"},
            modules={"Linear"},
            layers={"0"},
            max_calls=3,
            step_range="1-3",
            with_torch_funcs=True,
            ranks={0},
            with_values=True,
            with_metas=True,
            with_print=True,
            bg_workers=2,
        )
        assert seen == ["a"]
        assert io_dumper._enabled is True
        assert io_dumper._match_all is False
        assert io_dumper._op_filter == {"op_a"}
        assert io_dumper._module_filter == {"Linear"}
        assert "model.layers.0" in io_dumper._layer_filter
        assert io_dumper._max_calls == 3
        assert io_dumper._step_range == (1, 4)
        assert io_dumper._rank_filter == {0}
        assert io_dumper._with_values is True
        assert io_dumper._with_metas is True
        assert io_dumper._print_enabled is True
        assert io_dumper._bg_workers == 2

    def test_enable_match_all_and_print_only(self, monkeypatch):
        self._patch_activate(monkeypatch)
        io_dumper.enable_io_dump(with_print=True)
        assert io_dumper._match_all is True
        assert io_dumper._print_enabled is True
        assert io_dumper._dump_dir == ""

    def test_enable_from_env(self, monkeypatch, tmp_path):
        self._patch_activate(monkeypatch)
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
        monkeypatch.setenv("VLLM_FL_IO_DUMP_BG_WORKERS", "3")
        io_dumper.enable_io_dump()
        assert io_dumper._op_filter == {"a", "b"}
        assert io_dumper._module_filter == {"Linear"}
        assert "model.layers.0" in io_dumper._layer_filter
        assert io_dumper._max_calls == 4
        assert io_dumper._torch_funcs_enabled is True
        assert io_dumper._torch_func_filter == {"matmul"}
        assert io_dumper._with_values is True
        assert io_dumper._with_metas is True
        assert io_dumper._bg_workers == 3

    def test_enable_bad_env_ignored(self, monkeypatch, tmp_path):
        self._patch_activate(monkeypatch)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "bad"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_MAX_CALLS", "not-int")
        monkeypatch.setenv("VLLM_FL_IO_DUMP_BG_WORKERS", "not-int")
        io_dumper.enable_io_dump()
        assert io_dumper._max_calls == 0
        assert io_dumper._bg_workers == 8

    def test_enable_from_yaml(self, monkeypatch, tmp_path):
        self._patch_activate(monkeypatch)
        ydir = tmp_path / "ydir"
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "io_dump:\n"
            f"  dir: '{ydir}'\n"
            "  ops: [rms_norm]\n"
            "  max_calls: 7\n"
            "  with_values: true\n"
            "  with_metas: true\n"
            "  with_print: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        io_dumper.enable_io_dump()
        assert io_dumper._op_filter == {"rms_norm"}
        assert io_dumper._max_calls == 7
        assert io_dumper._with_values is True
        assert io_dumper._with_metas is True
        assert io_dumper._print_enabled is True

    def test_api_overrides_env(self, monkeypatch, tmp_path):
        self._patch_activate(monkeypatch)
        monkeypatch.setenv("VLLM_FL_IO_DUMP", str(tmp_path / "e"))
        monkeypatch.setenv("VLLM_FL_IO_DUMP_OPS", "env_op")
        io_dumper.enable_io_dump(ops={"api_op"})
        assert io_dumper._op_filter == {"api_op"}

    def test_disable_resets_state(self, monkeypatch, tmp_path):
        self._patch_activate(monkeypatch)
        io_dumper.enable_io_dump(str(tmp_path / "d"))
        assert io_dumper._enabled is True
        io_dumper.disable_io_dump()
        assert io_dumper._enabled is False
        assert io_dumper._dump_dir == ""
        assert io_common.is_io_active() is False

    def test_set_and_clear_env_vars(self, tmp_path):
        io_dumper._set_env_vars(
            str(tmp_path),
            {"a"},
            {"Linear"},
            {"model.layers.0"},
            3,
            (0, 3),
            True,
            {1, 2},
            True,
            True,
            True,
        )
        assert os.environ["VLLM_FL_IO_DUMP_OPS"] == "a"
        assert os.environ["VLLM_FL_IO_DUMP_STEP_RANGE"] == "0-2"
        assert os.environ["VLLM_FL_IO_DUMP_RANK"] == "1,2"
        assert os.environ["VLLM_FL_IO_DUMP_WITH_PRINT"] == "1"
        io_dumper._set_env_vars(
            str(tmp_path), None, None, set(), 0, None, False, None, False, False, False
        )
        assert "VLLM_FL_IO_DUMP_OPS" not in os.environ
        assert "VLLM_FL_IO_DUMP_STEP_RANGE" not in os.environ
        io_dumper._clear_env_vars()
        assert "VLLM_FL_IO_DUMP" not in os.environ


class TestSummary:
    def test_format_dispatch_keys_for_json(self):
        out = io_dumper._format_dispatch_keys_for_json(
            [("CPU", "CPU", False), ("CUDA", "FlagGems", False)]
        )
        assert out == "[(CPU, CPU, False), (CUDA, FlagGems, False)]"

    def test_record_op_summary_counts(self):
        io_dumper._op_summary.clear()
        io_dumper._record_op_summary("op", [("CPU", "CPU", False)])
        io_dumper._record_op_summary("op", None)
        assert io_dumper._op_summary["op"]["call_count"] == 2
        assert "CPU" in io_dumper._op_summary["op"]["dispatch_keys"]

    def test_is_flaggems_op_string_hit(self):
        assert io_dumper._is_flaggems_op("op", "[(CUDA, FlagGems, False)]") is True

    def test_is_flaggems_op_manager_fallback(self, monkeypatch):
        impl = types.SimpleNamespace(impl_id="default.x", kind=BackendImplKind.DEFAULT)
        snap = types.SimpleNamespace(impls_by_op={"op": [impl]})
        fake_mgr = types.SimpleNamespace(
            _called_ops={"op": "default.x"},
            _registry=types.SimpleNamespace(snapshot=lambda: snap),
        )
        monkeypatch.setattr(manager_mod, "get_default_manager", lambda: fake_mgr)
        assert io_dumper._is_flaggems_op("op", "") is True
        impl.kind = BackendImplKind.REFERENCE
        assert io_dumper._is_flaggems_op("op", "") is False

    def test_is_flaggems_op_manager_exception(self, monkeypatch):
        def _raise():
            raise RuntimeError("no manager")

        monkeypatch.setattr(manager_mod, "get_default_manager", _raise)
        assert io_dumper._is_flaggems_op("op", "") is False

    def test_write_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        io_dumper._op_summary.clear()
        io_dumper._op_summary["flagop"] = {
            "dispatch_keys": "[(CUDA, FlagGems, False)]",
            "call_count": 1,
        }
        io_dumper._op_summary["plainop"] = {"dispatch_keys": "", "call_count": 2}
        io_dumper._write_summary()
        data = json.loads(
            (tmp_path / "rank_0000" / "summary.json").read_text(encoding="utf-8")
        )
        assert data["flaggems_ops"] == ["flagop"]
        assert data["non_flaggems_ops"] == ["plainop"]

    def test_write_summary_noop_without_dir_or_data(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_dump_dir", "")
        io_dumper._write_summary()
        monkeypatch.setattr(io_dumper, "_dump_dir", "/tmp/x")
        io_dumper._op_summary.clear()
        io_dumper._write_summary()

    def test_write_summary_oserror_branch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        io_dumper._op_summary.clear()
        io_dumper._op_summary["op"] = {"dispatch_keys": "", "call_count": 1}
        (tmp_path / "rank_0000" / "summary.json").mkdir(parents=True)
        io_dumper._write_summary()


class TestHookLifecycle:
    def test_enter_exit_modes_with_fake_managers(self, monkeypatch):
        events = []

        class _FakeMgr:
            def enter(self, name, inst):
                events.append(("enter", name))

            def request_exit(self, name):
                events.append(("exit", name))

        monkeypatch.setattr(io_dumper, "dispatch_mode_mgr", _FakeMgr())
        monkeypatch.setattr(io_dumper, "func_mode_mgr", _FakeMgr())
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        io_dumper._enter_dispatch_modes()
        io_dumper._exit_dispatch_modes()
        assert ("exit", "dump") in events

    def test_activate_hooks_immediate(self, monkeypatch):
        entered = []
        monkeypatch.setattr(
            io_dumper, "_enter_dispatch_modes", lambda: entered.append(1)
        )
        monkeypatch.setattr(io_dumper, "_step_range", None)
        io_dumper._hooks_activated = False
        io_dumper._activate_hooks()
        assert entered == [1]
        assert io_dumper._hooks_activated is True
        io_dumper._deactivate_hooks()
        assert io_dumper._hooks_activated is False

    def test_activate_hooks_deferred(self, monkeypatch):
        entered = []
        monkeypatch.setattr(
            io_dumper, "_enter_dispatch_modes", lambda: entered.append(1)
        )
        monkeypatch.setattr(io_dumper, "_step_range", (2, 5))
        io_dumper._hooks_activated = False
        io_dumper._activate_hooks()
        assert entered == []
        assert io_dumper._hooks_activated is True
        io_dumper._deactivate_hooks()

    def test_on_step_advance_lazy_activation(self, monkeypatch):
        entered = []
        monkeypatch.setattr(
            io_dumper, "_enter_dispatch_modes", lambda: entered.append(1)
        )
        monkeypatch.setattr(io_dumper, "_step_range", (2, 5))
        monkeypatch.setattr(io_dumper, "_dump_dir", "")
        monkeypatch.setattr(io_dumper, "_print_enabled", False)
        monkeypatch.setattr(
            io_dumper,
            "dispatch_mode_mgr",
            types.SimpleNamespace(is_entered=lambda n: False),
        )
        with io_dumper._lock:
            io_dumper._call_counters["op"] = 3
        io_dumper._on_step_advance(1, set(), set())
        assert entered == [1]
        assert io_dumper._call_counters == {}

    def test_on_step_advance_writes_and_prints(self, monkeypatch):
        waited, wrote = [], []
        monkeypatch.setattr(io_dumper, "_dump_dir", "/tmp/x")
        monkeypatch.setattr(io_dumper, "_print_enabled", True)
        monkeypatch.setattr(io_dumper, "_wait_and_flush", lambda: waited.append(1))
        monkeypatch.setattr(io_dumper, "_write_summary", lambda: wrote.append(1))
        io_dumper._on_step_advance(2, {"M"}, {"op"})
        assert waited == [1]
        assert wrote == [1]

    def test_pause_resume_dispatch_modes(self, monkeypatch):
        events = []

        class _Mgr:
            def __init__(self, entered):
                self._entered = entered

            def is_entered(self, name):
                return self._entered

            def request_exit(self, name):
                events.append(("exit", name))

        monkeypatch.setattr(io_dumper, "dispatch_mode_mgr", _Mgr(True))
        monkeypatch.setattr(io_dumper, "func_mode_mgr", _Mgr(False))
        io_dumper.pause_dispatch_modes()
        assert ("exit", "dump") in events

        entered = []
        monkeypatch.setattr(io_dumper, "dispatch_mode_mgr", _Mgr(False))
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_hooks_activated", True)
        monkeypatch.setattr(
            io_dumper, "_enter_dispatch_modes", lambda: entered.append(1)
        )
        io_dumper.resume_dispatch_modes()
        assert entered == [1]

        entered.clear()
        monkeypatch.setattr(io_dumper, "_enabled", False)
        io_dumper.resume_dispatch_modes()
        assert entered == []

    def test_register_and_remove_module_hooks(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", False)
        model = torch.nn.Linear(2, 2)
        io_dumper.register_io_module_hooks(model)
        assert io_dumper._module_hook_handles == []

        monkeypatch.setattr(io_dumper, "_enabled", True)
        io_dumper.register_io_module_hooks(model)
        assert io_dumper._module_hook_handles
        model(torch.ones(1, 2))
        io_dumper._remove_module_hooks()
        assert io_dumper._module_hook_handles == []

    def test_reset_state(self, monkeypatch):
        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_dump_dir", "/tmp/x")
        monkeypatch.setattr(io_dumper, "_match_all", True)
        io_dumper._reset_state()
        assert io_dumper._enabled is False
        assert io_dumper._dump_dir == ""
        assert io_dumper._match_all is False
        assert io_dumper._op_summary == {}


class TestTorchModes:
    @pytest.mark.skipif(
        not io_common.HAS_TORCH_DISPATCH_MODE, reason="TorchDispatchMode unavailable"
    )
    def test_dispatch_mode_paths(self, monkeypatch, tmp_path):
        mode = io_dumper._DumpDispatchMode()

        def make_op(result=None, raises=None):
            def fn(*a, **k):
                if raises is not None:
                    raise raises
                return result

            fn.name = lambda: "aten::__cov_mode_op__"
            fn.namespace = "aten"
            return fn

        monkeypatch.setattr(io_dumper, "_enabled", False)
        assert mode.__torch_dispatch__(make_op("r"), (), (), {}) == "r"

        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_rank_filter", None)
        monkeypatch.setattr(io_dumper, "_step_range", None)
        monkeypatch.setattr(io_dumper, "_match_all", False)
        monkeypatch.setattr(io_dumper, "_op_filter", {"other"})
        assert mode.__torch_dispatch__(make_op("r"), (), (), {}) == "r"

        monkeypatch.setattr(io_dumper, "_match_all", True)
        monkeypatch.setattr(io_dumper, "_op_filter", set())
        monkeypatch.setattr(io_dumper, "_with_metas", False)
        monkeypatch.setattr(io_dumper, "_layer_filter", set())
        monkeypatch.setattr(io_dumper, "_module_filter", set())
        io_dumper._op_summary.clear()
        assert mode.__torch_dispatch__(make_op("r"), (), (), {}) == "r"
        assert io_dumper._op_summary

        monkeypatch.setattr(io_dumper, "_with_metas", True)
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "_with_values", False)
        monkeypatch.setattr(io_dumper, "_print_enabled", False)
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        monkeypatch.setattr(io_dumper, "get_step", lambda: 0)
        monkeypatch.setattr(io_dumper, "_bg_workers", 1)
        with io_dumper._lock:
            io_dumper._call_counters.clear()
        result = mode.__torch_dispatch__(make_op(torch.ones(1)), (), (), {})
        assert isinstance(result, torch.Tensor)
        io_dumper._wait_and_flush()

        with pytest.raises(ValueError):
            mode.__torch_dispatch__(make_op(raises=ValueError("x")), (), (), {})
        io_dumper._wait_and_flush()

    @pytest.mark.skipif(
        not io_common.HAS_TORCH_FUNC_MODE, reason="TorchFunctionMode unavailable"
    )
    def test_torch_func_mode_paths(self, monkeypatch, tmp_path):
        mode = io_dumper._DumpTorchFuncMode()

        def make_func(name="matmul", result=None, raises=None):
            def fn(*a, **k):
                if raises is not None:
                    raise raises
                return result

            fn.__name__ = name
            return fn

        monkeypatch.setattr(
            torch, "compiler", types.SimpleNamespace(is_compiling=lambda: False), raising=False
        )
        monkeypatch.setattr(io_dumper, "_enabled", False)
        assert mode.__torch_function__(make_func(result="r"), (), (), {}) == "r"

        monkeypatch.setattr(io_dumper, "_enabled", True)
        monkeypatch.setattr(io_dumper, "_rank_filter", None)
        monkeypatch.setattr(io_dumper, "_step_range", None)
        monkeypatch.setattr(io_dumper, "_layer_filter", set())
        monkeypatch.setattr(io_dumper, "_torch_funcs_enabled", True)
        monkeypatch.setattr(io_dumper, "_torch_func_filter", set())
        monkeypatch.setattr(io_dumper, "_match_all", True)
        monkeypatch.setattr(io_dumper, "_op_filter", set())
        monkeypatch.setattr(io_dumper, "_module_filter", set())
        monkeypatch.setattr(io_dumper, "_with_metas", False)
        assert mode.__torch_function__(make_func(result="r"), (), (), {}) == "r"

        monkeypatch.setattr(io_dumper, "_with_metas", True)
        monkeypatch.setattr(io_dumper, "_dump_dir", str(tmp_path))
        monkeypatch.setattr(io_dumper, "_with_values", False)
        monkeypatch.setattr(io_dumper, "_print_enabled", False)
        monkeypatch.setattr(io_dumper, "get_rank", lambda: 0)
        monkeypatch.setattr(io_dumper, "get_step", lambda: 0)
        monkeypatch.setattr(io_dumper, "_bg_workers", 1)
        with io_dumper._lock:
            io_dumper._call_counters.clear()
        result = mode.__torch_function__(make_func(result=torch.ones(1)), (), (), {})
        assert isinstance(result, torch.Tensor)
        io_dumper._wait_and_flush()

        with pytest.raises(ValueError):
            mode.__torch_function__(make_func(raises=ValueError("x")), (), (), {})
        io_dumper._wait_and_flush()


# =========================================================================== #
# manager.py
# =========================================================================== #

class TestCanRetryFailedOp:
    def test_non_special_op(self, monkeypatch):
        monkeypatch.setattr(
            manager_mod, "get_policy", lambda: types.SimpleNamespace(strict=False)
        )
        assert _can_retry_failed_op("custom_op") is True

    def test_strict_blocks_retry(self, monkeypatch):
        monkeypatch.setattr(
            manager_mod, "get_policy", lambda: types.SimpleNamespace(strict=True)
        )
        assert _can_retry_failed_op("rms_norm") is False

    def test_special_op_import_error(self, monkeypatch):
        monkeypatch.setattr(
            manager_mod, "get_policy", lambda: types.SimpleNamespace(strict=False)
        )
        _install_fake_platform(monkeypatch, missing_current=True)
        assert _can_retry_failed_op("rms_norm") is True

    def test_special_op_amd_blocks(self, monkeypatch):
        monkeypatch.setattr(
            manager_mod, "get_policy", lambda: types.SimpleNamespace(strict=False)
        )
        _install_fake_platform(monkeypatch, types.SimpleNamespace(vendor_name="amd"))
        assert _can_retry_failed_op("rotary_embedding") is False

    def test_special_op_non_amd_allows(self, monkeypatch):
        monkeypatch.setattr(
            manager_mod, "get_policy", lambda: types.SimpleNamespace(strict=False)
        )
        _install_fake_platform(monkeypatch, types.SimpleNamespace(vendor_name="cuda"))
        assert _can_retry_failed_op("rotary_embedding") is True


class TestRecordDefaultFlagosOp:
    def test_writes_through_owned_handler(self, tmp_path):
        log_path = tmp_path / "oplist.txt"
        logger = logging.getLogger("flag_gems")
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler._flaggems_owned = True
        logger.addHandler(handler)
        try:
            impl = OpImpl(
                op_name="rms_norm",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=lambda: None,
            )
            _record_default_flagos_op("rms_norm", impl)
            handler.flush()
            content = log_path.read_text(encoding="utf-8")
            assert "vllm_fl.dispatch.ops.rms_norm: default.flagos" in content
            _record_default_flagos_op("rms_norm", impl)
            handler.flush()
            assert log_path.read_text(encoding="utf-8").count("default.flagos") == 1
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_handler_write_error_swallowed(self, tmp_path):
        log_path = tmp_path / "oplist2.txt"
        logger = logging.getLogger("flag_gems")
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler._flaggems_owned = True
        logger.addHandler(handler)
        original_stream = handler.stream

        class _BadStream:
            def write(self, *a, **k):
                raise RuntimeError("disk full")

            def flush(self):
                pass

        try:
            handler.stream = _BadStream()
            impl = OpImpl(
                op_name="bug_op",
                impl_id="default.x",
                kind=BackendImplKind.DEFAULT,
                fn=lambda: None,
            )
            _record_default_flagos_op("bug_op", impl)
        finally:
            handler.stream = original_stream
            logger.removeHandler(handler)
            handler.close()


class TestManagerLifecycle:
    def test_state_defaults(self):
        state = _OpManagerState()
        assert state.init_pid == -1
        assert state.initialized is False
        assert state.policy_epoch == 0

    def test_reset_after_fork(self):
        mgr = _make_manager()
        epoch = mgr._state.policy_epoch
        mgr._dispatch_cache[("k",)] = "v"
        mgr._candidates_cache[("k",)] = ["v"]
        mgr._called_ops["op"] = "impl"
        mgr._failed_impls["op"] = {"impl"}
        mgr._reset_after_fork()
        assert mgr._state.initialized is False
        assert mgr._state.init_pid == -1
        assert mgr._state.policy_epoch == epoch + 1
        assert mgr._dispatch_cache == {}
        assert mgr._candidates_cache == {}
        assert mgr._called_ops == {}
        assert mgr._failed_impls == {}

    def test_bump_policy_epoch_clears(self):
        mgr = _make_manager()
        epoch = mgr.policy_epoch
        mgr._failed_impls["op"] = {"x"}
        mgr.bump_policy_epoch()
        assert mgr.policy_epoch == epoch + 1
        assert mgr._failed_impls == {}

    def test_get_failed_impls_missing_op(self):
        mgr = _make_manager()
        assert mgr.get_failed_impls("missing") == {}

    def test_ensure_initialized_registers_once(self, monkeypatch):
        calls = []

        def fake_register(registry):
            calls.append(1)
            registry.register_impl(
                OpImpl(
                    op_name="dummy",
                    impl_id="default.dummy",
                    kind=BackendImplKind.DEFAULT,
                    fn=lambda: 1,
                )
            )

        monkeypatch.setattr(builtin_ops, "register_builtins", fake_register)
        mgr = OpManager(registry=OpRegistry())
        mgr.ensure_initialized()
        assert calls == [1]
        assert "dummy" in mgr.registry.list_operators()
        mgr.ensure_initialized()
        assert calls == [1]

    def test_ensure_initialized_debug_prints(self, monkeypatch):
        monkeypatch.setattr(builtin_ops, "register_builtins", lambda reg: None)
        monkeypatch.setattr(manager_mod, "_DISPATCH_DEBUG", True)
        printed = []
        monkeypatch.setattr(
            OpManager, "_print_registered_operators", lambda self: printed.append(1)
        )
        mgr = OpManager(registry=OpRegistry())
        mgr.ensure_initialized()
        assert printed == [1]

    def test_print_registered_operators(self):
        reg = OpRegistry()
        reg.register_impl(
            OpImpl(
                op_name="op",
                impl_id="default.a",
                kind=BackendImplKind.DEFAULT,
                fn=lambda: None,
            )
        )

        def unavailable():
            return False

        bad = lambda: None
        bad._is_available = unavailable
        reg.register_impl(
            OpImpl(
                op_name="op",
                impl_id="vendor.cuda",
                kind=BackendImplKind.VENDOR,
                fn=bad,
                vendor="CUDA",
            )
        )
        _make_manager(reg)._print_registered_operators()

    def test_get_selected_impl_id(self):
        reg = OpRegistry()
        _reg_impl(reg, "op", "default.a", BackendImplKind.DEFAULT, lambda: 1)
        assert _make_manager(reg).get_selected_impl_id("op") == "default.a"

    def test_default_manager_singleton(self):
        reset_default_manager()
        mgr1 = get_default_manager()
        mgr2 = get_default_manager()
        assert mgr1 is mgr2
        reset_default_manager()
        assert get_default_manager() is not mgr1


class TestManagerSelection:
    def test_matches_vendor_filters(self):
        mgr = _make_manager()
        policy = SelectionPolicy(deny_vendors=frozenset({"CUDA"}), allow_vendors=None)
        default_impl = OpImpl(
            op_name="op", impl_id="default.a", kind=BackendImplKind.DEFAULT, fn=lambda: 1
        )
        vendor_cuda = OpImpl(
            op_name="op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda: 1,
            vendor="CUDA",
        )
        vendor_amd = OpImpl(
            op_name="op",
            impl_id="vendor.amd",
            kind=BackendImplKind.VENDOR,
            fn=lambda: 1,
            vendor="AMD",
        )
        assert mgr._matches_vendor_filters(default_impl, policy) is True
        assert mgr._matches_vendor_filters(vendor_cuda, policy) is False
        assert mgr._matches_vendor_filters(vendor_amd, policy) is True

        allow_policy = SelectionPolicy(allow_vendors=frozenset({"AMD"}))
        assert mgr._matches_vendor_filters(vendor_cuda, allow_policy) is False
        assert mgr._matches_vendor_filters(vendor_amd, allow_policy) is True

        no_vendor = OpImpl(
            op_name="op",
            impl_id="v.none",
            kind=BackendImplKind.VENDOR,
            fn=lambda: 1,
            vendor="X",
        )
        object.__setattr__(no_vendor, "vendor", None)
        assert mgr._matches_vendor_filters(no_vendor, allow_policy) is False

    def test_resolve_candidates_order(self):
        reset_global_policy()
        reg = OpRegistry()
        _reg_impl(reg, "op", "default.low", BackendImplKind.DEFAULT, lambda: 1, priority=100)
        _reg_impl(reg, "op", "default.high", BackendImplKind.DEFAULT, lambda: 1, priority=150)
        _reg_impl(reg, "op", "reference.torch", BackendImplKind.REFERENCE, lambda: 1, priority=50)
        mgr = _make_manager(reg)
        ids = [c.impl_id for c in mgr.resolve_candidates("op")]
        assert ids == ["default.high", "default.low", "reference.torch"]

        set_global_policy(
            SelectionPolicy.from_dict(
                prefer=PREFER_DEFAULT,
                per_op_order={"op": ["reference", "flagos", "impl:default.high"]},
            )
        )
        ids2 = [c.impl_id for c in mgr.resolve_candidates("op")]
        assert ids2[0] == "reference.torch"
        assert ids2.count("default.high") == 1

    def test_compute_candidates_unavailable_raises(self):
        reg = OpRegistry()

        def unavailable():
            return False

        fn = lambda: 1
        fn._is_available = unavailable
        reg.register_impl(
            OpImpl(op_name="op", impl_id="default.a", kind=BackendImplKind.DEFAULT, fn=fn)
        )
        with pytest.raises(RuntimeError, match="No available implementation"):
            _make_manager(reg).resolve("op")

    def test_resolve_cached_and_epoch_invalidation(self):
        reset_global_policy()
        reg = OpRegistry()
        _reg_impl(reg, "op", "default.a", BackendImplKind.DEFAULT, lambda: 1)
        mgr = _make_manager(reg)
        assert mgr.resolve("op") is mgr.resolve("op")
        assert len(mgr._dispatch_cache) == 1
        mgr.bump_policy_epoch()
        assert mgr._dispatch_cache == {}

    def test_call_with_hooks_disabled(self, monkeypatch):
        mgr = _make_manager()
        monkeypatch.setattr(manager_mod, "is_dump_enabled", lambda: False)
        assert mgr._call_with_hooks("op", lambda x: x + 1, (1,), {}) == 2

    def test_call_with_hooks_enabled_hooks(self, monkeypatch):
        mgr = _make_manager()
        monkeypatch.setattr(manager_mod, "is_dump_enabled", lambda: True)
        monkeypatch.setattr(manager_mod, "next_exec_order", lambda: 9)
        monkeypatch.setattr(manager_mod, "make_module_tag", lambda: "[module=0,1]")
        monkeypatch.setattr(manager_mod, "make_op_tag", lambda name: "[op=0,1]")

        def bad(*a, **k):
            raise RuntimeError("hook failure")

        monkeypatch.setattr(manager_mod, "dump_before", bad)
        monkeypatch.setattr(manager_mod, "dump_after", bad)
        monkeypatch.setattr(manager_mod, "dump_cleanup", lambda n: None)
        assert mgr._call_with_hooks("op", lambda x: x + 1, (1,), {}) == 2

    def test_call_with_hooks_failure_cleans_up(self, monkeypatch):
        mgr = _make_manager()
        events = []
        monkeypatch.setattr(manager_mod, "is_dump_enabled", lambda: True)
        monkeypatch.setattr(manager_mod, "next_exec_order", lambda: 1)
        monkeypatch.setattr(manager_mod, "make_module_tag", lambda: "")
        monkeypatch.setattr(manager_mod, "make_op_tag", lambda n: "")
        monkeypatch.setattr(manager_mod, "dump_before", lambda *a, **k: None)
        monkeypatch.setattr(manager_mod, "dump_after", lambda *a, **k: None)
        monkeypatch.setattr(manager_mod, "dump_cleanup", lambda n: events.append(n))

        def boom(_x):
            raise ValueError("fail")

        with pytest.raises(ValueError):
            mgr._call_with_hooks("op", boom, (1,), {})
        assert events == ["op"]


class TestManagerCall:
    def test_call_fallback_to_second_impl(self):
        reset_global_policy()
        reg = OpRegistry()

        def bad():
            raise RuntimeError("first failed")

        _reg_impl(reg, "op", "default.bad", BackendImplKind.DEFAULT, bad, priority=150)
        _reg_impl(reg, "op", "default.good", BackendImplKind.DEFAULT, lambda: "ok", priority=100)
        mgr = _make_manager(reg)
        assert mgr.call("op") == "ok"
        assert mgr._failed_impls["op"] == {"default.bad"}
        assert mgr._called_ops["op"] == "default.good"

    def test_call_all_fail(self):
        reset_global_policy()
        reg = OpRegistry()

        def bad1():
            raise RuntimeError("1")

        def bad2():
            raise RuntimeError("2")

        _reg_impl(reg, "op", "default.b1", BackendImplKind.DEFAULT, bad1, priority=150)
        _reg_impl(reg, "op", "default.b2", BackendImplKind.DEFAULT, bad2, priority=100)
        with pytest.raises(RuntimeError, match="All 2 implementation"):
            _make_manager(reg).call("op")

    def test_call_all_previously_failed(self):
        reset_global_policy()
        reg = OpRegistry()
        _reg_impl(reg, "op", "default.a", BackendImplKind.DEFAULT, lambda: 1)
        mgr = _make_manager(reg)
        mgr._failed_impls["op"] = {"default.a"}
        with pytest.raises(RuntimeError, match="have failed previously"):
            mgr.call("op")

    def test_call_strict_no_fallback(self):
        reg = OpRegistry()

        def boom():
            raise ValueError("strict failure")

        _reg_impl(reg, "op", "default.a", BackendImplKind.DEFAULT, boom)
        mgr = _make_manager(reg)
        set_global_policy(SelectionPolicy(strict=True, prefer=PREFER_DEFAULT))
        with pytest.raises(ValueError):
            mgr.call("op")

    def test_call_resolves_and_calls_ok(self):
        reset_global_policy()
        reg = OpRegistry()
        _reg_impl(reg, "op", "default.a", BackendImplKind.DEFAULT, lambda x: x * 2)
        assert _make_manager(reg).call("op", 3) == 6


# =========================================================================== #
# policy.py
# =========================================================================== #

class TestSelectionPolicyModel:
    def test_invalid_prefer_raises(self):
        with pytest.raises(ValueError):
            SelectionPolicy(prefer="bogus")

    def test_from_dict_and_accessors(self):
        pol = SelectionPolicy.from_dict(
            prefer="VENDOR",
            strict=True,
            per_op_order={"b": ["reference"], "a": ["flagos", "vendor"]},
            deny_vendors={"cuda"},
            allow_vendors={"amd"},
        )
        assert pol.prefer == "vendor"
        assert pol.strict is True
        assert pol.get_per_op_order("a") == ["flagos", "vendor"]
        assert pol.get_per_op_order("missing") is None
        assert pol.per_op_order_dict == {"a": ["flagos", "vendor"], "b": ["reference"]}
        assert pol.is_vendor_allowed("amd") is True
        assert pol.is_vendor_allowed("cuda") is False
        assert pol.is_vendor_allowed("other") is False

    def test_default_order_all_prefers(self):
        assert SelectionPolicy(prefer=PREFER_DEFAULT).get_default_order() == [
            "flagos",
            "vendor",
            "reference",
        ]
        assert SelectionPolicy(prefer=PREFER_VENDOR).get_default_order() == [
            "vendor",
            "flagos",
            "reference",
        ]
        assert SelectionPolicy(prefer=PREFER_REFERENCE).get_default_order() == [
            "reference",
            "flagos",
            "vendor",
        ]

    def test_fingerprint_and_hash(self):
        pol = SelectionPolicy.from_dict(
            prefer="vendor",
            strict=True,
            per_op_order={"op": ["flagos"]},
            deny_vendors={"cuda"},
            allow_vendors={"amd"},
        )
        fp = pol.fingerprint()
        assert "prefer=vendor" in fp
        assert "st=1" in fp
        assert "allow=amd" in fp
        assert "deny=cuda" in fp
        assert "per=op=flagos" in fp
        same = SelectionPolicy.from_dict(
            prefer="vendor",
            strict=True,
            per_op_order={"op": ["flagos"]},
            deny_vendors={"cuda"},
            allow_vendors={"amd"},
        )
        assert hash(pol) == hash(same)
        assert pol == same

    def test_is_vendor_allowed_no_allow(self):
        pol = SelectionPolicy()
        assert pol.is_vendor_allowed("anything") is True


class TestPolicyManagerCore:
    def test_get_instance_singleton(self):
        inst1 = PolicyManager.get_instance()
        assert PolicyManager.get_instance() is inst1
        inst1.__init__()
        assert inst1.get_policy_epoch() >= 0

    def test_set_and_reset_global(self):
        reset_global_policy()
        old = set_global_policy(SelectionPolicy(prefer=PREFER_VENDOR))
        assert isinstance(old, SelectionPolicy)
        assert get_policy().prefer == PREFER_VENDOR
        prev = set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE))
        assert prev.prefer == PREFER_VENDOR
        reset_global_policy()
        assert get_policy().prefer == PREFER_DEFAULT

    def test_bump_policy_epoch_free_function(self):
        start = get_policy_epoch()
        assert bump_policy_epoch() == start + 1

    def test_create_policy_context(self):
        mgr = PolicyManager.get_instance()
        ctx = mgr.create_policy_context(SelectionPolicy(prefer=PREFER_VENDOR))
        with ctx:
            assert get_policy().prefer == PREFER_VENDOR


class TestPolicyParsingHelpers:
    def test_parse_csv_set(self):
        assert PolicyManager._parse_csv_set("") == set()
        assert PolicyManager._parse_csv_set("a, b ,c") == {"a", "b", "c"}

    def test_parse_per_op(self):
        assert PolicyManager._parse_per_op("") == {}
        assert PolicyManager._parse_per_op("bad; op=a|b; =x; op2=a||") == {
            "op": ["a", "b"],
            "op2": ["a"],
        }

    def test_parse_op_config(self):
        assert PolicyManager._parse_op_config({"op": "vendor"}) == {"op": ["vendor"]}
        with pytest.raises(ValueError):
            PolicyManager._parse_op_config({"op": "bogus"})


class TestPolicyFromConfig:
    def _write(self, tmp_path, text):
        p = tmp_path / "policy.yaml"
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            policy_from_config(str(tmp_path / "nope.yaml"))

    def test_invalid_yaml_raises(self, tmp_path):
        with pytest.raises(ValueError):
            policy_from_config(self._write(tmp_path, "prefer: [unclosed"))

    def test_full_config(self, tmp_path):
        path = self._write(
            tmp_path,
            "prefer: vendor\n"
            "strict: true\n"
            "deny_vendors:\n  - cuda\n"
            "allow_vendors: ascend\n"
            "op_backends:\n"
            "  rms_norm: [vendor, flagos]\n"
            "  silu_and_mul: 'vendor:cuda|flagos'\n",
        )
        pol = policy_from_config(path)
        assert pol.prefer == PREFER_VENDOR
        assert pol.strict is True
        assert pol.deny_vendors == frozenset({"cuda"})
        assert pol.allow_vendors == frozenset({"ascend"})
        assert pol.get_per_op_order("rms_norm") == ["vendor", "flagos"]
        assert pol.get_per_op_order("silu_and_mul") == ["vendor:cuda", "flagos"]

    def test_invalid_prefer_falls_back(self, tmp_path):
        pol = policy_from_config(self._write(tmp_path, "prefer: nonsense\n"))
        assert pol.prefer == PREFER_DEFAULT

    def test_empty_config_defaults(self, tmp_path):
        pol = policy_from_config(self._write(tmp_path, "# empty\n"))
        assert pol.prefer == PREFER_DEFAULT
        assert pol.strict is False
        assert pol.deny_vendors == frozenset()
        assert pol.allow_vendors is None


class TestPolicyFromEnv:
    def test_config_file_has_priority(self, tmp_path, monkeypatch):
        p = tmp_path / "c.yaml"
        p.write_text("prefer: reference\nstrict: true\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(p))
        pol = policy_from_env()
        assert pol.prefer == PREFER_REFERENCE
        assert pol.strict is True

    def test_env_overrides_platform(self, tmp_path, monkeypatch):
        plat = tmp_path / "plat.yaml"
        plat.write_text(
            "prefer: reference\n"
            "strict: false\n"
            "deny_vendors: [cuda]\n"
            "op_backends:\n  op: [reference]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("vllm_fl.dispatch.config.get_config_path", lambda: plat)
        monkeypatch.setattr(policy_mod, "get_op_config", lambda: None)
        monkeypatch.setenv("VLLM_FL_PREFER", "vendor")
        monkeypatch.setenv("VLLM_FL_STRICT", "1")
        monkeypatch.setenv("VLLM_FL_DENY_VENDORS", "x,y")
        monkeypatch.setenv("VLLM_FL_ALLOW_VENDORS", "a")
        monkeypatch.setenv("VLLM_FL_PER_OP", "op1=a|b")
        pol = policy_from_env()
        assert pol.prefer == PREFER_VENDOR
        assert pol.strict is True
        assert pol.deny_vendors == frozenset({"x", "y"})
        assert pol.allow_vendors == frozenset({"a"})
        assert pol.get_per_op_order("op1") == ["a", "b"]

    def test_invalid_strict_warns(self, monkeypatch):
        monkeypatch.setattr("vllm_fl.dispatch.config.get_config_path", lambda: None)
        monkeypatch.setattr(policy_mod, "get_op_config", lambda: None)
        monkeypatch.setenv("VLLM_FL_STRICT", "2")
        assert policy_from_env().strict is False

    def test_op_config_takes_precedence(self, monkeypatch):
        monkeypatch.setattr("vllm_fl.dispatch.config.get_config_path", lambda: None)
        monkeypatch.setattr(
            policy_mod, "get_op_config", lambda: {"rms_norm": "vendor"}
        )
        monkeypatch.setenv("VLLM_FL_PER_OP", "ignored=a")
        pol = policy_from_env()
        assert pol.get_per_op_order("rms_norm") == ["vendor"]

    def test_platform_per_op_used_when_no_env(self, tmp_path, monkeypatch):
        plat = tmp_path / "plat2.yaml"
        plat.write_text(
            "op_backends:\n  rms_norm: [reference, flagos]\n", encoding="utf-8"
        )
        monkeypatch.setattr("vllm_fl.dispatch.config.get_config_path", lambda: plat)
        monkeypatch.setattr(policy_mod, "get_op_config", lambda: None)
        assert policy_from_env().get_per_op_order("rms_norm") == [
            "reference",
            "flagos",
        ]

    def test_platform_load_error_warns(self, monkeypatch):
        mgr = PolicyManager.get_instance()
        monkeypatch.setattr(
            "vllm_fl.dispatch.config.get_config_path", lambda: "x.yaml"
        )

        def _raise(_path):
            raise RuntimeError("bad platform")

        monkeypatch.setattr(mgr, "_policy_from_config", _raise)
        monkeypatch.setattr(policy_mod, "get_op_config", lambda: None)
        assert mgr._policy_from_env().prefer == PREFER_DEFAULT


class TestPolicyContextManagers:
    def test_policy_context_enter_exit(self):
        custom = SelectionPolicy(prefer=PREFER_VENDOR)
        start = get_policy_epoch()
        with policy_context(custom):
            assert get_policy() is custom
            assert get_policy_epoch() > start
        assert get_policy().prefer == PREFER_DEFAULT

    def test_policy_context_exit_without_enter(self):
        ctx = _PolicyContext(PolicyManager.get_instance(), SelectionPolicy())
        ctx.__exit__(None, None, None)

    def test_with_strict_mode(self):
        with with_strict_mode():
            assert get_policy().strict is True
        assert get_policy().strict is False

    def test_with_preference(self):
        with with_preference("vendor"):
            assert get_policy().prefer == PREFER_VENDOR
        with pytest.raises(ValueError):
            with with_preference("bogus"):
                pass

    def test_with_allowed_and_denied_vendors(self):
        with with_allowed_vendors("cuda", "amd"):
            assert get_policy().allow_vendors == frozenset({"cuda", "amd"})
        with with_denied_vendors("cuda"):
            assert "cuda" in get_policy().deny_vendors


# =========================================================================== #
# discovery.py
# =========================================================================== #

class TestEntryPoints:
    def test_select_style(self, monkeypatch):
        class _EPs:
            def select(self, group):
                assert group == discovery.PLUGIN_GROUP
                return ["ep1"]

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EPs())
        assert discovery._get_entry_points() == ["ep1"]

    def test_dict_style(self, monkeypatch):
        monkeypatch.setattr(
            importlib.metadata,
            "entry_points",
            lambda: {discovery.PLUGIN_GROUP: ["ep1"]},
        )
        assert discovery._get_entry_points() == ["ep1"]

    def test_get_style(self, monkeypatch):
        class _EPs:
            def get(self, group, default):
                return ["ep1"]

        monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EPs())
        assert discovery._get_entry_points() == ["ep1"]

    def test_fallback_empty(self, monkeypatch):
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda: object())
        assert discovery._get_entry_points() == []

    def test_exception_returns_empty(self, monkeypatch):
        def _raise():
            raise RuntimeError("boom")

        monkeypatch.setattr(importlib.metadata, "entry_points", _raise)
        assert discovery._get_entry_points() == []


class TestCallRegisterFunction:
    def test_direct_callable(self):
        reg = MagicMock()
        fn = MagicMock()
        assert discovery._call_register_function(fn, reg, "src") is True
        fn.assert_called_once_with(reg)

    def test_direct_callable_error(self):
        fn = MagicMock(side_effect=RuntimeError("x"))
        assert discovery._call_register_function(fn, MagicMock(), "src") is False

    def test_class_with_register(self):
        reg = MagicMock()

        class Plugin:
            called = []

            @staticmethod
            def register(registry):
                Plugin.called.append(registry)

        assert discovery._call_register_function(Plugin, reg, "src") is True
        assert Plugin.called == [reg]

    def test_vllm_fl_register_error(self):
        class Plugin:
            @staticmethod
            def vllm_fl_register(registry):
                raise RuntimeError("x")

        assert discovery._call_register_function(Plugin(), MagicMock(), "src") is False

    def test_no_register_function(self):
        assert discovery._call_register_function(object(), MagicMock(), "src") is False


class TestDiscoverFunctions:
    def test_discover_from_entry_points_via_patch(self, monkeypatch):
        discovery.clear_discovered_plugins()

        class GoodMod:
            @staticmethod
            def register(registry):
                return None

        class EP:
            def __init__(self, name, loader):
                self.name = name
                self._loader = loader

            def load(self):
                return self._loader()

        eps = [
            EP("good", lambda: GoodMod()),
            EP("bad_load", lambda: (_ for _ in ()).throw(RuntimeError("x"))),
            EP("no_reg", lambda: object()),
        ]
        monkeypatch.setattr(discovery, "_get_entry_points", lambda: eps)
        loaded = discovery.discover_from_entry_points(MagicMock())
        assert loaded == 1
        discovered = discovery.get_discovered_plugins()
        assert len(discovered) == 3
        assert ("good", "entry_point", True) in discovered
        assert ("bad_load", "entry_point", False) in discovered
        assert ("no_reg", "entry_point", False) in discovered

    def test_discover_from_entry_points_empty(self, monkeypatch):
        monkeypatch.setattr(discovery, "_get_entry_points", lambda: [])
        assert discovery.discover_from_entry_points(MagicMock()) == 0

    def test_discover_from_env_modules(self, monkeypatch):
        discovery.clear_discovered_plugins()

        class GoodMod:
            @staticmethod
            def register(registry):
                return None

        def fake_import(name):
            if name == "missing":
                raise ImportError("missing")
            if name == "boom":
                raise RuntimeError("boom")
            if name == "good":
                return GoodMod()
            return object()

        monkeypatch.setattr(discovery.importlib, "import_module", fake_import)
        monkeypatch.setenv(discovery.PLUGIN_MODULES_ENV, "good,missing,boom,nofunc")
        loaded = discovery.discover_from_env_modules(MagicMock())
        assert loaded == 1
        sources = {(n, src, ok) for n, src, ok in discovery.get_discovered_plugins()}
        assert ("good", "env_module", True) in sources
        assert ("missing", "env_module", False) in sources
        assert ("boom", "env_module", False) in sources
        assert ("nofunc", "env_module", False) in sources

    def test_discover_from_env_modules_empty(self, monkeypatch):
        monkeypatch.delenv(discovery.PLUGIN_MODULES_ENV, raising=False)
        assert discovery.discover_from_env_modules(MagicMock()) == 0

    def test_discover_plugins_none_registry(self):
        assert discovery.discover_plugins(None) == 0

    def test_discover_plugins_aggregates(self, monkeypatch):
        monkeypatch.setattr(discovery, "discover_from_entry_points", lambda r: 2)
        monkeypatch.setattr(discovery, "discover_from_env_modules", lambda r: 3)
        assert discovery.discover_plugins(MagicMock()) == 5

    def test_get_and_clear(self):
        discovery.clear_discovered_plugins()
        a = discovery.get_discovered_plugins()
        b = discovery.get_discovered_plugins()
        assert a is not b
        assert a == []


# =========================================================================== #
# config/utils.py
# =========================================================================== #

class TestConfigPath:
    def test_alias_and_unknown(self):
        assert cfg_utils.get_config_path("mthreads") == cfg_utils.get_config_path("musa")
        assert cfg_utils.get_config_path("gcu").name == "enflame.yaml"
        assert cfg_utils.get_config_path("definitely_unknown_platform") is None

    def test_nvidia_plain(self, monkeypatch):
        monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
        path = cfg_utils.get_config_path("nvidia")
        assert path is not None and path.name == "nvidia.yaml"

    def test_nvidia_hopper_enabled(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        platform = types.SimpleNamespace(
            get_device_capability=lambda: types.SimpleNamespace(major=9, minor=0)
        )
        _install_fake_platform(monkeypatch, platform)
        path = cfg_utils.get_config_path("nvidia")
        assert path is not None and path.name == "nvidia_hopper.yaml"

    def test_nvidia_non_hopper_capability(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "true")
        platform = types.SimpleNamespace(
            get_device_capability=lambda: types.SimpleNamespace(major=8, minor=0)
        )
        _install_fake_platform(monkeypatch, platform)
        assert cfg_utils.get_config_path("nvidia").name == "nvidia.yaml"

    def test_nvidia_capability_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        _install_fake_platform(
            monkeypatch, types.SimpleNamespace(get_device_capability=lambda: None)
        )
        assert cfg_utils.get_config_path("nvidia").name == "nvidia.yaml"

    def test_nvidia_capability_probe_error(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        platform = types.SimpleNamespace(
            get_device_capability=lambda: (_ for _ in ()).throw(RuntimeError("no gpu"))
        )
        _install_fake_platform(monkeypatch, platform)
        assert cfg_utils.get_config_path("nvidia").name == "nvidia.yaml"

    def test_platform_name_detection(self, monkeypatch):
        _install_fake_platform(monkeypatch, types.SimpleNamespace(vendor_name="acme"))
        assert cfg_utils.get_platform_name() == "acme"

    def test_platform_name_import_error_env_override(self, monkeypatch):
        _install_fake_platform(monkeypatch, missing_current=True)
        monkeypatch.setenv("VLLM_FL_PLATFORM", "  MUSA  ")
        assert cfg_utils.get_platform_name() == "musa"

    def test_platform_name_unknown(self, monkeypatch):
        _install_fake_platform(monkeypatch, missing_current=True)
        monkeypatch.delenv("VLLM_FL_PLATFORM", raising=False)
        assert cfg_utils.get_platform_name() == "unknown"


class TestLoadConfig:
    def test_load_missing(self, monkeypatch):
        monkeypatch.setattr(cfg_utils, "get_config_path", lambda *a, **k: None)
        assert cfg_utils.load_platform_config() is None

    def test_load_valid(self, tmp_path, monkeypatch):
        p = tmp_path / "cfg.yaml"
        p.write_text("prefer: flagos\nextra: 1\n", encoding="utf-8")
        monkeypatch.setattr(cfg_utils, "get_config_path", lambda *a, **k: p)
        assert cfg_utils.load_platform_config() == {"prefer": "flagos", "extra": 1}

    def test_load_invalid_yaml(self, tmp_path, monkeypatch):
        p = tmp_path / "bad.yaml"
        p.write_text("a: [unclosed", encoding="utf-8")
        monkeypatch.setattr(cfg_utils, "get_config_path", lambda *a, **k: p)
        assert cfg_utils.load_platform_config() is None

    def test_load_non_dict_scalar(self, tmp_path, monkeypatch):
        p = tmp_path / "scalar.yaml"
        p.write_text("123", encoding="utf-8")
        monkeypatch.setattr(cfg_utils, "get_config_path", lambda *a, **k: p)
        assert cfg_utils.load_platform_config() is None


class TestConfigExtractors:
    def test_per_op_order(self, monkeypatch):
        monkeypatch.setattr(cfg_utils, "load_platform_config", lambda *a, **k: None)
        assert cfg_utils.get_per_op_order() is None
        assert cfg_utils.get_per_op_order({"per_op": "bad"}) is None
        assert cfg_utils.get_per_op_order({"per_op": {}}) is None
        out = cfg_utils.get_per_op_order(
            {"per_op": {"a": ["x", "y"], "b": "z", "c": 5}}
        )
        assert out == {"a": ["x", "y"], "b": ["z"]}

    def test_blacklist_whitelist_oot(self, monkeypatch):
        monkeypatch.setattr(cfg_utils, "load_platform_config", lambda *a, **k: None)
        assert cfg_utils.get_flagos_blacklist() is None
        assert cfg_utils.get_flagos_whitelist() is None
        assert cfg_utils.get_oot_blacklist() is None
        assert cfg_utils.get_flagos_blacklist({"flagos_blacklist": "bad"}) is None
        assert cfg_utils.get_flagos_blacklist({"flagos_blacklist": ["a", 1]}) == [
            "a",
            "1",
        ]
        assert cfg_utils.get_flagos_whitelist({"flagos_whitelist": ["w"]}) == ["w"]
        assert cfg_utils.get_oot_blacklist({"oot_blacklist": ["o"]}) == ["o"]

    def test_effective_config(self, tmp_path, monkeypatch):
        user = tmp_path / "user.yaml"
        user.write_text("prefer: vendor\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(user))
        assert cfg_utils.get_effective_config() == {"prefer": "vendor"}

        bad = tmp_path / "bad.yaml"
        bad.write_text("a: [unclosed", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(bad))
        monkeypatch.setattr(cfg_utils, "load_platform_config", lambda *a, **k: {"k": 1})
        assert cfg_utils.get_effective_config() == {"k": 1}

        monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
        monkeypatch.setattr(cfg_utils, "load_platform_config", lambda *a, **k: None)
        assert cfg_utils.get_effective_config() == {}

    def test_vendor_device_map(self, monkeypatch):
        monkeypatch.setattr(cfg_utils, "VENDOR_DEVICE_MAP", "not-a-dict")
        assert cfg_utils.get_vendor_device_map() == {}

        monkeypatch.setattr(
            cfg_utils,
            "VENDOR_DEVICE_MAP",
            {
                "good": {"device_type": "cuda", "device_name": "gpu"},
                1: {"device_type": "cuda", "device_name": "gpu"},
                "nondict": "bad",
                "missing_type": {"device_name": "x"},
                "missing_name": {"device_type": "cuda"},
            },
        )
        assert cfg_utils.get_vendor_device_map() == {
            "good": {"device_type": "cuda", "device_name": "gpu"}
        }


# =========================================================================== #
# builtin_ops.py
# =========================================================================== #

class TestFindVendorBackendDir:
    def test_mapping_candidates(self):
        assert builtin_ops._find_vendor_backend_dir("nvidia", {"cuda"}) == "cuda"
        assert builtin_ops._find_vendor_backend_dir("nvidia", {"nvidia"}) == "nvidia"
        assert builtin_ops._find_vendor_backend_dir("maca", {"metax"}) == "metax"
        assert builtin_ops._find_vendor_backend_dir("unknown_vendor", {"cuda"}) is None
        assert builtin_ops._find_vendor_backend_dir("nvidia", {"amd"}) is None


class TestCurrentVendorDetection:
    def test_success(self, monkeypatch):
        _install_fake_platform(monkeypatch, types.SimpleNamespace(vendor_name="nvidia"))
        assert builtin_ops._get_current_vendor_backend_dirs({"cuda"}) == "cuda"

    def test_non_string_vendor(self, monkeypatch):
        _install_fake_platform(monkeypatch, types.SimpleNamespace(vendor_name=None))
        assert builtin_ops._get_current_vendor_backend_dirs({"cuda"}) is None
        _install_fake_platform(monkeypatch, types.SimpleNamespace(vendor_name=""))
        assert builtin_ops._get_current_vendor_backend_dirs({"cuda"}) is None

    def test_import_failure_raises_runtime_error(self, monkeypatch):
        _install_fake_platform(monkeypatch, missing_current=True)
        with pytest.raises(RuntimeError):
            builtin_ops._get_current_vendor_backend_dirs({"cuda"})


class TestRegisterVendorBackends:
    def test_missing_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(builtin_ops, "_VENDOR_BACKENDS_DIR", str(tmp_path / "nope"))
        builtin_ops._register_vendor_backends(OpRegistry())

    def test_no_current_vendor(self, monkeypatch, tmp_path):
        vdir = tmp_path / "vendor"
        (vdir / "metax").mkdir(parents=True)
        monkeypatch.setattr(builtin_ops, "_VENDOR_BACKENDS_DIR", str(vdir))
        monkeypatch.setattr(
            builtin_ops, "_get_current_vendor_backend_dirs", lambda dirs: None
        )
        builtin_ops._register_vendor_backends(OpRegistry())

    def test_imports_only_current_vendor(self, monkeypatch, tmp_path):
        vdir = tmp_path / "vendor"
        for name in ("metax", "cuda", "_private"):
            (vdir / name).mkdir(parents=True)
            (vdir / name / "register_ops.py").write_text("", encoding="utf-8")
        (vdir / "ascend").mkdir()
        monkeypatch.setattr(builtin_ops, "_VENDOR_BACKENDS_DIR", str(vdir))
        monkeypatch.setattr(
            builtin_ops, "_get_current_vendor_backend_dirs", lambda dirs: "metax"
        )
        imported = []

        class FakeModule:
            def __init__(self):
                self.registrations = []

            def register_builtins(self, registry):
                self.registrations.append(registry)

        fake = FakeModule()

        def fake_import(module_name, package=None):
            imported.append(module_name)
            return fake

        monkeypatch.setattr(builtin_ops.importlib, "import_module", fake_import)
        registry = OpRegistry()
        builtin_ops._register_vendor_backends(registry)
        assert imported == [".backends.vendor.metax.register_ops"]
        assert fake.registrations == [registry]

    def test_missing_register_ops_file(self, monkeypatch, tmp_path):
        vdir = tmp_path / "vendor"
        (vdir / "metax").mkdir(parents=True)
        monkeypatch.setattr(builtin_ops, "_VENDOR_BACKENDS_DIR", str(vdir))
        monkeypatch.setattr(
            builtin_ops, "_get_current_vendor_backend_dirs", lambda dirs: "metax"
        )
        called = []
        monkeypatch.setattr(
            builtin_ops.importlib, "import_module", lambda *a, **k: called.append(1)
        )
        builtin_ops._register_vendor_backends(OpRegistry())
        assert called == []

    def test_import_error_swallowed(self, monkeypatch, tmp_path):
        vdir = tmp_path / "vendor"
        (vdir / "metax").mkdir(parents=True)
        (vdir / "metax" / "register_ops.py").write_text("", encoding="utf-8")
        monkeypatch.setattr(builtin_ops, "_VENDOR_BACKENDS_DIR", str(vdir))
        monkeypatch.setattr(
            builtin_ops, "_get_current_vendor_backend_dirs", lambda dirs: "metax"
        )

        def _raise(*a, **k):
            raise ImportError("vendor deps missing")

        monkeypatch.setattr(builtin_ops.importlib, "import_module", _raise)
        builtin_ops._register_vendor_backends(OpRegistry())  # swallowed

    def test_module_without_register_builtins(self, monkeypatch, tmp_path):
        vdir = tmp_path / "vendor"
        (vdir / "metax").mkdir(parents=True)
        (vdir / "metax" / "register_ops.py").write_text("", encoding="utf-8")
        monkeypatch.setattr(builtin_ops, "_VENDOR_BACKENDS_DIR", str(vdir))
        monkeypatch.setattr(
            builtin_ops, "_get_current_vendor_backend_dirs", lambda dirs: "metax"
        )
        monkeypatch.setattr(
            builtin_ops.importlib, "import_module", lambda *a, **k: object()
        )
        builtin_ops._register_vendor_backends(OpRegistry())


class TestRegisterBuiltins:
    def test_registers_reference_and_vendor(self, monkeypatch):
        vendor_calls = []
        monkeypatch.setattr(
            builtin_ops,
            "_register_vendor_backends",
            lambda registry: vendor_calls.append(registry),
        )
        monkeypatch.setattr(discovery, "discover_plugins", lambda registry: 1)
        registry = OpRegistry()
        builtin_ops.register_builtins(registry)
        assert vendor_calls == [registry]
        assert registry.list_operators()  # reference registration succeeded

    def test_plugin_discovery_error_swallowed(self, monkeypatch):
        monkeypatch.setattr(
            builtin_ops, "_register_vendor_backends", lambda registry: None
        )

        def _raise(_registry):
            raise RuntimeError("discovery boom")

        monkeypatch.setattr(discovery, "discover_plugins", _raise)
        builtin_ops.register_builtins(OpRegistry())

    def test_flaggems_import_failure_swallowed(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "vllm_fl.dispatch.backends.flaggems.register_ops", None
        )
        monkeypatch.setattr(
            builtin_ops, "_register_vendor_backends", lambda registry: None
        )
        monkeypatch.setattr(discovery, "discover_plugins", lambda registry: 0)
        builtin_ops.register_builtins(OpRegistry())

    def test_reference_import_failure_swallowed(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "vllm_fl.dispatch.backends.reference.register_ops", None
        )
        monkeypatch.setattr(
            builtin_ops, "_register_vendor_backends", lambda registry: None
        )
        monkeypatch.setattr(discovery, "discover_plugins", lambda registry: 0)
        builtin_ops.register_builtins(OpRegistry())


# =========================================================================== #
# ops.py
# =========================================================================== #

class _ConcreteBackend(VLLMFLBackendBase):
    def is_available(self):
        return super().is_available()

    @property
    def name(self):
        return VLLMFLBackendBase.name.fget(self)

    @property
    def vendor(self):
        return super().vendor

    def dynamic_per_token_quant_int8(self, x):
        return super().dynamic_per_token_quant_int8(x)

    def silu_and_mul(self, x):
        return super().silu_and_mul(x)

    def gelu_and_mul(self, x, approximate="none"):
        return super().gelu_and_mul(x, approximate)

    def rms_norm(self, x, residual, weight, epsilon):
        return super().rms_norm(x, residual, weight, epsilon)

    def rotary_embedding(
        self,
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved=False,
        inplace=True,
    ):
        return super().rotary_embedding(
            query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )

    def attention_backend(self, use_mla=False, use_sparse=False):
        return super().attention_backend(use_mla, use_sparse)


class TestBackendBase:
    def test_abstract_method_bodies_and_vendor(self):
        backend = _ConcreteBackend()
        assert backend.is_available() is None
        assert backend.name is None
        assert backend.vendor is None
        assert backend.dynamic_per_token_quant_int8(torch.ones(1)) is None
        assert backend.silu_and_mul(torch.ones(1)) is None
        assert backend.gelu_and_mul(torch.ones(1)) is None
        assert backend.rms_norm(torch.ones(1), None, torch.ones(1), 1e-6) is None
        assert (
            backend.rotary_embedding(
                torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1)
            )
            is None
        )
        assert backend.attention_backend() is None

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VLLMFLBackendBase()

        class _Incomplete(VLLMFLBackendBase):
            def is_available(self):
                return True

            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            _Incomplete()


# =========================================================================== #
# __init__.py (CachedOp / call_op / resolve_op)
# =========================================================================== #

class _FakeImpl:
    def __init__(self, impl_id="impl.x", result=("impl",), raises=None):
        self.impl_id = impl_id

        def fn(*args, **kwargs):
            if raises is not None:
                raise raises
            return result + args

        self.fn = fn


class _FakeManager:
    def __init__(self, impl, *, epoch=0):
        self.policy_epoch = epoch
        self._impl = impl
        self.resolve_calls = 0
        self.recorded = []
        self.failed = []
        self.call_args = []

    def _resolve_impl(self, name):
        self.resolve_calls += 1
        return self._impl

    def _record_first_use(self, name, impl):
        self.recorded.append((name, impl))

    def call(self, name, *args, **kwargs):
        self.call_args.append((name, args, kwargs))
        return ("manager-call", name)

    def resolve(self, name):
        return self._impl.fn

    def _mark_failed_impl(self, name, impl_id):
        self.failed.append((name, impl_id))

    def ensure_initialized(self):
        return None


class TestCachedOp:
    def _patch(self, monkeypatch, mgr):
        epoch = {"value": mgr.policy_epoch}
        monkeypatch.setattr(dispatch_init, "get_default_manager", lambda: mgr)
        monkeypatch.setattr(
            dispatch_init, "get_policy_epoch", lambda: epoch["value"]
        )
        monkeypatch.setattr(dispatch_init, "is_dump_enabled", lambda: False)
        monkeypatch.setattr(dispatch_init, "_OP_FAST_PATH_ENABLED", True)
        monkeypatch.setattr(dispatch_init, "_can_retry_failed_op", lambda name: True)
        return epoch

    def test_warmup(self, monkeypatch):
        impl = _FakeImpl()
        mgr = _FakeManager(impl)
        self._patch(monkeypatch, mgr)
        op = dispatch_init.CachedOp("op")
        op.warmup(mgr)
        assert op._impl is impl
        assert mgr.recorded and mgr.recorded[0][0] == "op"

    def test_call_caches_impl(self, monkeypatch):
        impl = _FakeImpl()
        mgr = _FakeManager(impl)
        epoch = self._patch(monkeypatch, mgr)
        op = dispatch_init.CachedOp("op")
        assert op(5) == ("impl", 5)
        assert mgr.resolve_calls == 1
        assert op(6) == ("impl", 6)
        assert mgr.resolve_calls == 1

        epoch["value"] = 1
        assert op(7) == ("impl", 7)
        assert mgr.resolve_calls == 2

    def test_fast_path_disabled_and_dump(self, monkeypatch):
        impl = _FakeImpl()
        mgr = _FakeManager(impl)
        self._patch(monkeypatch, mgr)
        op = dispatch_init.CachedOp("op")
        monkeypatch.setattr(dispatch_init, "_OP_FAST_PATH_ENABLED", False)
        assert op(1) == ("manager-call", "op")
        monkeypatch.setattr(dispatch_init, "_OP_FAST_PATH_ENABLED", True)
        monkeypatch.setattr(dispatch_init, "is_dump_enabled", lambda: True)
        assert op(2) == ("manager-call", "op")
        assert len(mgr.call_args) == 2

    def test_use_manager_call_flag(self, monkeypatch):
        impl = _FakeImpl()
        mgr = _FakeManager(impl)
        self._patch(monkeypatch, mgr)
        op = dispatch_init.CachedOp("op")
        assert op(1) == ("impl", 1)
        op._use_manager_call = True
        assert op(2) == ("manager-call", "op")

    def test_failure_retry_allowed(self, monkeypatch):
        impl = _FakeImpl(raises=ValueError("fail"))
        mgr = _FakeManager(impl)
        self._patch(monkeypatch, mgr)
        op = dispatch_init.CachedOp("op")
        assert op(1) == ("manager-call", "op")
        assert mgr.failed == [("op", "impl.x")]

    def test_failure_retry_blocked(self, monkeypatch):
        impl = _FakeImpl(raises=ValueError("fail"))
        mgr = _FakeManager(impl)
        self._patch(monkeypatch, mgr)
        monkeypatch.setattr(dispatch_init, "_can_retry_failed_op", lambda name: False)
        op = dispatch_init.CachedOp("op")
        with pytest.raises(ValueError):
            op(1)

    def test_amd_compile_ready(self, monkeypatch):
        impl = _FakeImpl()
        mgr = _FakeManager(impl)
        self._patch(monkeypatch, mgr)
        monkeypatch.setattr(
            torch, "compiler", types.SimpleNamespace(is_compiling=lambda: True), raising=False
        )
        op = dispatch_init.CachedOp("op")
        op._impl = impl
        op._amd_compile_ready = True
        assert op(3) == ("impl", 3)
        op._impl = None
        with pytest.raises(RuntimeError):
            op(3)

    def test_warmup_cached_ops(self, monkeypatch):
        impl = _FakeImpl()
        weak_cache = weakref.WeakSet()
        monkeypatch.setattr(dispatch_init, "_CACHED_OPS", weak_cache)
        op_good = dispatch_init.CachedOp("good")
        dispatch_init.CachedOp("bad")

        class _Mgr(_FakeManager):
            def _resolve_impl(self, name):
                if name == "bad":
                    raise RuntimeError("unavailable")
                return impl

        mgr = _Mgr(impl)
        monkeypatch.setattr(dispatch_init, "get_default_manager", lambda: mgr)
        monkeypatch.setattr(dispatch_init, "get_policy_epoch", lambda: 0)
        monkeypatch.setattr(dispatch_init, "is_dump_enabled", lambda: False)
        monkeypatch.setattr(dispatch_init, "_OP_FAST_PATH_ENABLED", True)
        dispatch_init.warmup_cached_ops(mgr, amd_compile=True)
        assert op_good._impl is impl
        assert op_good._amd_compile_ready is True

    def test_call_op_and_resolve_op(self, monkeypatch):
        impl = _FakeImpl()
        mgr = _FakeManager(impl)
        monkeypatch.setattr(dispatch_init, "get_default_manager", lambda: mgr)
        assert dispatch_init.call_op("op", 1, k=2) == ("manager-call", "op")
        assert dispatch_init.resolve_op("op") is impl.fn


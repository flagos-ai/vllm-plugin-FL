# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU controls for the production native-graph admission wrapper.

Native graph APIs are CPU doubles. Input/alias and borrowed-storage contracts
are separately covered by the unchanged original stream/weak-reference tests.
"""

import __future__

import ast
import builtins
import dataclasses
import gc
import importlib.util
import os
import sys
import threading
import time
import unittest
import weakref
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PRODUCTION = (
    Path(sys.argv.pop(1)) if __name__ == "__main__" and len(sys.argv) > 1 else ROOT
)
VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/musa/compilation"
spec = importlib.util.spec_from_file_location(
    "_admission_budget_test", VENDOR / "musa_graph_admission.py"
)
budget_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget_module)
Budget = budget_module._MusaGraphAdmissionBudget


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class FakeTensor:
    shape = (1,)
    dtype = "float32"
    device = SimpleNamespace(type="musa", index=0)
    layout = "strided"
    requires_grad = False

    def __init__(self, value=0):
        self.value = value

    def stride(self):
        return (1,)

    def storage_offset(self):
        return 0

    def data_ptr(self):
        return id(self)


class FakeGraph:
    def __init__(self):
        self.replays = 0
        self.error = None

    def replay(self):
        if self.error:
            raise self.error
        self.replays += 1


def build(cap=2, mode=Mode.FULL):
    events = []
    ctx = SimpleNamespace(batch_descriptor="one", cudagraph_runtime_mode=mode)
    budget = Budget(cap)

    @contextmanager
    def capture(*args, **kwargs):
        events.append("native_capture")
        yield

    @contextmanager
    def device(ordinal):
        events.append(("device_enter", ordinal))
        try:
            yield
        finally:
            events.append(("device_exit", ordinal))

    def deny_vendor_import(name, *args, **kwargs):
        if name.startswith(("vllm", "torch")):
            raise ImportError("No target dependency imports in CPU controls")
        return builtins.__import__(name, *args, **kwargs)

    common = ast.parse((PRODUCTION / "vllm_fl/compilation/graph.py").read_text())
    musa = ast.parse((VENDOR / "graph.py").read_text())
    common_names = {"GraphEntry", "GraphOptions", "GraphWrapper"}
    musa_names = {"MUSAGraphEntry", "MUSAGraphWrapper"}
    nodes = [
        n for n in common.body if isinstance(n, ast.ClassDef) and n.name in common_names
    ]
    nodes += [
        n for n in musa.body if isinstance(n, ast.ClassDef) and n.name in musa_names
    ]
    native = SimpleNamespace(
        MUSAGraph=FakeGraph, current_device=lambda: 3, device=device
    )
    ns = dict(
        __name__=__name__,
        __builtins__=dict(vars(builtins), __import__=deny_vendor_import),
        dataclasses=dataclasses,
        weakref=weakref,
        ExitStack=ExitStack,
        nullcontext=nullcontext,
        suppress=suppress,
        os=os,
        sys=sys,
        CUDAGraphMode=Mode,
        Graph=SimpleNamespace(graph=FakeGraph),
        torch=SimpleNamespace(Tensor=FakeTensor, musa=native),
        envs=SimpleNamespace(VLLM_LOGGING_LEVEL="INFO"),
        current_platform=SimpleNamespace(
            get_global_graph_pool=lambda: None,
            graph_pool_handle=lambda: object(),
            torch_device_fn=SimpleNamespace(graph=capture),
        ),
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: ctx,
        validate_cudagraph_capturing_enabled=lambda: None,
        set_graph_pool_id=lambda pool: None,
        _graph_capture_stream=lambda: object(),
        _weak_ref_musa_tensors=lambda value: value,
        weak_ref_tensors=lambda value: value,
        tree_flatten=lambda value: (
            list(value[0]) + list(value[1].values()),
            "same_spec",
        ),
        compilation_counter=SimpleNamespace(num_cudagraph_captured=0),
        init_logger=lambda name: SimpleNamespace(
            info=lambda *args: events.append(("log", args))
        ),
        logger=SimpleNamespace(debug=lambda *args: None),
        _musa_admission_budget=budget,
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(VENDOR / "graph.py"),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )

    def runnable(value=None):
        events.append("runnable")
        if isinstance(value, FakeTensor):
            value.value += 1
            return value
        return object()

    wrapper = ns["MUSAGraphWrapper"](
        runnable,
        SimpleNamespace(compilation_config=object()),
        mode,
        ns["GraphOptions"](debug_log_enable=False, weak_ref_output=False),
    )
    return wrapper, ctx, budget, events, ns


class AdmissionControls(unittest.TestCase):
    def test_public_native_class_and_shared_common_registry(self):
        wrapper, _, _, _, ns = build()
        self.assertIs(type(wrapper), ns["MUSAGraphWrapper"])
        self.assertIs(wrapper._graph_type, FakeGraph)
        self.assertEqual(type(wrapper).__bases__, (ns["GraphWrapper"],))
        self.assertIs(type(wrapper)._all_instances, ns["GraphWrapper"]._all_instances)
        self.assertIn(wrapper, ns["GraphWrapper"]._all_instances)
        entry_fields = {
            field.name for field in dataclasses.fields(ns["MUSAGraphEntry"])
        }
        self.assertIn("musa_eager_only", entry_fields)
        self.assertIn("musa_capture_device", entry_fields)

    def test_cap_permanent_eager_and_second_round_after_clear(self):
        wrapper, ctx, budget, events, _ = build()
        for key in ("a", "b", "c"):
            ctx.batch_descriptor = key
            wrapper()
        self.assertEqual(budget.snapshot()["live"], 2)
        self.assertEqual(events.count("native_capture"), 2)
        denied = wrapper.concrete_graph_entries["c"]
        self.assertTrue(denied.musa_eager_only)
        external = wrapper.concrete_graph_entries["a"].graph
        wrapper.clear_graphs()
        self.assertEqual(budget.snapshot()["live"], 1)
        del external
        gc.collect()
        self.assertEqual(budget.snapshot()["live"], 0)
        for key in ("d", "e"):
            ctx.batch_descriptor = key
            wrapper()
        self.assertEqual(budget.snapshot()["live"], 2)
        self.assertEqual(events.count("native_capture"), 4)

    def test_denied_entry_never_retries_after_capacity_frees(self):
        wrapper, ctx, budget, events, _ = build(cap=1)
        wrapper()
        ctx.batch_descriptor = "denied"
        wrapper()
        del wrapper.concrete_graph_entries["one"]
        gc.collect()
        self.assertEqual(budget.snapshot()["live"], 0)
        wrapper()
        self.assertEqual(events.count("native_capture"), 1)

    def test_replay_device_and_fresh_input_mutation_alias(self):
        wrapper, _, budget, events, _ = build(mode=Mode.PIECEWISE)
        original = FakeTensor()
        self.assertIs(wrapper(original), original)
        entry = wrapper.concrete_graph_entries["one"]
        self.assertIs(wrapper(original), entry.output)
        self.assertEqual(entry.graph.replays, 1)
        # Real inherited input staging returns false before any copy for fresh allocation.
        wrapper._musa_tensors_alias = lambda a, b: a is b
        fresh = FakeTensor()
        self.assertIs(wrapper(fresh), fresh)
        self.assertEqual(fresh.value, 1)
        self.assertEqual(entry.graph.replays, 1)
        self.assertIn(("device_enter", 3), events)
        self.assertFalse(budget.snapshot()["poisoned"])

    def test_capture_error_propagates_and_poison_blocks_further_GPU_work(self):
        wrapper, _, budget, events, _ = build()

        def broken(*args):
            raise ValueError("capture failed")

        wrapper.runnable = broken
        with self.assertRaisesRegex(ValueError, "capture failed"):
            wrapper()
        self.assertTrue(budget.snapshot()["poisoned"])
        before = list(events)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            wrapper()
        self.assertEqual(events, before)

    def test_replay_error_propagates_and_poison_persists_through_clear(self):
        wrapper, _, budget, _, _ = build()
        wrapper()
        wrapper.concrete_graph_entries["one"].graph.error = ValueError("native replay")
        with self.assertRaisesRegex(ValueError, "native replay"):
            wrapper()
        wrapper.clear_graphs()
        self.assertTrue(budget.snapshot()["poisoned"])
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            wrapper()

    def test_reentrant_clear_is_rejected_before_deleting_entries(self):
        wrapper, _, budget, _, _ = build()
        wrapper.runnable = lambda *args: wrapper.clear_graphs()
        with self.assertRaisesRegex(RuntimeError, "Cannot clear"):
            wrapper()
        self.assertIn("one", wrapper.concrete_graph_entries)
        self.assertTrue(budget.snapshot()["poisoned"])

    def test_threaded_reservations_never_exceed_global_cap(self):
        budget = Budget(2)
        keep, outcomes, errors = [], [], []

        def thread():
            try:
                with budget.capture() as admitted:
                    if admitted:
                        self.assertEqual(budget.snapshot()["pending"], 1)
                        graph = FakeGraph()
                        time.sleep(0.005)
                        keep.append(graph)
                        budget.register(graph)
                    outcomes.append(admitted)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(3)
            self.assertFalse(t.is_alive())
        self.assertFalse(errors)
        self.assertEqual(sum(outcomes), 2)
        self.assertEqual(budget.snapshot()["live"], 2)

    def test_uncommitted_capture_poison_and_no_budget_refund(self):
        budget = Budget(1)
        with (
            self.assertRaisesRegex(RuntimeError, "not committed"),
            budget.capture() as admitted,
        ):
            self.assertTrue(admitted)
        self.assertTrue(budget.snapshot()["poisoned"])
        with self.assertRaisesRegex(RuntimeError, "poisoned"), budget.capture():
            self.fail("must not enter")

    def test_finite_transition_logs_and_no_legacy_dependency(self):
        wrapper, ctx, _, events, _ = build(cap=2)
        for key in ("a", "b", "c", "c"):
            ctx.batch_descriptor = key
            wrapper()
        wrapper.clear_graphs()
        logs = [
            event[1][1]
            for event in events
            if isinstance(event, tuple) and event[0] == "log"
        ]
        self.assertEqual(
            logs,
            [
                "first_native_admission",
                "cap_reached",
                "first_admission_rejected",
                "cleared_to_zero",
            ],
        )
        candidate = ast.parse((VENDOR / "graph.py").read_text())
        wrappers = [
            n
            for n in candidate.body
            if isinstance(n, ast.ClassDef) and n.name == "MUSAGraphWrapper"
        ]
        self.assertEqual(len(wrappers), 1)
        self.assertEqual(
            ast.dump(wrappers[0].bases[0]),
            ast.dump(ast.Name(id="GraphWrapper", ctx=ast.Load())),
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "MUSAGraphWrapper"
                    for t in n.targets
                )
                for n in candidate.body
            )
        )
        self.assertFalse(
            any(
                isinstance(n, ast.ImportFrom)
                and n.module is not None
                and ("musa_graph_exec" in n.module)
                for n in ast.walk(candidate)
            )
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "musa_graph_class"
                for n in ast.walk(candidate)
            )
        )

    def test_default_and_nonpositive_admission_caps(self):
        self.assertEqual(Budget().snapshot()["cap"], 1900)
        for cap in (0, -1):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "positive"):
                Budget(cap)

    def test_one_budget_is_shared_across_wrapper_instances(self):
        first, ctx, budget, events, ns = build(cap=2)
        second = ns["MUSAGraphWrapper"](
            first.runnable,
            SimpleNamespace(compilation_config=object()),
            Mode.FULL,
            ns["GraphOptions"](debug_log_enable=False, weak_ref_output=False),
        )
        first()
        second()
        ctx.batch_descriptor = "denied"
        first()
        self.assertEqual(events.count("native_capture"), 2)
        self.assertEqual(budget.snapshot()["live"], 2)
        self.assertTrue(first.concrete_graph_entries["denied"].musa_eager_only)
        self.assertIs(type(first)._all_instances, type(second)._all_instances)

    def test_51_by_121_admission_and_two_runtime_cycles(self):
        wrapper, ctx, budget, events, _ = build(cap=1900)
        sizes = [1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16)]
        original_sizes = tuple(sizes)
        keys = [(size, piece) for size in sizes for piece in range(121)]
        self.assertEqual(len(sizes), 51)
        self.assertEqual(len(keys), 6171)
        for key in keys:
            ctx.batch_descriptor = key
            wrapper()
        entries = wrapper.concrete_graph_entries
        self.assertEqual(len(entries), 6171)
        self.assertEqual(sum(e.graph is not None for e in entries.values()), 1900)
        self.assertEqual(sum(e.musa_eager_only for e in entries.values()), 4271)
        self.assertEqual(events.count("native_capture"), 1900)
        self.assertEqual(budget.snapshot()["live"], 1900)
        for cycle in (1, 2):
            eager_before = events.count("runnable")
            for key in keys:
                ctx.batch_descriptor = key
                wrapper()
            self.assertEqual(events.count("runnable") - eager_before, 4271)
            self.assertEqual(
                sum(e.graph.replays for e in entries.values() if e.graph is not None),
                cycle * 1900,
            )
            self.assertEqual(events.count("native_capture"), 1900)
            self.assertEqual(budget.snapshot()["live"], 1900)
        self.assertEqual(tuple(sizes), original_sizes)
        self.assertFalse(budget.snapshot()["poisoned"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

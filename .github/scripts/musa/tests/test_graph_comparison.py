#!/usr/bin/env python3
"""CPU-only comparison smoke/failure controls; never import torch or access GPUs."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import warnings
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = load("_compare_worker_test", ROOT / "musa_graph_compare_worker.py")
controller = load("_compare_controller_test", ROOT / "run_musa_graph_comparison.py")
PRODUCTION = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-v" else None
if PRODUCTION:
    del sys.argv[1]
CACHE = None
if PRODUCTION:
    CACHE = worker.load_source(
        PRODUCTION, "musa_graph_exec_cache.py", "_comparison_test_production_cache"
    )


class FatalExit(BaseException):
    pass


class FakeGraph:
    def __init__(self, backend):
        self.backend = backend
        self.executable = False
        self.tensor = None
        self.output = None

    def replay(self):
        assert self.executable
        self.output["value"] = self.tensor["value"] * 2 + 3

    def reset(self):
        self.executable = False


class FakeBackend:
    def __init__(self, state):
        self.state = state
        self.policy_device = self
        self.replay_before_check = True
        self.memory_calls = 0

    def current_device(self):
        return 0

    def is_current_stream_capturing(self):
        return False

    def device(self, ordinal):
        assert ordinal == 0
        return nullcontext()

    def synchronize(self):
        self.state["explicit_python_operations"]["policy_sync"] += 1

    def input(self, size, value):
        return {"size": size, "value": value}

    def fill(self, tensor, value):
        tensor["value"] = value

    def graph(self):
        return FakeGraph(self)

    def capture(self, graph, tensor):
        graph.tensor = tensor
        graph.output = {"value": None, "size": tensor["size"]}
        graph.executable = True
        return graph.output  # Deliberately uninitialized until native replay.

    def replay(self, graph, cache=None):
        if cache is not None:
            return cache.replay(graph)
        graph.replay()
        return True

    def eager(self, tensor):
        return {"value": tensor["value"] * 2 + 3, "size": tensor["size"]}

    def reset(self, graph):
        graph.reset()

    def policy_sync(self):
        pass

    def measure_sync(self):
        self.state["measurement_barriers"] += 1

    def memory(self):
        self.memory_calls += 1
        return {"allocated": 0, "reserved": 0}

    def check(self, output, size, expected):
        assert output["size"] == size
        assert output["value"] == expected, "Capture did not execute: replay required"
        self.state["validator_d2h_checks"] += 1


class Ops:
    def __init__(self, state):
        self.state = state

    def has_exec(self, graph):
        return graph.executable

    def free_exec(self, graph):
        self.state["explicit_python_operations"]["free_exec"] += 1
        assert graph.executable
        graph.executable = False

    def inst_exec(self, graph):
        self.state["explicit_python_operations"]["instantiate"] += 1
        graph.executable = True


def state_pass(policy):
    state = worker.initial_state(policy)
    state.update(
        {
            "status": "PASS",
            "source_identity": {
                name: {"sha256": digest}
                for name, digest in worker.EXPECTED_SOURCE_SHA256.items()
            },
            "github_identity": {
                key: os.environ.get(key)
                for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
            },
            "runtime": {
                "declared_container_image": controller.IMAGE,
                "visible_device_count": 4,
                "device_ordinal": 0,
            },
            "capture_requested": 6171,
            "native_capture_completed": 1900 if policy == "admission" else 6171,
            "capture_exact_checks": 6171,
            "postcapture_validation_outcomes": {
                "native_replay": 6171 if policy == "native" else 1900,
                "eager": 0 if policy == "native" else 4271,
            },
            "runtime_cycles": [
                {
                    "cycle": cycle,
                    "exact_checks": 6171,
                    "native_replay": 6171 if policy == "native" else 1900,
                    "eager": 0 if policy == "native" else 4271,
                }
                for cycle in (1, 2)
            ],
        }
    )
    state["explicit_python_operations"]["ops_loader"] = 1 if policy == "current" else 0
    return state


def good_execution():
    return {
        "actual_exit": 0,
        "process_started": True,
        "direct_process_reaped": True,
        "cleanup_confirmed": True,
        "safe_to_continue": True,
        "remaining_owned_children_count": 0,
        "deadline_exceeded": False,
        "deadline_or_signal_exit": False,
    }


class ComparisonCPU(unittest.TestCase):
    def test_four_policy_full_51x121_smoke(self):
        if CACHE is None:
            self.skipTest(
                "Production source path required for source-bound cache smoke"
            )
        results = {}
        for policy in worker.POLICIES:
            state = worker.initial_state(policy)
            backend = FakeBackend(state)

            def loader(state=state):
                state["explicit_python_operations"]["ops_loader"] += 1
                return Ops(state)

            kept = worker.run_variant(
                backend,
                policy,
                state,
                CACHE if policy == "current" else None,
                loader if policy == "current" else None,
            )
            self.assertTrue(kept)
            self.assertEqual(state["status"], "PASS")
            self.assertEqual(state["capture_exact_checks"], 6171)
            self.assertEqual(state["validator_d2h_checks"], 3 * 6171)
            self.assertEqual(
                state["resident_execs"], 6171 if policy == "native" else 1900
            )
            self.assertEqual(
                state["retained_templates"],
                1900 if policy in {"admission", "public_drop"} else 6171,
            )
            self.assertEqual(
                state["public_drop_graphs_released"],
                4271 if policy == "public_drop" else 0,
            )
            self.assertEqual(state["explicit_python_operations"]["instantiate"], 0)
            self.assertEqual(
                state["explicit_python_operations"]["free_exec"],
                4271 if policy == "current" else 0,
            )
            self.assertEqual(
                state["explicit_python_operations"]["public_reset"],
                4271 if policy == "public_drop" else 0,
            )
            results[policy] = state
        (ROOT / "cpu-smoke-results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    def _controller_case(self, mutate_execution=None, mutate_report=None):
        calls = []

        def run(argv, remaining):
            policy = argv[argv.index("--policy") + 1]
            calls.append(policy)
            dest = Path(argv[argv.index("--output-dir") + 1])
            dest.mkdir()
            report = state_pass(policy)
            if mutate_report:
                mutate_report(report)
            controller.sealed_json(dest / "result.json", report)
            result = good_execution()
            if mutate_execution:
                mutate_execution(result)
            return result

        with tempfile.TemporaryDirectory(dir=ROOT) as dirname:
            result = controller.run_comparison(
                Path(dirname), 1500, run, total_seconds=3000
            )
        return calls, result

    def test_parent_four_serial_policies(self):
        calls, result = self._controller_case()
        self.assertEqual(calls, list(worker.POLICIES))
        self.assertEqual(result["status"], "PASS")

    def test_sdk_exit_one_blocks_even_if_reaping_safe(self):
        calls, result = self._controller_case(lambda row: row.update(actual_exit=1))
        self.assertEqual(calls, ["current"])
        self.assertEqual(
            result["blocked_policies"], ["admission", "public_drop", "native"]
        )

    def test_timeout_signal_unknown_cleanup_block(self):
        for change in (
            {"actual_exit": 124, "deadline_exceeded": True},
            {"actual_exit": -15, "deadline_or_signal_exit": True},
            {"cleanup_confirmed": False, "remaining_owned_children_count": None},
            {"actual_exit": False},
            {"remaining_owned_children_count": False},
        ):
            calls, result = self._controller_case(
                lambda row, change=change: row.update(change)
            )
            self.assertEqual(calls, ["current"], change)
            self.assertEqual(result["status"], "FAIL")

    def test_report_mismatch_and_false_success_block(self):
        for change in (
            {"policy": "native"},
            {"capture_exact_checks": 0},
            {"status": "FAIL"},
            {"source_identity": {}},
            {"runtime_cycles": []},
        ):
            calls, result = self._controller_case(
                mutate_report=lambda row, change=change: row.update(change)
            )
            self.assertEqual(calls, ["current"], change)
            self.assertEqual(result["status"], "FAIL")

    def test_missing_report_blocks_next_policy(self):
        calls = []

        def run(argv, remaining):
            calls.append("started")
            return good_execution()

        with tempfile.TemporaryDirectory(dir=ROOT) as dirname:
            result = controller.run_comparison(Path(dirname), 1500, run)
        self.assertEqual(calls, ["started"])
        self.assertEqual(result["status"], "FAIL")

    def test_fatal_context_does_not_restore_GPU_context_after_body_error(self):
        events = []

        class NativeContext:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("GPU_exit")

        def fail(exc):
            events.append("CPU_exit")
            raise FatalExit

        with (
            self.assertRaises(FatalExit),
            worker.FatalNativeContext(NativeContext(), fail),
        ):
            raise RuntimeError("synthetic native error")
        self.assertEqual(events, ["enter", "CPU_exit"])

    def test_failure_seal_IO_error_still_exits(self):
        old = worker.sealed_json
        events = []

        def bad_seal(*args):
            events.append("CPU_seal")
            raise OSError("disk full")

        def exit_process(code):
            events.append(("exit", code))
            raise FatalExit

        try:
            worker.sealed_json = bad_seal
            with self.assertRaises(FatalExit):
                worker.fail_immediately(
                    {}, ROOT, RuntimeError("SDK failure"), exit_process
                )
        finally:
            worker.sealed_json = old
        self.assertEqual(events, ["CPU_seal", ("exit", 1)])

    def test_reset_warning_is_fatal(self):
        events = []

        class WarnGraph:
            def reset(self):
                events.append("reset")
                warnings.warn("native reset failed", RuntimeWarning, stacklevel=2)

        def fail(exc):
            events.append("CPU_exit")
            raise FatalExit

        backend = type("Backend", (), {})()
        backend.fail = fail
        with self.assertRaises(FatalExit):
            worker.NativeBackend.reset(backend, WarnGraph())
        self.assertEqual(events, ["reset", "CPU_exit"])

    def test_source_guard_is_exact(self):
        if PRODUCTION is None:
            self.skipTest("Production source required")
        actual = worker.source_identity(PRODUCTION)
        self.assertEqual(
            {name: row["sha256"] for name, row in actual.items()},
            worker.EXPECTED_SOURCE_SHA256,
        )

    def test_sealed_report_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as dirname:
            path = Path(dirname) / "report.json"
            worker.sealed_json(path, {"status": "PASS"})
            with self.assertRaises(FileExistsError):
                worker.sealed_json(path, {"status": "FAIL"})
            self.assertEqual(json.loads(path.read_text())["status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)

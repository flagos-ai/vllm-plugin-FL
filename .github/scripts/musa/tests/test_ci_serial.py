import contextlib
import importlib.util
import io
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "musa_serial", ROOT / ".github/scripts/musa/run_ci_stage.py"
)
SERIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERIAL)


def group(task="inference"):
    return {
        "task": task,
        "device": "s5000",
        "cases": json.dumps([{"model": "qwen3_6", "case": "27b_tp4_graph"}]),
        "timeout": 60,
    }


def execution(code=0, safe=True):
    return {"actual_exit": code, "safe_to_continue": safe}


class SerialTests(unittest.TestCase):
    def test_rejects_invalid_group_without_any_execution(self):
        for key, value in [
            ("task", "inference;echo bad"),
            ("device", "s5000 --privileged"),
            ("timeout", True),
            ("timeout", 0),
            ("cases", "[]"),
            ("cases", '[{"model":"../../bad","case":"x"}]'),
            ("image", "different-image"),
            ("unknown", "x"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                SERIAL.matrix_groups(
                    json.dumps([dict(group(), **{key: value})]), "image"
                )
        with self.assertRaises(ValueError):
            SERIAL.matrix_groups('[{"task":"inference","task":"serving"}]', "image")

    def test_failed_group_does_not_omit_later_group_or_hide_failure(self):
        calls = []
        exits = iter([0, 1, 0, 0])

        def run(argv, seconds):
            calls.append((argv, seconds))
            return execution(next(exits))

        with tempfile.TemporaryDirectory() as directory:
            rows = SERIAL.run_groups(
                [group(), group("serving")], Path(directory), run, lambda: 1
            )
            self.assertEqual([row["passed"] for row in rows], [False, True])
            self.assertEqual(len(calls), 4)
            self.assertIn("--check-models", calls[0][0])
            self.assertIn("--scope", calls[1][0])
            self.assertIn("e2e", calls[1][0])
            self.assertIn("--task", calls[3][0])
            self.assertIn("serving", calls[3][0])
            self.assertEqual([seconds for _, seconds in calls], [3600] * 4)
            self.assertNotEqual(calls[1][0][-1], calls[3][0][-1])

    def test_missing_model_does_not_start_that_group(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def run(argv, seconds):
                calls.append(argv)
                return execution(1)

            rows = SERIAL.run_groups([group()], Path(directory), run, lambda: 1)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(rows[0]["test_exit"])
            self.assertFalse(rows[0]["passed"])

    def test_model_check_consumes_the_same_group_budget(self):
        clocks = iter([10, 40])
        seconds = []

        def run(argv, remaining):
            seconds.append(remaining)
            return execution()

        with tempfile.TemporaryDirectory() as directory:
            SERIAL.run_groups([group()], Path(directory), run, lambda: next(clocks))
        self.assertEqual(seconds, [3600, 3570])

    def test_one_musa_job_preserves_required_scopes_and_other_backend_paths(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/_musa_test.yml").read_text()
        )
        jobs = workflow["jobs"]
        self.assertNotIn("musa_gate", jobs)
        self.assertEqual(jobs["musa_suite"]["needs"], "setup")
        self.assertEqual(jobs["musa_suite"]["if"], "inputs.platform == 'musa'")
        platform = yaml.safe_load(
            (ROOT / ".github/workflows/_platform_test.yml").read_text()
        )
        generic_jobs = platform["jobs"]
        self.assertNotIn("musa_suite", generic_jobs)
        self.assertEqual(generic_jobs["unit"]["needs"], "setup")
        self.assertEqual(generic_jobs["functional"]["needs"], ["setup", "unit"])
        self.assertEqual(generic_jobs["e2e"]["needs"], ["setup", "functional"])
        self.assertEqual(
            generic_jobs["e2e"]["if"], "needs.setup.outputs.e2e_matrix != '[]'"
        )
        self.assertEqual(generic_jobs["benchmark"]["needs"], ["setup", "e2e"])
        self.assertEqual(
            generic_jobs["notify"]["needs"], ["unit", "functional", "e2e", "benchmark"]
        )
        caller = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        self.assertEqual(
            caller["jobs"]["test-musa"]["uses"], "./.github/workflows/_musa_test.yml"
        )
        for backend in (
            "cuda",
            "ascend",
            "hygon",
            "metax",
            "enflame",
            "kunlunxin",
            "iluvatar",
        ):
            self.assertEqual(
                caller["jobs"]["test-" + backend]["uses"],
                "./.github/workflows/_platform_test.yml",
            )
        steps = jobs["musa_suite"]["steps"]
        runs = [step.get("run", "") for step in steps]
        self.assertEqual(
            [run for run in runs if "run_ci_stage.py --scope " in run],
            [
                "python -I -B .github/scripts/musa/run_ci_stage.py --scope " + scope
                for scope in ("unit", "functional", "e2e", "benchmark")
            ],
        )
        self.assertFalse(any("run-adaptation-gate" in run for run in runs))
        self.assertNotIn("runner.temp", json.dumps(jobs["musa_suite"].get("env", {})))
        self.assertNotIn("MTHREADS_VISIBLE_DEVICES=all", json.dumps(workflow))

    def test_unsafe_exit_blocks_next_group(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            outcomes = iter([execution(), execution(124, False)])

            def run(argv, seconds):
                calls.append(argv)
                return next(outcomes)

            rows = SERIAL.run_groups(
                [group(), group("serving")], Path(directory), run, lambda: 1
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["safe_to_continue"])
            self.assertFalse(rows[0]["passed"])

    def test_wrapper_failure_and_signal_statuses_block_continuation(self):
        cleanup = {
            "adopted_children_reaped": 0,
            "remaining_owned_children_count": 0,
            "cleanup_confirmed": True,
        }
        codes = [
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            134,
            137,
            139,
            143,
            192,
            255,
            -2,
            -9,
            -15,
        ]
        for code in codes:
            with self.subTest(code=code):
                process = Mock()
                process.wait.return_value = code
                with (
                    patch.object(SERIAL.subprocess, "Popen", return_value=process),
                    patch.object(SERIAL, "reap_owned_children", return_value=cleanup),
                ):
                    row = SERIAL.bounded_run([sys.executable, "-c", "pass"], 3)
                self.assertEqual(row["actual_exit"], code)
                self.assertTrue(row["direct_process_reaped"])
                self.assertTrue(row["cleanup_confirmed"])
                self.assertEqual(row["deadline_exceeded"], code == 124)
                self.assertEqual(
                    row["deadline_or_signal_exit"],
                    code == 124 or code < 0 or code >= 128,
                )
                self.assertFalse(row["safe_to_continue"])

    def test_regular_exit_preserves_cleanup_requirement(self):
        for code in [0, 1, 2, 4, 5, 123]:
            for confirmed in [False, True]:
                with self.subTest(code=code, cleanup_confirmed=confirmed):
                    process = Mock()
                    process.wait.return_value = code
                    cleanup = {
                        "adopted_children_reaped": 0,
                        "remaining_owned_children_count": 0 if confirmed else None,
                        "cleanup_confirmed": confirmed,
                    }
                    with (
                        patch.object(SERIAL.subprocess, "Popen", return_value=process),
                        patch.object(
                            SERIAL, "reap_owned_children", return_value=cleanup
                        ),
                    ):
                        row = SERIAL.bounded_run([sys.executable, "-c", "pass"], 3)
                    self.assertEqual(row["actual_exit"], code)
                    self.assertFalse(row["deadline_or_signal_exit"])
                    self.assertEqual(row["safe_to_continue"], confirmed)

    def test_encoded_signal_stops_later_groups_and_gate_output(self):
        cleanup = {
            "adopted_children_reaped": 0,
            "remaining_owned_children_count": 0,
            "cleanup_confirmed": True,
        }
        for code in [130, 143]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                github_output = output / "github-output.txt"
                model_check, tests = Mock(), Mock()
                model_check.wait.return_value = 0
                tests.wait.return_value = code
                with (
                    patch.dict(
                        SERIAL.os.environ,
                        {
                            "GITHUB_OUTPUT": str(github_output),
                            "MUSA_SUITE_ARTIFACT_DIR": str(output),
                            "MUSA_CI_E2E_MATRIX": json.dumps(
                                [group(), group("serving")]
                            ),
                            "MUSA_CI_SUITE_IMAGE": "image",
                        },
                    ),
                    patch.object(
                        SERIAL.sys, "argv", ["run_ci_stage.py", "--scope", "e2e"]
                    ),
                    patch.object(
                        SERIAL.shutil, "which", return_value="/usr/bin/timeout"
                    ),
                    patch.object(SERIAL, "prepare_owned_reaping"),
                    patch.object(SERIAL, "reap_owned_children", return_value=cleanup),
                    patch.object(
                        SERIAL.subprocess,
                        "Popen",
                        side_effect=[model_check, tests, model_check, model_check],
                    ) as popen,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    actual_exit = SERIAL.main()
                report = json.loads((output / "e2e-execution.json").read_text())
                self.assertEqual(actual_exit, 1)
                self.assertEqual(popen.call_count, 2)
                self.assertEqual(report["requested_group_count"], 2)
                self.assertEqual(report["group_count"], 1)
                self.assertEqual(report["groups"][0]["test_exit"], code)
                self.assertTrue(
                    report["groups"][0]["test_execution"]["cleanup_confirmed"]
                )
                self.assertFalse(report["gpu_stages_safe"])
                self.assertEqual(report["status"], "FAIL")
                self.assertFalse(report["passed"])
                self.assertEqual(github_output.read_text(), "gpu_stages_safe=false\n")


@unittest.skipUnless(sys.platform == "linux", "Native Linux owned-child cleanup test")
class NativeCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SERIAL.prepare_owned_reaping()

    def test_success_reaps_direct_child(self):
        row = SERIAL.bounded_run([sys.executable, "-c", "pass"], 3)
        self.assertEqual(row["actual_exit"], 0)
        self.assertTrue(row["direct_process_reaped"])
        self.assertTrue(row["cleanup_confirmed"])
        self.assertEqual(row["remaining_owned_children_count"], 0)
        self.assertTrue(row["safe_to_continue"])

    def test_failure_preserved_after_owned_cleanup(self):
        row = SERIAL.bounded_run([sys.executable, "-c", "raise SystemExit(1)"], 3)
        self.assertEqual(row["actual_exit"], 1)
        self.assertTrue(row["cleanup_confirmed"])
        self.assertTrue(row["safe_to_continue"])

    def test_timeout_reaps_owned_tree_and_blocks_continuation(self):
        child = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)']); time.sleep(5)"
        row = SERIAL.bounded_run([sys.executable, "-c", child], 0.5)
        self.assertIn(row["actual_exit"], [124, 137])
        self.assertTrue(row["direct_process_reaped"])
        self.assertTrue(row["cleanup_confirmed"])
        self.assertEqual(row["remaining_owned_children_count"], 0)
        self.assertFalse(row["safe_to_continue"])

    def test_encoded_signal_exit_blocks_after_native_cleanup(self):
        # Shells and wrappers can return 128 + signal even after reaping a child.
        for code in [130, 143]:
            with self.subTest(code=code):
                row = SERIAL.bounded_run(
                    [sys.executable, "-c", f"raise SystemExit({code})"], 3
                )
                self.assertEqual(row["actual_exit"], code)
                self.assertTrue(row["direct_process_reaped"])
                self.assertTrue(row["cleanup_confirmed"])
                self.assertEqual(row["remaining_owned_children_count"], 0)
                self.assertFalse(row["safe_to_continue"])

    def test_native_signal_exit_blocks_after_native_cleanup(self):
        # GNU timeout can propagate the signal itself or return its encoded status.
        for signum in [signal.SIGINT, signal.SIGTERM]:
            with self.subTest(signal=signum):
                child = (
                    "import os,signal; "
                    f"signal.signal({int(signum)}, signal.SIG_DFL); "
                    f"os.kill(os.getpid(), {int(signum)})"
                )
                row = SERIAL.bounded_run([sys.executable, "-c", child], 3)
                self.assertIn(row["actual_exit"], [-int(signum), 128 + int(signum)])
                self.assertTrue(row["direct_process_reaped"])
                self.assertTrue(row["cleanup_confirmed"])
                self.assertEqual(row["remaining_owned_children_count"], 0)
                self.assertFalse(row["safe_to_continue"])


if __name__ == "__main__":
    unittest.main()

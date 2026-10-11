import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_one_musa_job_preserves_gate_and_other_backend_paths(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/_platform_test.yml").read_text()
        )
        jobs = workflow["jobs"]
        self.assertNotIn("musa_gate", jobs)
        self.assertEqual(jobs["musa_suite"]["needs"], "setup")
        self.assertEqual(jobs["musa_suite"]["if"], "inputs.platform == 'musa'")
        for key in ["unit", "functional", "benchmark"]:
            self.assertEqual(jobs[key]["if"], "inputs.platform != 'musa'")
        self.assertIn("inputs.platform != 'musa'", jobs["e2e"]["if"])
        steps = jobs["musa_suite"]["steps"]
        runs = [step.get("run", "") for step in steps]
        self.assertEqual(
            sum(
                "bash .github/scripts/musa/run-adaptation-gate.sh" in run
                for run in runs
            ),
            1,
        )
        gate = next(
            step for step in steps if "original Gate models" in step.get("name", "")
        )
        self.assertIn("always()", gate["if"])
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


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 FlagOS Contributors
"""CPU-only tests of actual Sunrise CI acceptance boundaries."""

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import resolve_config
from run_gate import (
    CHECKS,
    REQUEST_IDS,
    SCENARIOS,
    kv_cache_capacity,
    replay_evidence,
    server_command,
    validate_results,
)
from run_tests import UNIT_PATHS, junit_counts


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "runner_labels": [],
            "ci_image": "example.invalid/sunrise:v0.28.0-test-fixture",
            "visible_devices": "6,7",
            "container_volumes": ["/data:/data:ro"],
            "container_options": "--ipc=host",
            "model_27b": "/data/models/27b",
            "model_35b": "/data/models/35b",
            "flagcx_path": "/workspace/FlagCX",
        }
        self.env = {"SUNRISE_CI_RUNNER_LABELS": '["explicit-test-label"]'}

    def test_requires_configured_labels_without_using_runner_name(self):
        with self.assertRaisesRegex(ValueError, "real dedicated Sunrise runner"):
            resolve_config(self.config, {})

    def test_explicit_resources_are_preserved(self):
        config = resolve_config(self.config, self.env)
        self.assertEqual(config["runner_labels"], ["explicit-test-label"])
        self.assertEqual(config["model_27b"], "/data/models/27b")

    def test_rejects_old_image_without_actual_upgrade_input(self):
        self.config["ci_image"] = "example.invalid/vllm0.24.0:fixture"
        with self.assertRaisesRegex(ValueError, "real vLLM 0.28.0"):
            resolve_config(self.config, self.env)
        self.env["SUNRISE_CI_VLLM_WHEEL"] = "/data/wheels/vllm-0.28.0+empty.whl"
        self.assertTrue(resolve_config(self.config, self.env)["vllm_wheel"])

    def test_rejects_duplicate_devices_and_bad_label_types(self):
        for env in (
            {**self.env, "SUNRISE_CI_VISIBLE_DEVICES": "6,6"},
            {**self.env, "SUNRISE_CI_RUNNER_LABELS": '"Sunrise8"'},
            {**self.env, "SUNRISE_CI_MODEL_27B": "relative/path"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                resolve_config(self.config, env)


class DevicePreflightTests(unittest.TestCase):
    def test_host_initialized_count_guard_precedes_python_and_kernel_imports(self):
        source = (HERE / "check.sh").read_text()
        self.assertLess(source.index("/sys/class/pt/count"), source.index("<<'PY'"))
        self.assertIn("10#$initialized_count < 2", source)
        self.assertIn("Sunrise PTPU preflight failed", source)

    def test_runtime_availability_and_visible_count_guards_precede_kernel_imports(self):
        source = (
            (HERE / "check.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        )
        body = ast.parse(source).body
        imports = {
            alias.name: index
            for index, statement in enumerate(body)
            if isinstance(statement, ast.Import)
            for alias in statement.names
        }
        guards = [
            index
            for index, statement in enumerate(body)
            if isinstance(statement, ast.If)
            and any(isinstance(node, ast.Raise) for node in ast.walk(statement))
            and (
                "torch.ptpu.is_available" in ast.unparse(statement.test)
                or "visible_count" in ast.unparse(statement.test)
            )
        ]
        self.assertEqual(len(guards), 2)
        for name in ("flag_gems", "flaggems_vllm"):
            self.assertLess(max(guards), imports[name])
        self.assertLess(imports["torch_ptpu"], min(guards))


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def documents(self):
        for name, count in SCENARIOS.items():
            inputs = [{"request_id": value} for value in REQUEST_IDS[name]]
            row = {
                "case": {
                    "scenario": name,
                    "model_path": "/model",
                    "served_model_name": "owned-model",
                    "port": "8000",
                },
                "input": inputs,
                "output": [
                    {
                        "request_id": value,
                        "text": "response",
                        "checks": dict.fromkeys(CHECKS, True),
                        "passed": True,
                    }
                    for value in REQUEST_IDS[name]
                ],
                "summary": {
                    "request_count": count,
                    "passed_count": count,
                    "failed_count": 0,
                    "passed": True,
                },
            }
            (self.root / (name + ".json")).write_text(json.dumps(row))

    def validate(self):
        return validate_results(self.root, "/model", "owned-model", 8000)

    def test_requires_all_five_scenarios_and_26_requests(self):
        self.documents()
        self.assertEqual(
            self.validate(), {"scenarios": 5, "requests": 26, "passed": True}
        )
        (self.root / "image_single.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing Gate scenarios"):
            self.validate()

    def test_rejects_forged_pass_with_failed_or_missing_request_checks(self):
        self.documents()
        path = self.root / "text_single.json"
        original = json.loads(path.read_text())
        for checks in (
            {},
            {"invented": True},
            {**dict.fromkeys(CHECKS, True), "expected_semantics": False},
        ):
            row = copy.deepcopy(original)
            row["output"][0]["checks"] = checks
            path.write_text(json.dumps(row))
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                self.validate()

    def test_rejects_receipts_from_another_service(self):
        self.documents()
        path = self.root / "image_single.json"
        row = json.loads(path.read_text())
        row["case"]["served_model_name"] = "unrelated-service"
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "another model/server"):
            self.validate()

    def test_junit_rejects_empty_or_skipped_suites(self):
        path = self.root / "junit.xml"
        for counts in ('tests="0"', 'tests="1" skipped="1"', 'tests="1" failures="1"'):
            path.write_text("<testsuites><testsuite " + counts + "/></testsuites>")
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                junit_counts(path)
        path.write_text('<testsuites><testsuite tests="2"/></testsuites>')
        self.assertEqual(junit_counts(path)["tests"], 2)

    def test_requires_actual_kv_capacity_for_requested_context(self):
        for text in ("", "GPU KV cache size: 4,096 tokens"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                kv_cache_capacity(text, 8192)
        self.assertEqual(
            kv_cache_capacity("GPU KV cache size: 16,384 tokens", 8192)[
                "token_capacities"
            ],
            [16384],
        )

    def test_graph_requires_actual_non_none_replay_during_requests(self):
        for text in (
            "Graph capturing finished",
            "Sunrise graph replay completed runtime_mode=NONE",
            "Sunrise graph replay completed runtime_mode=PIECEWISE",
            "Sunrise graph replay completed runtime_mode=FULL_DECODE_ONLY",
            "Sunrise graph replay completed runtime_mode=CUDAGraphMode.NONE",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                replay_evidence(text)
        evidence = replay_evidence(
            "Sunrise graph replay completed runtime_mode=FULL batch_descriptor=test"
        )
        self.assertEqual(evidence["events"], 1)

    def test_duplicate_request_ids_cannot_substitute_for_real_cases(self):
        self.documents()
        path = self.root / "text_concurrent_8.json"
        row = json.loads(path.read_text())
        row["input"] = [copy.deepcopy(row["input"][0]) for _ in range(8)]
        row["output"] = [copy.deepcopy(row["output"][0]) for _ in range(8)]
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.validate()

    def test_finite_suite_includes_real_graph_capture_regressions(self):
        self.assertIn(
            "tests/unit_tests/dispatch/test_sunrise_graph_capture.py", UNIT_PATHS
        )
        repo = HERE.parents[2]
        self.assertTrue(all((repo / name).is_file() for name in UNIT_PATHS))

    def test_graph_and_eager_commands_have_distinct_real_execution_modes(self):
        eager = server_command("/model", "served", 8000, "eager", self.root, 4096, 2)
        graph = server_command("/model", "served", 8001, "graph", self.root, 4096, 2)
        self.assertIn("--enforce-eager", eager)
        self.assertNotIn("--enforce-eager", graph)
        self.assertEqual(
            json.loads(graph[graph.index("--compilation-config") + 1]),
            {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
        )
        self.assertIn("--kv-cache-memory-bytes", eager)


if __name__ == "__main__":
    unittest.main(verbosity=2)

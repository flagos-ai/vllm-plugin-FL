# Copyright 2026 FlagOS Contributors
"""CPU-only checks for NVIDIA CI selection and failure/reporting contracts."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.run import TestRunner
from tests.utils.model_config import ModelConfig
from tests.utils.nvidia_validation import (
    MANUAL_CASES,
    InferenceEvidence,
    checkpoint_facts,
    inference_environment,
    load_manual_case,
    validate_resources,
    verify_execution,
)
from tests.utils.platform_config import PlatformConfig

spec = importlib.util.spec_from_file_location(
    "nvidia_manual_runner", Path(__file__).with_name("run_regression.py")
)
manual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manual)


class NvidiaRegressionContracts(unittest.TestCase):
    def test_existing_four_h20_cases_preserved(self):
        cases = TestRunner("cuda", "h20", scope="e2e").discover_tests()
        self.assertEqual(
            {(c.task, c.model, c.case) for c in cases},
            {
                ("inference", "qwen3_6", "27b_tp2_eager"),
                ("inference", "qwen3_6", "35b_a3b_tp2_graph"),
                ("serving", "qwen3_6", "27b_tp2_graph"),
                ("serving", "qwen3_6", "35b_a3b_tp2_eager"),
            },
        )
        selected = [c for c in cases if c.task == "inference"]
        self.assertEqual(
            [c.extra_env["VLLM_USE_V2_MODEL_RUNNER"] for c in selected], ["0", "1"]
        )
        self.assertTrue(all("--json-report" in c.extra_args for c in selected))

    def test_other_platforms_and_cases_do_not_opt_in(self):
        self.assertEqual(
            inference_environment(
                PlatformConfig.load("ascend"), "qwen3_6", "27b_tp2_eager", "."
            ),
            {},
        )
        self.assertEqual(
            inference_environment(
                PlatformConfig.load("cuda", "a100"), "qwen3", "4b_tp2", "."
            ),
            {},
        )

    def test_manual_cases_not_in_automatic_device_matrices(self):
        config = PlatformConfig.load("cuda")
        for device in config.device_types:
            selected = TestRunner("cuda", device, scope="e2e").discover_tests()
            self.assertFalse(MANUAL_CASES & {(c.model, c.case) for c in selected})
        for model, case in MANUAL_CASES:
            cfg, checks = load_manual_case(model, case, "/reserved/checkpoint")
            self.assertIn(checks["runner"], ("v1", "v2"))
            self.assertEqual(cfg.generate.sampling["temperature"], 0)
        with self.assertRaises(ValueError):
            load_manual_case("hy4", "tp8_pp2", "/model")

    def test_selected_missing_model_is_failure_with_report(self):
        with (
            tempfile.TemporaryDirectory() as d,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = manual.main(
                [
                    "--model",
                    "qwen3",
                    "--case",
                    "4b_v1_graph",
                    "--model-path",
                    d,
                    "--output-dir",
                    d,
                ]
            )
            self.assertEqual(code, 1)
            report = json.loads((Path(d) / "test-results-cuda.json").read_text())
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertIn("missing config.json", report["results"][0]["message"])

    def test_missing_case_is_not_an_empty_success(self):
        with (
            tempfile.TemporaryDirectory() as d,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(manual.main(["--output-dir", d]), 1)

    def test_checkpoint_requires_all_indexed_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "missing.safetensors"}})
            )
            with self.assertRaises(FileNotFoundError):
                checkpoint_facts(d)
            (path / "missing.safetensors").write_bytes(b"test")
            self.assertEqual(checkpoint_facts(d)["weight_files"], 1)

    def test_wrong_checkpoint_architecture_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text('{"architectures": ["WrongModel"]}')
            with self.assertRaisesRegex(ValueError, "Wrong checkpoint architecture"):
                checkpoint_facts(d, ["Qwen4ExpForConditionalGeneration"])

    def test_passed_pytest_without_actual_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            report, evidence = Path(d) / "pytest.json", Path(d) / "inference.json"
            report.write_text('{"summary": {"passed": 1, "collected": 1}}')
            evidence.write_text('{"status": "not_run"}')
            with self.assertRaisesRegex(RuntimeError, "worker/generation"):
                verify_execution(report, evidence)

    def test_no_generation_cannot_finish_as_passed(self):
        with tempfile.TemporaryDirectory() as d:
            evidence = InferenceEvidence(
                ModelConfig("/model"), "v2", Path(d) / "facts.json"
            )
            with self.assertRaisesRegex(RuntimeError, "did not execute"):
                evidence.finish()
            self.assertEqual(evidence.data["status"], "failed")

    def test_wrong_actual_model_class_is_rejected(self):
        from tests.utils.nvidia_validation import RUNNERS

        facts = [
            {
                "rank": 0,
                "use_v2_model_runner": True,
                "runner_class": RUNNERS["v2"],
                "model_class": "WrongModel",
                "native_vllm_kernels": False,
            }
        ]
        with (
            tempfile.TemporaryDirectory() as d,
            patch("importlib.metadata.version", return_value="test"),
        ):
            evidence = InferenceEvidence(
                ModelConfig("/model"), "v2", Path(d) / "facts.json"
            )
            with self.assertRaisesRegex(AssertionError, "Wrong actual model"):
                evidence.workers(
                    SimpleNamespace(collective_rpc=lambda *a, **kw: facts),
                    "ExpectedModel",
                )

    def test_insufficient_or_busy_gpus_fail(self):
        cfg = ModelConfig(
            "/model", engine={"tensor_parallel_size": 8, "gpu_memory_utilization": 0.9}
        )
        gpu = {"total_bytes": 80 * 1024**3, "free_bytes": 79 * 1024**3}
        with self.assertRaisesRegex(RuntimeError, "requires 8"):
            validate_resources([gpu] * 7, cfg, 70)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            validate_resources([gpu] * 8, cfg, 90)
        with self.assertRaisesRegex(RuntimeError, "reserve an idle"):
            validate_resources([{**gpu, "free_bytes": 1024**3}] * 8, cfg, 70)
        validate_resources([gpu] * 8, cfg, 70)

    def test_skip_or_no_execution_cannot_pass_manual_runner(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pytest.json"
            for summary in (
                {"passed": 0, "collected": 0},
                {"passed": 0, "collected": 1, "skipped": 1},
            ):
                path.write_text(json.dumps({"summary": summary}))
                with self.assertRaises(RuntimeError):
                    verify_execution(path)
            path.write_text(json.dumps({"summary": {"passed": 1, "collected": 1}}))
            verify_execution(path)

    def test_wrong_actual_runner_and_empty_rpc_are_rejected(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("importlib.metadata.version", return_value="test"),
        ):
            evidence = InferenceEvidence(
                ModelConfig("/model"), "v1", Path(d) / "facts.json"
            )
            for facts in (
                [],
                [
                    {
                        "rank": 0,
                        "use_v2_model_runner": True,
                        "runner_class": "wrong",
                        "native_vllm_kernels": False,
                    }
                ],
            ):
                with self.assertRaises(AssertionError):
                    evidence.workers(
                        SimpleNamespace(
                            collective_rpc=lambda *args, facts=facts, **kwargs: facts
                        )
                    )

    def test_empty_output_wrong_order_and_repeat_mismatch_fail(self):
        def output(text, ids):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=ids)])
            ]

        with tempfile.TemporaryDirectory() as d:
            evidence = InferenceEvidence(
                ModelConfig("/model"), "v2", Path(d) / "facts.json"
            )
            for first, repeat, expected in [
                (output("", [1]), output("", [1]), [[]]),
                (output("blue red", [1]), output("blue red", [1]), [["red", "blue"]]),
                (output("4", [1]), output("4", [2]), [["4"]]),
            ]:
                with self.assertRaises(AssertionError):
                    evidence.outputs("test", first, repeat, expected)
            with self.assertRaisesRegex(AssertionError, "Expected exact"):
                evidence.outputs(
                    "test", output("14", [1]), output("14", [1]), [["4"]], match="exact"
                )
            evidence.outputs(
                "test", output("4", [1]), output("4", [1]), [["4"]], match="exact"
            )
            evidence.outputs(
                "test",
                output("red blue", [1]),
                output("red blue", [1]),
                [["red", "blue"]],
            )


if __name__ == "__main__":
    unittest.main()

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "musa_gate_entry_diagnostics", TOOLS / "gate_entry.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


class EntryDiagnosticsTests(unittest.TestCase):
    def exercise(
        self,
        failure,
        *,
        huge_stderr=False,
        unavailable_report=False,
        stderr_override=None,
        report_error_override=None,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "repo"
            repo.mkdir()
            inputs = root / "results" / ("a" * 32 + "-inputs")
            calls = []

            def child(argv, env, cwd, directory, seconds, deadline):
                phase = "preflight" if len(calls) == 0 else "observer-config"
                calls.append((phase, seconds))
                directory.mkdir()
                stderr = (
                    "Authorization: Bearer DO_NOT_PRINT_THIS\n"
                    "SAFE_ENV=DO_NOT_PRINT_THIS\n"
                    "ValueError: child " + phase + " source contract failed\n"
                )
                if stderr_override is not None:
                    stderr = stderr_override
                if huge_stderr:
                    stderr = "DO_NOT_PRINT_THIS" * 8192
                (directory / "stderr.raw").write_text(stderr)
                passed = failure != phase
                report = {
                    "passed": passed,
                    "errors": []
                    if passed
                    else [
                        report_error_override
                        if report_error_override is not None
                        else phase + " precise source mismatch"
                    ],
                    "traceback": None
                    if passed
                    else "ValueError: rejected " + phase + " RECORD",
                    "observer_source_sha256": "b" * 64,
                }
                output = inputs / (
                    "preflight.json"
                    if phase == "preflight"
                    else "observer-config-report.json"
                )
                if not unavailable_report:
                    output.write_text(json.dumps(report))
                return {
                    "argv": argv,
                    "actual_exit": 0 if passed else 1,
                    "errors": [],
                    "cleanup": {
                        "closed": True,
                        "errors": [],
                        "remaining_observation_complete": True,
                        "remaining_processes": [],
                    },
                }

            env = {
                "GITHUB_WORKSPACE": str(repo),
                "MUSA_CI_RESULT_ROOT": str(root / "results"),
                "MUSA_CI_CACHE_ROOT": str(root / "cache"),
                "MUSA_CI_EXPECTED_COMMIT": "e" * 40,
                "MUSA_CI_RUN_ID": "a" * 32,
                "SECRET_ENV": "DO_NOT_PRINT_THIS",
            }
            captured = io.StringIO()
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(ENTRY.sys, "argv", ["gate_entry.py"]),
                patch.object(ENTRY.signal, "signal"),
                patch.object(ENTRY, "run_command", side_effect=child),
                patch.object(
                    ENTRY.os, "execv", side_effect=OSError("exec launch denied")
                ) as launch,
                contextlib.redirect_stderr(captured),
            ):
                actual_exit = ENTRY.main()
            initial = json.loads((inputs / "entry-result.json").read_text())
            launch_failure = inputs / "entry-launch-failure.json"
            return (
                actual_exit,
                captured.getvalue(),
                calls,
                initial,
                json.loads(launch_failure.read_text())
                if launch_failure.exists()
                else None,
                launch.call_count,
            )

    def test_actual_preflight_failure_retains_error_and_never_launches(self):
        code, output, calls, initial, later, launches = self.exercise("preflight")
        self.assertEqual(code, 1)
        self.assertEqual(calls, [("preflight", 300)])
        self.assertFalse(initial["passed"])
        self.assertEqual(len(initial["steps"]), 1)
        self.assertIn("Actual source/cache/device origin preflight failed", output)
        self.assertIn("preflight precise source mismatch", output)
        self.assertIn("child preflight source contract failed", output)
        self.assertNotIn("DO_NOT_PRINT_THIS", output)
        self.assertIsNone(later)
        self.assertEqual(launches, 0)

    def test_actual_config_failure_retains_error_and_never_launches(self):
        code, output, calls, initial, later, launches = self.exercise("observer-config")
        self.assertEqual(code, 1)
        self.assertEqual(calls, [("preflight", 300), ("observer-config", 120)])
        self.assertFalse(initial["passed"])
        self.assertEqual(len(initial["steps"]), 2)
        self.assertIn(
            "Actual readonly observer source/request configuration failed", output
        )
        self.assertIn("observer-config precise source mismatch", output)
        self.assertIn("child observer-config source contract failed", output)
        self.assertNotIn("DO_NOT_PRINT_THIS", output)
        self.assertIsNone(later)
        self.assertEqual(launches, 0)

    def test_exec_error_preserves_original_receipts_and_reports_launch_failure(self):
        code, output, calls, initial, later, launches = self.exercise("launch")
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(initial["passed"])
        self.assertEqual(initial["errors"], [])
        self.assertTrue(
            later["passed"]
        )  # Original semantics: launch error has a separate receipt.
        self.assertIn("OSError: exec launch denied", later["errors"][0])
        self.assertIn("OSError: exec launch denied", output)
        self.assertNotIn("SECRET_ENV", output)
        self.assertNotIn("DO_NOT_PRINT_THIS", output)
        self.assertEqual(launches, 1)

    def test_oversized_child_log_is_omitted_and_output_stays_bounded(self):
        code, output, calls, initial, later, launches = self.exercise(
            "preflight", huge_stderr=True
        )
        self.assertEqual(code, 1)
        self.assertIn("Small child stderr unavailable: ValueError", output)
        self.assertIn("preflight precise source mismatch", output)
        self.assertNotIn("DO_NOT_PRINT_THIS", output)
        self.assertLess(len(output), 16000)
        self.assertEqual(launches, 0)

    def test_missing_failure_report_does_not_hide_entry_error_or_launch(self):
        code, output, calls, initial, later, launches = self.exercise(
            "preflight", unavailable_report=True
        )
        self.assertEqual(code, 1)
        self.assertIn("Small failure report unavailable: FileNotFoundError", output)
        self.assertIn("Actual source/cache/device origin preflight failed", output)
        self.assertIn("child preflight source contract failed", output)
        self.assertEqual(launches, 0)

    def test_long_credential_and_environment_lines_are_filtered_before_tail(self):
        controls = (
            ("Authorization: Bearer ", "AUTH_SENTINEL"),
            ("HF_TOKEN=", "HF_SENTINEL"),
            ("SOME_ENV=", "ENV_SENTINEL"),
            ('{"API_KEY": "', "JSON_KEY_SENTINEL"),
        )
        for key, sentinel in controls:
            payload = key + "s" * 6000 + sentinel
            self.assertLess(len(payload.encode()), ENTRY._DIAGNOSTIC_FILE_CAP)
            code, output, calls, initial, later, launches = self.exercise(
                "preflight", stderr_override=payload, report_error_override=payload
            )
            self.assertEqual(code, 1, key)
            self.assertNotIn(sentinel, output, key)
            self.assertNotIn("s" * 100, output, key)
            self.assertIn("[credential/environment line omitted]", output, key)
            self.assertIn("Actual source/cache/device origin preflight failed", output)
            self.assertEqual(launches, 0)

    def test_nonsecret_multiline_failure_tail_remains_visible_and_bounded(self):
        payload = "\n".join(f"ordinary traceback frame {n}" for n in range(500))
        payload += "\nValueError: useful final source mismatch"
        self.assertLess(len(payload.encode()), ENTRY._DIAGNOSTIC_FILE_CAP)
        code, output, calls, initial, later, launches = self.exercise(
            "preflight", stderr_override=payload
        )
        self.assertEqual(code, 1)
        self.assertIn("ValueError: useful final source mismatch", output)
        self.assertNotIn("ordinary traceback frame 0\n", output)
        self.assertLess(len(output), 16000)
        self.assertEqual(launches, 0)

    def test_opaque_oversized_line_is_omitted_without_losing_short_error_tail(self):
        payload = "z" * 6000 + "OPAQUE_SENTINEL\nValueError: short final mismatch"
        code, output, calls, initial, later, launches = self.exercise(
            "preflight", stderr_override=payload
        )
        self.assertEqual(code, 1)
        self.assertNotIn("OPAQUE_SENTINEL", output)
        self.assertIn("[oversized diagnostic line omitted]", output)
        self.assertIn("ValueError: short final mismatch", output)
        self.assertEqual(launches, 0)


if __name__ == "__main__":
    unittest.main()

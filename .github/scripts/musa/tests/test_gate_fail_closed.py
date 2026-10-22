import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "musa_gate_fail_closed", TOOLS / "run_gate.py"
)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def complete_cleanup(actual_exit=0, known=True):
    return {
        "closed": True,
        "errors": [],
        "actual_exit": actual_exit,
        "actual_exit_known": known,
        "remaining_observation_complete": True,
        "remaining_processes": [],
    }


class GateFailureTests(unittest.TestCase):
    def assert_one_failed_route_blocks_later_routes(
        self, server_cleanup, *, client_failure=False, server_identity=None
    ):
        writes = {}

        def write_json(path, value):
            writes[Path(path).name] = value
            return {"path": str(path), "bytes": 1, "sha256": "a" * 64}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            binding = root / "binding.json"
            binding.write_text("{}")
            argv = [
                "run_gate.py",
                "--repo",
                str(root),
                "--expected-commit",
                "a" * 40,
                "--gate-dir",
                str(root),
                "--source-binding",
                str(binding),
                "--output-root",
                str(root / "results"),
                "--cache-root",
                str(root),
            ]

            def one_route(args, route, output, binding_id):
                return route | {
                    "source_binding_id": binding_id,
                    "server": {
                        "attempted": True,
                        "started": True,
                        "identity": server_identity,
                    },
                    "preflight_command": {"cleanup": complete_cleanup()},
                    "pytest": [
                        {
                            "file": name,
                            "attempted": True,
                            "actual_exit": 1 if client_failure and index == 0 else 0,
                            "timed_out": False,
                            "errors": [],
                            "cleanup": complete_cleanup(),
                        }
                        for index, name in enumerate(GATE.TEST_FILES)
                    ],
                    "cleanup": copy.deepcopy(server_cleanup),
                    "errors": [],
                }

            with (
                patch.object(GATE.sys, "argv", argv),
                patch.object(GATE.sys, "platform", "linux"),
                patch.object(GATE.os, "pidfd_open", create=True),
                patch.object(GATE.os, "WNOWAIT", create=True),
                patch.object(GATE.signal, "pidfd_send_signal", create=True),
                patch.object(GATE.signal, "signal"),
                patch.object(GATE, "INTERRUPTED", False),
                patch.object(GATE, "one_route", side_effect=one_route) as run,
                patch.object(
                    GATE,
                    "route_business",
                    side_effect=ValueError("Original pytest exit1")
                    if client_failure
                    else None,
                    return_value={},
                ) as business,
                patch.object(GATE, "write_json", side_effect=write_json),
                patch.object(GATE, "aggregate", return_value={"gate_passed": False}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                actual_exit = GATE.main()

        self.assertEqual(actual_exit, 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(business.call_count, int(client_failure))
        manifest = writes["route-manifest.json"]
        self.assertEqual(len(manifest["routes"]), 4)
        self.assertTrue(manifest["routes"][0]["cleanup"]["closed"])
        self.assertEqual(manifest["routes"][0]["cleanup"], server_cleanup)
        self.assertTrue(manifest["errors"])
        self.assertTrue(
            all(
                row["errors"][0].startswith("NOT_RUN:")
                for row in manifest["routes"][1:]
            )
        )
        return manifest

    def test_failed_client_with_confirmed_cleanup_never_starts_later_route(self):
        manifest = self.assert_one_failed_route_blocks_later_routes(
            complete_cleanup(), client_failure=True
        )
        self.assertEqual(manifest["routes"][0]["pytest"][0]["actual_exit"], 1)

    def test_nonzero_unknown_or_noninteger_server_exit_blocks_later_routes(self):
        for actual, known in (
            (1, True),
            (-15, True),
            (None, False),
            (False, True),
            (0, False),
        ):
            with self.subTest(actual_exit=actual, known=known):
                manifest = self.assert_one_failed_route_blocks_later_routes(
                    complete_cleanup(actual, known)
                )
                self.assertTrue(
                    all(
                        item["actual_exit"] == 0
                        for item in manifest["routes"][0]["pytest"]
                    )
                )
                self.assertIn(
                    "Owned server did not exit normally", manifest["errors"][0]
                )

    @unittest.skipUnless(
        GATE.sys.platform == "linux"
        and hasattr(GATE.os, "pidfd_open")
        and hasattr(GATE.os, "WNOWAIT")
        and hasattr(GATE.signal, "pidfd_send_signal"),
        "Actual pidfd owned-process control requires Linux; not a GPU test",
    )
    def test_actual_reaped_server_exit1_blocks_after_successful_client_reports(self):
        # This actual owned CPU child exits1 normally. No accelerator module is imported.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            execution = GATE.run_command(
                [GATE.sys.executable, "-I", "-B", "-c", "raise SystemExit(1)"],
                GATE.os.environ.copy(),
                str(root),
                root / "server-exit1",
                10,
                GATE.time.monotonic() + 15,
            )
            self.assertEqual(execution["actual_exit"], 1)
            self.assertFalse(execution["timed_out"])
            self.assertTrue(GATE.cleanup_complete(execution["cleanup"]))
            self.assertTrue(execution["cleanup"]["actual_exit_known"])
            self.assertEqual(execution["cleanup"]["remaining_processes"], [])
            manifest = self.assert_one_failed_route_blocks_later_routes(
                execution["cleanup"], server_identity=execution["identity"]
            )
            self.assertTrue(
                all(
                    item["actual_exit"] == 0 for item in manifest["routes"][0]["pytest"]
                )
            )


if __name__ == "__main__":
    unittest.main()

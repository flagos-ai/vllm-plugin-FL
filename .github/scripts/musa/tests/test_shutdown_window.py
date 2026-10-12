"""CPU fixtures of real server argv and shutdown windows; no vendor/GPU execution."""

import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_gate
from gate_common import ROUTES


class ShutdownWindow(unittest.TestCase):
    def test_actual_four_route_argv_and_server_grace(self):
        with tempfile.TemporaryDirectory(prefix="musa-shutdown-argv-") as directory:
            root = Path(directory).resolve()
            for name in ("model", "cache", "kit", "runs"):
                (root / name).mkdir()
            args = SimpleNamespace(
                python=str(root / "venv/bin/python"),
                repo=str(root),
                gate_dir=str(root / "kit"),
                cache_root=str(root / "cache"),
                expected_commit="a" * 40,
                source_binding=str(root / "binding.json"),
                observer_config=None,
            )
            constructed = []

            def preflight(argv, env, cwd, directory, seconds, deadline):
                output = Path(argv[argv.index("--output") + 1])
                output.write_text(json.dumps({"passed": True}), encoding="utf-8")
                return {"actual_exit": 0, "errors": []}

            class HaltBeforeProcess:
                def __init__(self, argv, env, cwd, logs, term_seconds):
                    constructed.append((argv, term_seconds))
                    self.initial = None
                    self.start_error = "Fixture halts before starting a real process"

                def close(self):
                    return {
                        "closed": True,
                        "errors": [],
                        "remaining_processes": [],
                        "remaining_observation_complete": True,
                        "actual_exit": None,
                    }

            with (
                patch.object(run_gate, "run_command", preflight),
                patch.object(run_gate, "OwnedProcess", HaltBeforeProcess),
                patch.object(run_gate.socket, "socket"),
            ):
                for route_id, model, mode, port in ROUTES:
                    route = {
                        "route_id": route_id,
                        "model": model,
                        "mode": mode,
                        "model_path": str(root / "model"),
                        "served_model_name": model,
                        "port": port,
                    }
                    result = run_gate.one_route(args, route, root / "runs", "b" * 64)
                    argv, grace = constructed[-1]
                    with self.subTest(route=route_id):
                        self.assertEqual(argv.count("--shutdown-timeout"), 1)
                        self.assertEqual(
                            argv[argv.index("--shutdown-timeout") + 1], "60"
                        )
                        self.assertEqual(grace, 90)
                        self.assertEqual("--enforce-eager" in argv, mode == "eager")
                        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "8")
                        self.assertEqual(argv.count("--gpu-memory-utilization"), 1)
                        self.assertEqual(
                            argv[argv.index("--gpu-memory-utilization") + 1],
                            "0.92" if mode == "graph" else "0.6",
                        )
                        budget = int(argv[argv.index("--max-num-batched-tokens") + 1])
                        self.assertEqual(budget, 4096)
                        if mode == "graph":
                            self.assertEqual(argv.count("--compilation-config"), 1)
                            config = json.loads(
                                argv[argv.index("--compilation-config") + 1]
                            )
                            self.assertEqual(set(config), {"cudagraph_capture_sizes"})
                            sizes = config["cudagraph_capture_sizes"]
                            self.assertEqual(sizes, [1, 2, 4, 8, 16, 256, 1024, 4096])
                            self.assertEqual(sizes, sorted(set(sizes)))
                            self.assertTrue(
                                all(type(size) is int and size > 0 for size in sizes)
                            )
                            self.assertEqual(sizes[0], 1)
                            self.assertEqual(sizes[-1], budget)
                            # Every admitted batch has a bounded capture key,
                            # including the observed mixed-prefill sizes.
                            for tokens in range(1, budget + 1):
                                padded = next(size for size in sizes if size >= tokens)
                                self.assertLessEqual(padded, budget)
                            for tokens, padded in (
                                (194, 256),
                                (527, 1024),
                                (722, 1024),
                                (967, 1024),
                            ):
                                self.assertEqual(
                                    next(size for size in sizes if size >= tokens),
                                    padded,
                                )
                        else:
                            self.assertNotIn("--compilation-config", argv)
                        self.assertEqual(result["server"]["argv"], argv)
                        self.assertTrue(result["server"]["attempted"])
                        self.assertFalse(result["server"]["started"])
            self.assertEqual(len(constructed), 4)
            self.assertEqual(run_gate.TERM_SECONDS, 15)
            self.assertEqual(run_gate.ROUTE_SECONDS, 6900)

    def test_api_gets_completion_window_before_forced_kill(self):
        with tempfile.TemporaryDirectory(prefix="musa-shutdown-close-") as directory:
            root = Path(directory).resolve()
            for grace, expected_forced in (
                (run_gate.API_TERM_SECONDS, False),
                (run_gate.TERM_SECONDS, True),
            ):
                with self.subTest(grace=grace):
                    clock = [0.0]
                    alive = [True]
                    calls = []
                    # Linux-only runner; supply its signal constant in Windows fixtures.
                    kill_signal = getattr(signal, "SIGKILL", 9)
                    actor = run_gate.OwnedProcess.__new__(run_gate.OwnedProcess)
                    actor.term_seconds = grace
                    actor.initial = {"pid": 12345, "pgid": 12345, "starttime": 42}
                    actor.pidfd = None
                    actor.process = SimpleNamespace(wait=lambda timeout: 0)
                    actor.out_path = root / f"{grace}.stdout.raw"
                    actor.err_path = root / f"{grace}.stderr.raw"
                    actor.out = actor.out_path.open("xb")
                    actor.err = actor.err_path.open("xb")
                    actor.members = (
                        lambda deadline=None, actor=actor, alive=alive, clock=clock: (
                            [actor.initial] if alive[0] and clock[0] < 70 else []
                        )
                    )

                    def signal_owned(
                        signum,
                        group=False,
                        calls=calls,
                        clock=clock,
                        kill_signal=kill_signal,
                        alive=alive,
                    ):
                        calls.append((signum, group, clock[0]))
                        if signum == kill_signal:
                            alive[0] = False

                    actor.signal_owned = signal_owned

                    def sleep(seconds, clock=clock):
                        clock[0] += seconds

                    with (
                        patch.object(
                            run_gate.signal, "SIGKILL", kill_signal, create=True
                        ),
                        patch.object(
                            run_gate.time, "monotonic", lambda clock=clock: clock[0]
                        ),
                        patch.object(run_gate.time, "sleep", sleep),
                        patch.object(
                            run_gate,
                            "proc_stat",
                            lambda pid: {"state": "S", "starttime": 42},
                        ),
                    ):
                        result = actor.close()
                    self.assertEqual(result["forced_kill"], expected_forced)
                    self.assertEqual(result["closed"], not expected_forced)
                    self.assertEqual(result["actual_exit"], 0)
                    self.assertTrue(result["remaining_observation_complete"])
                    self.assertEqual(result["remaining_processes"], [])
                    self.assertEqual(calls[0][:2], (signal.SIGTERM, False))
                    if expected_forced:
                        self.assertEqual(calls[1][:2], (kill_signal, True))
                        self.assertGreaterEqual(calls[1][2], 15)
                        self.assertIn("TERM15", result["errors"][0])
                    else:
                        self.assertEqual(len(calls), 1)
                        self.assertGreaterEqual(clock[0], 70)
                        self.assertLess(clock[0], 90)


if __name__ == "__main__":
    unittest.main()

"""CPU controls for /proc exit races; no vendor or GPU execution."""

import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_gate


class ProcessLifetime(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "pid": 166303,
            "ppid": 165365,
            "pgid": 166303,
            "sid": 166303,
            "starttime": 206879606,
            "uid": 0,
            "namespaces": {"pid": "pid:[1]", "mnt": "mnt:[2]", "user": "user:[3]"},
        }
        self.known = {(self.previous["pid"], self.previous["starttime"]): self.previous}
        self.live = self.previous | {"state": "S"}

    def verify_after_missing_identity(self, final):
        missing = FileNotFoundError("/proc/166303/ns/mnt")
        reader = Mock(side_effect=[self.live, final])
        identity_reader = Mock(side_effect=missing)
        result = run_gate.verify_known_lifetimes(self.known, reader, identity_reader)
        self.assertEqual(reader.call_count, 2)
        identity_reader.assert_called_once_with(self.previous["pid"])
        return result

    def test_namespace_disappears_then_same_lifetime_is_zombie(self):
        self.assertEqual(
            self.verify_after_missing_identity(self.live | {"state": "Z"}), []
        )

    def test_namespace_disappears_then_pid_is_gone(self):
        for failure in (FileNotFoundError("stat"), ProcessLookupError("stat")):
            with self.subTest(failure=type(failure).__name__):
                self.assertEqual(self.verify_after_missing_identity(failure), [])

    def test_pid_reuse_after_namespace_disappears_still_fails(self):
        for state in ("S", "Z"):
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(ValueError, "PID was reused"),
            ):
                self.verify_after_missing_identity(
                    self.live
                    | {"state": state, "starttime": self.previous["starttime"] + 1}
                )

    def test_live_pid_with_missing_namespace_still_fails(self):
        missing = FileNotFoundError("/proc/166303/ns/mnt")
        reader = Mock(return_value=self.live)
        with self.assertRaises(FileNotFoundError) as raised:
            run_gate.verify_known_lifetimes(
                self.known, reader, Mock(side_effect=missing), deadline=0
            )
        self.assertIs(raised.exception, missing)
        self.assertEqual(reader.call_count, 2)

    def test_identity_permission_error_is_not_swallowed(self):
        denied = PermissionError("/proc/166303/ns/mnt")
        reader = Mock(return_value=self.live)
        with self.assertRaises(PermissionError) as raised:
            run_gate.verify_known_lifetimes(
                self.known, reader, Mock(side_effect=denied)
            )
        self.assertIs(raised.exception, denied)
        reader.assert_called_once_with(self.previous["pid"])

    def test_stat_permission_error_after_missing_identity_is_not_swallowed(self):
        with self.assertRaises(PermissionError):
            self.verify_after_missing_identity(PermissionError("/proc/166303/stat"))

    def test_changed_live_identity_still_fails(self):
        for changes in (
            {"sid": 1},
            {"pgid": 1},
            {"uid": 1},
            {"namespaces": {"pid": "pid:[999]"}},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, "escaped"),
            ):
                run_gate.verify_known_lifetimes(
                    self.known,
                    Mock(return_value=self.live),
                    Mock(return_value=self.previous | changes),
                )

    def test_last_live_sample_rejects_reused_zombie_pid(self):
        with self.assertRaisesRegex(ValueError, "PID was reused"):
            run_gate.same_lifetime_is_live(
                self.previous,
                Mock(return_value=self.live | {"state": "Z", "starttime": 1}),
            )

    def test_last_live_sample_accepts_proven_exit_and_propagates_permission(self):
        for observation in (
            FileNotFoundError("stat"),
            ProcessLookupError("stat"),
            self.live | {"state": "Z"},
        ):
            with self.subTest(observation=observation):
                reader = (
                    Mock(side_effect=observation)
                    if isinstance(observation, Exception)
                    else Mock(return_value=observation)
                )
                self.assertFalse(run_gate.same_lifetime_is_live(self.previous, reader))
        self.assertTrue(
            run_gate.same_lifetime_is_live(self.previous, Mock(return_value=self.live))
        )
        with self.assertRaises(PermissionError):
            run_gate.same_lifetime_is_live(
                self.previous, Mock(side_effect=PermissionError("stat"))
            )

    def test_close_reaps_after_actual_namespace_exit_race_without_kill(self):
        with tempfile.TemporaryDirectory(prefix="musa-lifetime-close-") as directory:
            root = Path(directory)
            actor = run_gate.OwnedProcess.__new__(run_gate.OwnedProcess)
            actor.initial = self.previous
            actor.known = self.known
            actor.term_seconds = run_gate.API_TERM_SECONDS
            actor.pidfd = None
            actor.process = SimpleNamespace(wait=Mock(return_value=0))
            actor.out_path, actor.err_path = root / "stdout.raw", root / "stderr.raw"
            actor.out, actor.err = actor.out_path.open("xb"), actor.err_path.open("xb")
            state = {"closing": False, "namespace_gone": False}

            def stat_reader(pid):
                return self.live | {"state": "Z" if state["namespace_gone"] else "S"}

            def identity_reader(pid):
                if state["closing"]:
                    state["namespace_gone"] = True
                    raise FileNotFoundError("/proc/166303/ns/mnt")
                return self.previous

            def stop(signum, group=False):
                state["closing"] = True

            actor.signal_owned = Mock(side_effect=stop)
            verify = run_gate.verify_known_lifetimes
            with (
                patch.object(run_gate, "proc_stat", stat_reader),
                patch.object(
                    run_gate,
                    "verify_known_lifetimes",
                    lambda known, **kwargs: verify(
                        known, stat_reader, identity_reader, **kwargs
                    ),
                ),
                patch.object(run_gate, "session_processes", return_value=[]),
                patch.object(
                    run_gate.signal,
                    "SIGKILL",
                    getattr(signal, "SIGKILL", 9),
                    create=True,
                ),
            ):
                result = actor.close()
            self.assertTrue(state["namespace_gone"])
            self.assertTrue(result["closed"], result["errors"])
            self.assertTrue(result["remaining_observation_complete"])
            self.assertEqual(result["remaining_processes"], [])
            self.assertEqual(result["actual_exit"], 0)
            self.assertFalse(result["forced_kill"])
            self.assertEqual(result["errors"], [])
            actor.process.wait.assert_called_once_with(timeout=1)
            actor.signal_owned.assert_called_once_with(signal.SIGTERM, group=False)

    def test_namespace_teardown_precedes_zombie_state(self):
        # Reproduce the CI order: missing namespace, still-live stat, then exit.
        for final in (self.live | {"state": "Z"}, FileNotFoundError("stat")):
            with (
                self.subTest(final=final),
                patch.object(run_gate.time, "sleep") as sleep,
            ):
                reader = Mock(side_effect=[self.live, self.live, self.live, final])
                missing = Mock(side_effect=FileNotFoundError("namespace teardown"))
                self.assertEqual(
                    run_gate.verify_known_lifetimes(self.known, reader, missing), []
                )
                self.assertEqual(missing.call_count, 2)
                sleep.assert_called_once()

    def test_missing_namespace_can_recover_only_complete_original_identity(self):
        missing = FileNotFoundError("namespace teardown")
        for changes in ({}, {"namespaces": {"pid": "pid:[other]"}}):
            with (
                self.subTest(changes=changes),
                patch.object(run_gate.time, "sleep") as sleep,
            ):
                identity_reader = Mock(side_effect=[missing, self.previous | changes])
                if changes:
                    with self.assertRaisesRegex(ValueError, "escaped"):
                        run_gate.verify_known_lifetimes(
                            self.known, Mock(return_value=self.live), identity_reader
                        )
                else:
                    self.assertEqual(
                        run_gate.verify_known_lifetimes(
                            self.known, Mock(return_value=self.live), identity_reader
                        ),
                        [self.previous],
                    )
                sleep.assert_called_once()

    def test_delayed_reobservation_reuse_or_permission_remains_fatal(self):
        for failure in (self.live | {"starttime": 1}, PermissionError("stat")):
            with self.subTest(failure=failure), patch.object(run_gate.time, "sleep"):
                reader = Mock(side_effect=[self.live, self.live, failure])
                expected = (
                    PermissionError
                    if isinstance(failure, PermissionError)
                    else ValueError
                )
                with self.assertRaises(expected):
                    run_gate.verify_known_lifetimes(
                        self.known,
                        reader,
                        Mock(side_effect=FileNotFoundError("namespace")),
                    )

    def test_resampling_never_renews_existing_cleanup_deadline(self):
        for deadline, expected_elapsed in ((None, 0.1), (0.015, 0.015), (0, 0)):
            with self.subTest(deadline=deadline):
                clock = [0.0]
                missing = FileNotFoundError("persistent live namespace absence")

                def sleep(seconds, clock=clock):
                    clock[0] += seconds

                with (
                    patch.object(
                        run_gate.time, "monotonic", lambda clock=clock: clock[0]
                    ),
                    patch.object(run_gate.time, "sleep", sleep),
                    self.assertRaises(FileNotFoundError) as caught,
                ):
                    run_gate.verify_known_lifetimes(
                        self.known,
                        Mock(return_value=self.live),
                        Mock(side_effect=missing),
                        deadline=deadline,
                    )
                self.assertIs(caught.exception, missing)
                self.assertAlmostEqual(clock[0], expected_elapsed)

    def test_short_observation_window_is_shared_between_known_pids(self):
        second = self.previous | {"pid": self.previous["pid"] + 1}
        known = self.known | {(second["pid"], second["starttime"]): second}
        clock = [0.0]
        first_samples = [0]

        def identity_reader(pid):
            if pid == self.previous["pid"]:
                first_samples[0] += 1
                if first_samples[0] == 1:
                    raise FileNotFoundError("first teardown")
                return self.previous
            raise FileNotFoundError("second persistent absence")

        def sleep(seconds, clock=clock):
            clock[0] += seconds

        with (
            patch.object(run_gate.time, "monotonic", lambda clock=clock: clock[0]),
            patch.object(run_gate.time, "sleep", sleep),
            self.assertRaises(FileNotFoundError),
        ):
            run_gate.verify_known_lifetimes(
                known, lambda pid: self.live | {"pid": pid}, identity_reader
            )
        self.assertAlmostEqual(clock[0], 0.1)

    def test_close_does_not_renew_expired_term_window_before_kill(self):
        with tempfile.TemporaryDirectory(prefix="musa-expired-close-") as directory:
            root = Path(directory)
            actor = run_gate.OwnedProcess.__new__(run_gate.OwnedProcess)
            actor.initial, actor.known = self.previous, self.known
            actor.term_seconds, actor.pidfd = 1, None
            actor.process = SimpleNamespace(wait=Mock(return_value=0))
            actor.out_path, actor.err_path = root / "stdout.raw", root / "stderr.raw"
            actor.out, actor.err = actor.out_path.open("xb"), actor.err_path.open("xb")
            actor.signal_owned = Mock()
            clock = [0.0]

            def stat_reader(pid):
                return self.live | {"state": "Z" if clock[0] >= 1.05 else "S"}

            def identity_reader(pid):
                if clock[0] >= 1:
                    raise FileNotFoundError("Namespace gone at expired TERM deadline")
                return self.previous

            def sleep(seconds):
                clock[0] += seconds

            verify = run_gate.verify_known_lifetimes
            with (
                patch.object(run_gate.time, "monotonic", lambda: clock[0]),
                patch.object(run_gate.time, "sleep", sleep),
                patch.object(run_gate, "proc_stat", stat_reader),
                patch.object(run_gate, "session_processes", return_value=[]),
                patch.object(
                    run_gate,
                    "verify_known_lifetimes",
                    lambda known, **kwargs: verify(
                        known, stat_reader, identity_reader, **kwargs
                    ),
                ),
                patch.object(
                    run_gate.signal,
                    "SIGKILL",
                    getattr(signal, "SIGKILL", 9),
                    create=True,
                ),
            ):
                result = actor.close()
            self.assertFalse(result["closed"])
            self.assertFalse(result["remaining_observation_complete"])
            self.assertFalse(result["forced_kill"])
            self.assertIn("FileNotFoundError", result["errors"][0])
            self.assertEqual(clock[0], 1)
            actor.process.wait.assert_not_called()
            actor.signal_owned.assert_called_once_with(signal.SIGTERM, group=False)


if __name__ == "__main__":
    unittest.main()

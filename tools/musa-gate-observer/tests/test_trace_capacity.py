"""Portable stdlib observer regressions; no GPU, model or Gate acceptance."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "musa_gate_observer"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "capacity_" + name, SOURCE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace, consumer = load("trace"), load("summarize")


class TraceCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="observer-capacity-")
        self.addCleanup(self.temp.cleanup)
        self.config = SimpleNamespace(
            directory=Path(self.temp.name),
            data={
                "run_id": "synthetic-run",
                "observer_source_sha256": "a" * 64,
                "source_binding_id": "b" * 64,
            },
            route={
                "route_id": "synthetic-graph",
                "model": "synthetic-model",
                "mode": "graph",
            },
            original={"path": "/synthetic/config", "bytes": 1, "sha256": "c" * 64},
        )
        self.identity = mock.patch.object(
            trace, "process_identity", return_value=(4321, "123", "pid:[123]")
        )
        self.identity.start()
        self.addCleanup(self.identity.stop)
        # The real producer runs on Linux. Windows CRT text descriptors can
        # translate LF into CRLF, so use a binary local fixture descriptor.
        native_open = trace.os.open
        self.binary_open = mock.patch.object(
            trace.os,
            "open",
            side_effect=lambda path, flags, *args, **kwargs: native_open(
                path, flags | getattr(trace.os, "O_BINARY", 0), *args, **kwargs
            ),
        )
        self.binary_open.start()
        self.addCleanup(self.binary_open.stop)
        self.routes = {self.config.route["route_id"]: {"route": self.config.route}}

    def producer(self):
        t = trace.Trace(self.config, "worker", 0)
        self.addCleanup(t.close)
        return t

    def expanded(self, raw):
        state = {}
        return [
            consumer._expand_event(consumer._json(line), state)
            for line in raw.splitlines()
        ]

    def rejected(self, rows):
        raw = b"".join(trace.canonical(r) + b"\n" for r in rows)
        with self.assertRaises(consumer.ContractError):
            consumer._events(raw, self.config.data, self.routes, set())

    def test_dynamic_graph_and_request_fields_roundtrip_legacy_equal(self):
        t = self.producer()
        dynamic = {
            "rank": 0,
            "engine_request_ids": ["request-a", "request-b"],
            "execute_call_id": "execute-1",
            "capture_id": "capture-1",
            "graph_generation": 3,
            "graph_id": "graph-7",
            "wrapper_uid": "wrapper-2",
            "batch": {
                "num_tokens": 17,
                "num_reqs": 2,
                "uniform": False,
                "has_lora": False,
                "num_active_loras": 0,
            },
            "buffers": {"address": 1234567, "shape": [17, 128], "stride": [128, 1]},
            "thread_id": 999,
            "nested": {"list": [None, False, 1.5, {"text": "UTF8 \u4e2d"}]},
        }
        t.activation(["synthetic-hook"], {})
        for kind in (
            "scheduler_begin",
            "capture_success",
            "replay_success",
            "scheduler_return",
        ):
            t.emit(kind, dynamic)
        t.close()
        raw = t.path.read_bytes()
        physical = [consumer._json(line) for line in raw.splitlines()]
        self.assertIn("profile", physical[0])
        self.assertTrue(all("profile" not in row for row in physical[1:]))
        self.assertEqual(
            physical[0]["profile_id"], trace.digest(physical[0]["profile"])
        )
        expanded = self.expanded(raw)
        self.assertTrue(all(row["data"] == dynamic for row in expanded[1:5]))
        self.assertEqual(expanded[-1]["status"], "ok")
        self.assertEqual(expanded[-1]["data"]["errors"], [])
        legacy = b"".join(trace.canonical(row) + b"\n" for row in expanded)
        self.assertEqual(
            consumer._events(raw, self.config.data, self.routes, set()),
            consumer._events(legacy, self.config.data, self.routes, set()),
        )

    def test_unknown_mismatched_and_redefined_profile_refs_reject(self):
        t = self.producer()
        t.activation([], {})
        t.close()
        rows = [consumer._json(line) for line in t.path.read_bytes().splitlines()]
        self.rejected([rows[1]])
        mismatch = copy.deepcopy(rows)
        mismatch[1]["profile_id"] = "0" * 64
        self.rejected(mismatch)
        bad_profile = copy.deepcopy(rows)
        bad_profile[0]["profile"]["pid"] += 1
        self.rejected(bad_profile)
        redefine = copy.deepcopy(rows)
        redefine[1]["profile"] = redefine[0]["profile"]
        self.rejected(redefine)

    def test_recomputed_digest_wrong_binding_and_original_checks_reject(self):
        t = self.producer()
        t.activation([], {})
        t.close()
        rows = [consumer._json(line) for line in t.path.read_bytes().splitlines()]
        wrong = copy.deepcopy(rows)
        wrong[0]["profile"]["run_id"] = "different-run"
        digest = trace.digest(wrong[0]["profile"])
        for row in wrong:
            row["profile_id"] = digest
        self.rejected(wrong)
        gap = copy.deepcopy(rows)
        gap[1]["seq"] = 3
        self.rejected(gap)
        error = copy.deepcopy(rows)
        error[1]["status"] = "error"
        self.rejected(error)
        mixed = [rows[0], self.expanded(t.path.read_bytes())[1]]
        self.rejected(mixed)

    def test_overflow_latches_bounded_errors_and_failure_end(self):
        with (
            mock.patch.object(trace, "MAX_FILE", 32768),
            mock.patch.object(trace, "END_RESERVE", 8192),
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                t = self.producer()
                t.activation([], {})
                t.emit("graph_call", {"oversized": "x" * 30000})
                for _ in range(1000):
                    t.emit(
                        "replay_success", {"must_not_be_presented_as_observed": True}
                    )
                for _ in range(20):
                    t.fail(RuntimeError("x" * 20000))
                t.close()
                t.close()
            raw = t.path.read_bytes()
            rows = self.expanded(raw)
            self.assertEqual(len(rows), 2)
            end = rows[-1]
            self.assertEqual(end["kind"], "telemetry_end")
            self.assertEqual(end["status"], "error")
            self.assertEqual(end["data"]["error_count"], 21)
            self.assertEqual(end["data"]["dropped_events"], 1021)
            self.assertEqual(len(end["data"]["errors"]), 4)
            self.assertTrue(end["data"]["cap_exceeded"])
            self.assertTrue(end["data"]["io_failed"])
            self.assertLess(len(trace.canonical(end["data"])), trace.END_RESERVE)
            self.assertLessEqual(len(raw), trace.MAX_FILE)
            self.assertEqual(stderr.getvalue().count("MUSA_GATE_OBSERVER_IO_ERROR"), 1)
            self.assertLess(len(stderr.getvalue().encode()), 20000)
            self.rejected(rows)

    def test_write_failure_does_not_raise_or_repeat_warning(self):
        t = self.producer()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            mock.patch.object(
                trace.os, "write", side_effect=OSError("synthetic write failure")
            ),
        ):
            t.activation([], {})
            for _ in range(100):
                t.emit("graph_call", {"x": 1})
            t.close()
        self.assertTrue(t.closed)
        self.assertTrue(t.io_failed)
        self.assertEqual(stderr.getvalue().count("MUSA_GATE_OBSERVER_IO_ERROR"), 1)
        self.assertEqual(t.path.read_bytes(), b"")
        with self.assertRaises(consumer.ContractError):
            consumer._events(b"", self.config.data, self.routes, set())


if __name__ == "__main__":
    unittest.main()

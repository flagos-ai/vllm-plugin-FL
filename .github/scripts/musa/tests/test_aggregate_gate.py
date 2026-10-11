"""Pure consumer-contract filesystem fixtures; no model/vendor/GPU execution."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import aggregate_gate  # noqa: E402
import gate_common  # noqa: E402
from aggregate_gate import aggregate, canonical_sha, original_scenario  # noqa: E402
from gate_common import (CHECKS, GATE_FILES, ROUTES, SCENARIOS, TEST_FILES, fingerprint, fingerprint_streaming,
                         read_bytes, strict_json, write_json)  # noqa: E402
from run_gate import route_cleanup_complete, verify_known_lifetimes  # noqa: E402


class Contract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="musa-gate-consumer-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        kit = self.root / "fixture-kit"
        kit.mkdir()
        (kit / "images").mkdir()
        gate_pins = {}
        for name in GATE_FILES:
            (kit / name).write_bytes(("fixture source " + name + "\n").encode())
            gate_pins[name] = fingerprint(kit / name)
        package = self.root / "fixture-observer-package"
        package.mkdir()
        package_pins = {}
        for name in ("__init__.py", "trace.py", "server.py", "pytest_plugin.py", "summarize.py"):
            (package / name).write_bytes(("# fixture " + name + "\n").encode())
            package_pins[name] = fingerprint(package / name)
        own_sources = {name: {key: ref[key] for key in ("bytes", "sha256")}
                       for name, ref in package_pins.items()}
        own_digest = canonical_sha(own_sources)
        self.manifest = {"schema_version": 1, "run_id": "1" * 32,
                         "source_commit": "a" * 40, "observer_source_sha256": own_digest,
                         "routes": [], "errors": [],
                         "fixture_scope": "Consumer-only synthetic documents, not runtime observations"}
        binding = write_json(self.root / "binding.json", {"source_commit": "a" * 40,
                            "original_gate": gate_pins, "observer_sources": package_pins})
        self.manifest["source_binding"] = binding
        self.manifest["source_binding_id"] = binding["sha256"]
        config = {"run_id": "1" * 32, "source_binding_id": binding["sha256"],
                  "observer_source_sha256": own_digest,
                  "observer_sources": own_sources, "gate_kit_files": gate_pins,
                  "sources": {"fixture-source": gate_pins["case_utils.py"]}}
        self.manifest["observer_config"] = write_json(self.root / "config.json", config)
        empty = self.root / "empty.raw"
        empty.write_bytes(b"")
        raw = fingerprint(empty)
        command = {"attempted": True, "actual_exit": 0, "timed_out": False,
                   "errors": [], "cleanup": {"closed": True}, "stdout": raw, "stderr": raw}
        self.manifest["observer_command"] = write_json(self.root / "observer-command.json", command)
        self.observer = {"schema_version": 1, "run_id": "1" * 32,
                         "source_binding_id": binding["sha256"],
                         "observer_source_sha256": own_digest, "observer_complete": True,
                         "graph_observation_complete": True, "routes": [],
                         "raw_inputs": [raw], "errors": []}
        for index, (route_id, model, mode, port) in enumerate(ROUTES):
            route_dir = self.root / route_id
            route_dir.mkdir()
            server = {"pid": 100 + index, "starttime": 10000 + index,
                      "sid": 100 + index, "pgid": 100 + index, "ppid": 50}
            preflight = write_json(route_dir / "preflight.json", {
                "passed": True, "logical_device_count": 4,
                "source_binding_id": binding["sha256"]})
            route = {"route_id": route_id, "model": model, "mode": mode,
                     "model_path": "/data/models/" + model, "served_model_name": model,
                     "port": port, "source_binding_id": binding["sha256"],
                     "preflight": preflight, "ready": {"ready": True},
                     "server": {"started": True, "identity": server},
                     "cleanup": {"closed": True, "actual_exit": -15, "errors": [],
                                 "stdout": raw, "stderr": raw},
                     "pytest": [], "scenarios": {}, "errors": []}
            for filename in TEST_FILES:
                junit = route_dir / (filename + ".xml")
                count = 1 if filename == "test_mix_text_image.py" else 2
                junit.write_text("<testsuites><testsuite>" + "<testcase/>" * count
                                 + "</testsuite></testsuites>", encoding="utf-8")
                route["pytest"].append({"file": filename, **command,
                                        "junit": fingerprint(junit)})
            identities = []
            observed_processes = []
            for rank in range(-1, 4):
                pid = server["pid"] if rank == -1 else server["pid"] + 1000 + rank
                ticks = server["starttime"] + rank + 1
                identities.append({"pid": pid, "starttime": ticks, "sid": server["sid"],
                                   "lifetime_stable": True, "namespaces": {"pid": "pid:[123]"}})
                observed_processes.append({"role": "api" if rank == -1 else "worker",
                                           "pid": pid, "starttime": ticks, "rank": rank,
                                           "process_key": f"{pid}:{ticks}", "hooks": ["fixture"],
                                           "sources_verified": True, "telemetry_complete": True})
            for n, item in enumerate(route["pytest"]):
                client = {"pid": server["pid"] + 2000 + n,
                          "starttime": server["starttime"] + 100 + n,
                          "sid": server["pid"] + 2000 + n,
                          "lifetime_stable": True, "owner": "pytest", "entrypoint": item["file"],
                          "namespaces": {"pid": "pid:[123]"}}
                item["identity"] = client
                identities.append(client)
                observed_processes.append({"role": "client", "pid": client["pid"],
                                           "starttime": client["starttime"], "rank": None,
                                           "process_key": f"{client['pid']}:{client['starttime']}",
                                           "sources_verified": True, "telemetry_complete": True})
            route["process_attestations"] = write_json(route_dir / "processes.json", {
                "server_identity": server, "processes": identities, "errors": []})
            observed = {"route_id": route_id, "model": model, "mode": mode,
                        "expected_request_count": 26, "observed_logical_count": 26,
                        "quality_complete": True, "graph_complete": mode == "graph",
                        "processes": observed_processes, "scenario_outputs": [],
                        "requests": [], "errors": []}
            ordinal = 0
            for scenario, count in SCENARIOS.items():
                inputs = [{"request_id": f"original-{n}"} for n in range(count)]
                if scenario == "text_single":
                    inputs[0].update({"min_length": 256, "max_tokens": 512})
                outputs = []
                for n in range(count):
                    response_id = f"{route_id}-{scenario}-{n}"
                    outputs.append({"request_id": inputs[n]["request_id"], "passed": True,
                                    "checks": dict.fromkeys(CHECKS, True),
                                    "response": {"id": response_id}})
                    observed["requests"].append({"logical_id": f"{route_id}:{scenario}:{n}",
                                                  "scenario": scenario, "ordinal": ordinal,
                                                  "response_id": response_id, "passed": True,
                                                  "graph_ranks_complete": [0, 1, 2, 3]
                                                  if mode == "graph" else [], "errors": []})
                    ordinal += 1
                report = {"schema_version": 1, "case": {"scenario": scenario,
                          "model_path": route["model_path"], "served_model_name": model,
                          "port": str(port), "concurrency": count},
                          "input": inputs, "output": outputs,
                          "summary": {"request_count": count, "passed_count": count,
                                      "failed_count": 0, "passed": True}}
                ref = write_json(route_dir / (scenario + ".json"), report)
                route["scenarios"][scenario] = ref
                observed["scenario_outputs"].append({"scenario": scenario, **ref})
            self.manifest["routes"].append(route)
            self.observer["routes"].append(observed)
        # Adapt the fixture to the actual companion summarizer's flat schema.
        config["routes"] = []
        flat = {key: self.observer[key] for key in ("schema_version", "run_id", "source_binding_id",
                "observer_source_sha256", "observer_complete", "graph_observation_complete", "errors")}
        flat.update({"acceptance": dict.fromkeys(("source", "GPU", "ABI", "model", "Gate"), False),
                     "scope": "full_matrix", "full_matrix_complete": True,
                     "full_matrix_observer_complete": True,
                     "full_matrix_graph_observation_complete": True,
                     "counts": {"logical_results": 104, "scenario_results": 20,
                                "api_responses": 104, "queued_requests": 104,
                                "graph_request_joins": 52, "graph_request_rank_joins": 208},
                     "process_identities": {}, "logical_response_engine_joins": [],
                     "graph_request_joins": [], "graph_ranks_complete": [0, 1, 2, 3],
                     "scenario_output_fingerprint_claims": []})
        for route, observed in zip(self.manifest["routes"], self.observer["routes"]):
            rid = route["route_id"]
            processes = []
            for old in observed["processes"]:
                process = {key: old[key] for key in ("role", "rank", "pid", "process_key")}
                process.update({"starttime": str(old["starttime"]), "pid_namespace": "pid:[123]"})
                if process["role"] == "client":
                    original_client = next(item for item in route["pytest"]
                                           if item["identity"]["pid"] == process["pid"])
                    process["loaded_test_modules"] = [Path(original_client["file"]).stem]
                processes.append(process)
            flat["process_identities"][rid] = processes
            expected_route = {"route_id": rid, "model": route["model"], "mode": route["mode"],
                              "expected_ranks": [0, 1, 2, 3], "requests": []}
            for old in observed["requests"]:
                scenario = old["scenario"]
                report = json.loads(Path(route["scenarios"][scenario]["path"]).read_text())
                local = int(old["response_id"].rsplit("-", 1)[1])
                input_row, output_row = report["input"][local], report["output"][local]
                expected = {"logical_id": old["logical_id"], "scenario": scenario,
                            "ordinal": old["ordinal"], "original_request_id": input_row["request_id"],
                            "request_input_sha256": canonical_sha(input_row), "required_checks": sorted(CHECKS)}
                expected_route["requests"].append(expected)
                logical = {key: expected[key] for key in ("logical_id", "scenario", "ordinal",
                            "original_request_id", "request_input_sha256")}
                logical.update({"route_id": rid, "response_id": old["response_id"],
                                "api_call_id": old["logical_id"] + ":api",
                                "engine_request_id": old["logical_id"] + ":engine",
                                "checks": output_row["checks"], "passed": True,
                                "original_result_sha256": canonical_sha(output_row), "original_error": None})
                flat["logical_response_engine_joins"].append(logical)
                if route["mode"] == "graph":
                    graph = {key: logical[key] for key in ("route_id", "logical_id", "response_id",
                             "api_call_id", "engine_request_id")}
                    graph.update({"ranks": [0, 1, 2, 3], "rank_evidence": []})
                    for worker in (p for p in processes if p["role"] == "worker"):
                        evidence = {key: worker[key] for key in ("rank", "process_key", "pid", "starttime", "pid_namespace")}
                        evidence.update({"graph_generation": 1, "capture_seq": 2,
                                         "scheduler_seq": 3, "replay_seq": 4})
                        graph["rank_evidence"].append(evidence)
                    flat["graph_request_joins"].append(graph)
            config["routes"].append(expected_route)
            for item in observed["scenario_outputs"]:
                flat["scenario_output_fingerprint_claims"].append({"route_id": rid,
                    "scenario": item["scenario"], "output_fingerprint": {key: item[key] for key in ("path", "bytes", "sha256")}})
        self.manifest["observer_config"] = write_json(self.root / "config-final.json", config)
        traces = []
        for n in range(32):
            trace = self.root / f"fixture-trace-{n}.jsonl"
            trace.write_bytes(b'{"fixture":true}\n')
            traces.append(fingerprint(trace))
        flat["input_fingerprints"] = [self.manifest["observer_config"], *traces]
        self.observer = flat

    def result(self, observer=True):
        manifest_path = self.root / "route-manifest.json"
        observer_path = self.root / "observer-summary.json"
        write_json(manifest_path, self.manifest)
        if observer:
            write_json(observer_path, self.observer)
        return aggregate(manifest_path, observer_path if observer else None)

    def test_complete_four_route_contract(self):
        result = self.result()
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["expected_logical_requests"], 104)
        self.assertEqual(len(result["routes"]), 4)


    def test_observer_stream_above_old_cap_is_hashed_in_bounded_chunks(self):
        trace = Path(self.observer["input_fingerprints"][1]["path"])
        block = b"x" * (1024 * 1024)
        digest = hashlib.sha256()
        with trace.open("wb") as stream:
            for _ in range(64):
                stream.write(block)
                digest.update(block)
            stream.write(b"x")
            digest.update(b"x")
        expected = {"path": str(trace), "bytes": 64 * 1024 * 1024 + 1,
                    "sha256": digest.hexdigest()}
        read_sizes = []
        original_read = gate_common.os.read

        def measured_read(fd, size):
            read_sizes.append(size)
            return original_read(fd, size)

        with mock.patch.object(gate_common.os, "read", side_effect=measured_read), \
                mock.patch.object(gate_common, "read_bytes",
                                  side_effect=AssertionError("Body-retaining reader used")):
            self.assertEqual(fingerprint_streaming(trace, 2 * 1024**3, self.root), expected)
        self.assertGreater(len(read_sizes), 1024)
        self.assertLessEqual(max(read_sizes), 65536)
        self.observer["input_fingerprints"][1] = expected
        self.assertTrue(self.result()["gate_passed"])

    def test_observer_file_budget_boundary_and_overflow(self):
        cap = self.observer["input_fingerprints"][1]["bytes"]
        with mock.patch.object(aggregate_gate, "OBSERVER_RAW_MAX_FILE_BYTES", cap):
            self.assertTrue(self.result()["gate_passed"])
        with mock.patch.object(aggregate_gate, "OBSERVER_RAW_MAX_FILE_BYTES", cap - 1):
            result = aggregate(self.root / "route-manifest.json", self.root / "observer-summary.json")
        self.assertFalse(result["gate_passed"])
        self.assertTrue(all(route["business_passed"] for route in result["routes"]))
        self.assertIn("Input exceeds cap", result["routes"][0]["errors"][0])

    def test_observer_total_budget_boundary_and_overflow(self):
        cap = sum(ref["bytes"] for ref in self.observer["input_fingerprints"])
        with mock.patch.object(aggregate_gate, "OBSERVER_RAW_MAX_TOTAL_BYTES", cap):
            self.assertTrue(self.result()["gate_passed"])
        with mock.patch.object(aggregate_gate, "OBSERVER_RAW_MAX_TOTAL_BYTES", cap - 1), \
                mock.patch.object(aggregate_gate, "fingerprint_streaming",
                                  side_effect=AssertionError("Over-budget input was read")):
            result = aggregate(self.root / "route-manifest.json", self.root / "observer-summary.json")
        self.assertFalse(result["gate_passed"])
        self.assertTrue(all(route["business_passed"] for route in result["routes"]))
        self.assertIn("total cap exceeded", result["routes"][0]["errors"][0])

    def test_observer_stream_hash_mismatch_rejects(self):
        trace = Path(self.observer["input_fingerprints"][1]["path"])
        trace.write_bytes(b'{"fixture":fail}\n')
        self.assertFalse(self.result()["gate_passed"])

    def test_streaming_fingerprint_empty_exact_cap_and_overflow(self):
        path = self.root / "streaming-boundary.raw"
        for data in (b"", b"x", b"abcd"):
            path.write_bytes(data)
            with self.subTest(bytes=len(data)):
                self.assertEqual(fingerprint_streaming(path, len(data), self.root),
                                 fingerprint(path, len(data), self.root))
                if data:
                    with mock.patch.object(gate_common.os, "open") as opened:
                        with self.assertRaisesRegex(ValueError, "Input exceeds cap"):
                            fingerprint_streaming(path, len(data) - 1, self.root)
                    opened.assert_not_called()

    def test_streaming_fingerprint_root_escape_rejects(self):
        outside = self.root.parent / (self.root.name + "-streaming-outside")
        outside.write_bytes(b"out")
        self.addCleanup(outside.unlink)
        with self.assertRaises(ValueError):
            fingerprint_streaming(outside, root=self.root)

    def test_streaming_fingerprint_changed_file_rejects(self):
        path = self.root / "streaming-mutation.raw"
        original_read = gate_common.os.read
        for data in (b"abcdx", b"abc"):
            path.write_bytes(b"abcd")
            changed = False

            def mutate_after_read(fd, size):
                nonlocal changed
                part = original_read(fd, size)
                if not changed:
                    changed = True
                    path.write_bytes(data)
                return part

            with self.subTest(after_size=len(data)), \
                    mock.patch.object(gate_common.os, "read", side_effect=mutate_after_read), \
                    self.assertRaises(ValueError):
                fingerprint_streaming(path, 16, self.root)

    def test_missing_observer_rejects_full_gate(self):
        result = self.result(False)
        self.assertFalse(result["gate_passed"])
        self.assertTrue(all(route["business_passed"] for route in result["routes"]))

    def test_unknown_graph_coverage_rejects(self):
        self.observer["graph_observation_complete"] = False
        self.assertFalse(self.result()["gate_passed"])

    def test_missing_rank_rejects_graph_route(self):
        self.observer["graph_request_joins"][0]["ranks"] = [0, 1, 2]
        self.assertFalse(self.result()["gate_passed"])

    def test_pid_lifetime_mismatch_rejects(self):
        self.observer["process_identities"]["27b-graph"][1]["starttime"] = "99999999"
        self.assertFalse(self.result()["gate_passed"])

    def test_missing_original_pytest_client_rejects(self):
        self.observer["process_identities"]["27b-eager"].pop()
        self.assertFalse(self.result()["gate_passed"])

    def test_original_row_hash_join_rejects(self):
        self.observer["logical_response_engine_joins"][0]["original_result_sha256"] = "0" * 64
        self.assertFalse(self.result()["gate_passed"])

    def test_namespace_join_rejects(self):
        self.observer["process_identities"]["27b-eager"][0]["pid_namespace"] = "pid:[999]"
        self.assertFalse(self.result()["gate_passed"])

    def test_incomplete_raw_trace_set_rejects(self):
        self.observer["input_fingerprints"].pop()
        self.assertFalse(self.result()["gate_passed"])

    def test_partial_route_is_not_a_full_matrix(self):
        self.observer["scope"] = "route_only"
        self.observer["full_matrix_complete"] = False
        self.assertFalse(self.result()["gate_passed"])

    def test_original_payload_change_rejects(self):
        ref = self.manifest["routes"][0]["scenarios"]["text_single"]
        Path(ref["path"]).write_bytes(Path(ref["path"]).read_bytes() + b" ")
        self.assertFalse(self.result()["gate_passed"])

    def test_failed_original_check_rejects_even_with_new_fingerprints(self):
        route = self.manifest["routes"][0]
        ref = route["scenarios"]["text_single"]
        report = json.loads(Path(ref["path"]).read_text())
        report["output"][0]["checks"]["minimum_length"] = False
        Path(ref["path"]).write_text(json.dumps(report), encoding="utf-8")
        route["scenarios"]["text_single"] = fingerprint(ref["path"])
        self.assertFalse(self.result()["gate_passed"])

    def test_pytest_timeout_is_not_success(self):
        self.manifest["routes"][0]["pytest"][0].update({"actual_exit": None, "timed_out": True})
        self.assertFalse(self.result()["gate_passed"])

    def test_skip_in_junit_rejects(self):
        item = self.manifest["routes"][0]["pytest"][0]
        Path(item["junit"]["path"]).write_text(
            "<testsuite><testcase><skipped/></testcase><testcase/></testsuite>", encoding="utf-8")
        item["junit"] = fingerprint(item["junit"]["path"])
        self.assertFalse(self.result()["gate_passed"])

    def test_reordered_logical_observer_join_rejects(self):
        self.observer["logical_response_engine_joins"][1]["ordinal"] = 0
        self.assertFalse(self.result()["gate_passed"])

    def test_count_loss_rejects_original_scenario(self):
        ref = self.manifest["routes"][0]["scenarios"]["text_concurrent_8"]
        report = json.loads(Path(ref["path"]).read_text())
        report["output"].pop()
        with self.assertRaises(ValueError):
            original_scenario(report, "text_concurrent_8")

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                strict_json(data)

    def test_root_escape_rejected(self):
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_bytes(b"out")
        self.addCleanup(outside.unlink)
        with self.assertRaises(ValueError):
            read_bytes(outside, root=self.root)

    def cleanup_fixture(self):
        complete = {"closed": True, "errors": [], "remaining_observation_complete": True,
                    "remaining_processes": []}
        return {"server": {"attempted": True}, "cleanup": copy.deepcopy(complete),
                "preflight_command": {"cleanup": copy.deepcopy(complete)},
                "pytest": [{"attempted": True, "cleanup": copy.deepcopy(complete)} for _ in range(3)]}

    def test_next_route_requires_all_actor_cleanup(self):
        self.assertTrue(route_cleanup_complete(self.cleanup_fixture()))

    def test_failed_preflight_cleanup_blocks_next_route(self):
        route = self.cleanup_fixture()
        route["server"]["attempted"] = False
        route["preflight_command"]["cleanup"]["closed"] = False
        self.assertFalse(route_cleanup_complete(route))

    def test_failed_pytest_cleanup_blocks_next_route(self):
        route = self.cleanup_fixture()
        route["pytest"][0]["cleanup"]["remaining_processes"] = [{"pid": 55}]
        self.assertFalse(route_cleanup_complete(route))

    def test_unknown_remaining_state_blocks_next_route(self):
        route = self.cleanup_fixture()
        del route["cleanup"]["remaining_observation_complete"]
        self.assertFalse(route_cleanup_complete(route))

    def test_known_child_new_session_is_not_treated_as_gone(self):
        old = {"pid": 10, "starttime": 200, "pgid": 10, "sid": 10, "uid": 0,
               "namespaces": {"pid": "pid:[1]"}}
        current = old | {"sid": 20, "pgid": 20}
        with self.assertRaises(ValueError):
            verify_known_lifetimes({(10, 200): old},
                                   stat_reader=lambda pid: current | {"state": "S"},
                                   identity_reader=lambda pid: current)

    def test_known_pid_reuse_is_unknown(self):
        old = {"pid": 10, "starttime": 200}
        with self.assertRaises(ValueError):
            verify_known_lifetimes({(10, 200): old},
                                   stat_reader=lambda pid: {"starttime": 201, "state": "S"},
                                   identity_reader=lambda pid: {})


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    print(json.dumps({"scope": "Local stdlib consumer filesystem contract only",
                      "tests_run": result.testsRun, "failures": len(result.failures),
                      "errors": len(result.errors), "skips": len(result.skipped),
                      "vendor_gpu_model_execution": False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)

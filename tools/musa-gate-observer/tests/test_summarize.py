"""Synthetic consumer-contract tests only: no GPU, model or production imports."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "musa_gate_observer" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("pure_observer_consumer", SOURCE)
consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consumer)


class ConsumerContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="observer-consumer-contract-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.config = {
            "schema_version": 1, "run_id": "synthetic-run",
            "source_binding_id": "d" * 64,
            "observer_sources": {name: {"bytes": 1, "sha256": "a" * 64}
                                 for name in consumer.OBSERVER_MEMBERS},
            "sources": {role + "-source": {"path": "/synthetic/" + role,
                                           "bytes": 1, "sha256": "a" * 64}
                        for role in ("api", "worker", "client")},
            "required_role_sources": {role: [role + "-source"] for role in ("api", "worker", "client")},
            "required_role_hooks": {role: ["synthetic-" + role + "-hook"]
                                    for role in ("api", "worker", "client")},
            "routes": [],
        }
        self.config["gate_kit_files"] = {
            name: {"path": "/synthetic/gate/" + name, "bytes": len(name) + 1,
                   "sha256": hashlib.sha256(("synthetic-kit:" + name).encode()).hexdigest()}
            for name in consumer.GATE_MEMBERS
        }
        for name in consumer.GATE_MODULES:
            self.config["sources"][name] = copy.deepcopy(self.config["gate_kit_files"][name + ".py"])
        for name in ("httpx._client", "openai._base_client"):
            self.config["sources"][name] = {"path": "/synthetic/sdk/" + name,
                                            "bytes": 1, "sha256": "a" * 64}
        self.config["required_role_sources"]["client"] += sorted(
            consumer.GATE_MODULES | {"httpx._client", "openai._base_client"})
        canonical = json.dumps(self.config["observer_sources"], sort_keys=True,
                               separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        self.config["observer_source_sha256"] = hashlib.sha256(canonical).hexdigest()
        self.events, self.streams = {}, {}
        for model in ("Qwen3.6-27B", "Qwen3.6-35B-A3B"):
            for mode in ("eager", "graph"):
                rid = ("m27" if model.endswith("27B") else "m35") + "-" + mode
                requests = []
                for scenario, count in consumer.SCENARIOS.items():
                    for number in range(count):
                        ordinal = len(requests)
                        lid = rid + ":" + scenario + ":" + str(number)
                        requests.append({
                            "logical_id": lid, "scenario": scenario, "ordinal": ordinal,
                            "original_request_id": "original:" + scenario + ":" + str(number),
                            "request_input_sha256": hashlib.sha256(lid.encode()).hexdigest(),
                            "required_checks": sorted(consumer.CHECKS),
                        })
                route = {"route_id": rid, "model": model, "mode": mode,
                         "expected_ranks": [0, 1, 2, 3], "requests": requests}
                self.config["routes"].append(route)
                self._route_events(route)
        self.config_path = self.root / "config.json"
        self._save_config()

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config) + "\n", encoding="utf-8")

    def _ids(self, request):
        lid = request["logical_id"]
        return "api:" + lid, "response:" + lid, "engine:" + lid

    def _stream(self, route, role, records, rank=None, client_module=None):
        number = len(self.events) + 1
        pid, starttime = 1000 + number, str(100000 + number)
        process = "synthetic:" + str(pid) + ":" + starttime
        activation = {"role": role, "sources_verified": True,
                      "hooks": ["synthetic-" + role + "-hook"],
                      "sources": {name: self.config["sources"][name]
                                  for name in self.config["required_role_sources"][role]}}
        if rank is not None:
            activation["rank"] = rank
        if role == "client":
            activation.update(loaded_test_modules=[client_module],
                              gate_kit_files=copy.deepcopy(self.config["gate_kit_files"]))
        records = [("activation", activation)] + records + [("telemetry_end", {"errors": []})]
        events = []
        for sequence, (kind, data) in enumerate(records, 1):
            events.append({
                "schema_version": 1, "run_id": self.config["run_id"],
                "route_id": route["route_id"], "model": route["model"], "mode": route["mode"],
                "event_id": process + ":" + str(sequence), "pid": pid, "starttime": starttime,
                "pid_namespace": "pid:[10000]", "time_ns": 1000000 + sequence,
                "monotonic_ns": 10000 + sequence,
                "process_key": process, "seq": sequence, "kind": kind, "status": "ok",
                "observer_source_sha256": self.config["observer_source_sha256"],
                "source_binding_id": self.config["source_binding_id"], "data": copy.deepcopy(data),
            })
        stream_scope = client_module if role == "client" else rank
        path = self.root / (route["route_id"] + "-" + role + "-" + str(stream_scope) + ".jsonl")
        self.events[path] = events
        self.streams[route["route_id"], role, stream_scope] = path
        if role == "client" and client_module == "test_text":
            # Convenient test helper alias; it does not add or merge an event file.
            self.streams[route["route_id"], role, None] = path
        self._save(path)

    def _save(self, path):
        path.write_bytes(b"".join((json.dumps(e) + "\n").encode() for e in self.events[path]))

    def _insert(self, role, kind, data, rank=None, rid="m27-graph"):
        path = self.streams[rid, role, rank]
        event = copy.deepcopy(self.events[path][-1])
        event.update(kind=kind, data=copy.deepcopy(data))
        self.events[path].insert(-1, event)
        for index, item in enumerate(self.events[path], 1):
            item.update(seq=index, event_id=item["process_key"] + ":" + str(index),
                        time_ns=1000000 + index, monotonic_ns=10000 + index)
        self._save(path)

    def _route_events(self, route):
        api, client = [], []
        for request in route["requests"]:
            aid, response, engine = self._ids(request)
            api.extend([
                ("engine_queued", {"api_call_id": aid, "external_req_id": "external:" + aid,
                                   "engine_request_id": engine}),
                ("api_response", {"api_call_id": aid, "response_id": response}),
            ])
            client.append(("logical_result", {
                **{key: request[key] for key in
                   ("logical_id", "scenario", "ordinal", "request_input_sha256")},
                "response_id": response, "passed": True,
                "checks": {key: True for key in consumer.CHECKS},
                # Digest of one synthetic returned row, distinct from file bytes.
                "original_result_sha256": hashlib.sha256(json.dumps(
                    {"synthetic_returned_logical_id": request["logical_id"]},
                    sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                    allow_nan=False).encode()).hexdigest(),
                "original_error": None,
            }))
        for scenario in consumer.SCENARIOS:
            client.append(("scenario_result", {
                "scenario": scenario,
                "expected_logical_ids": [r["logical_id"] for r in route["requests"]
                                         if r["scenario"] == scenario],
                "output_fingerprint": {"path": "/synthetic/" + route["route_id"] + "/" + scenario,
                                       "bytes": 123, "sha256": "b" * 64},
            }))
        self._stream(route, "api", api)
        for module, scope in consumer.CLIENT_SCOPES.items():
            scoped_records = [(kind, data) for kind, data in client if data["scenario"] in scope]
            self._stream(route, "client", scoped_records, client_module=module)
        for rank in range(4):
            records = []
            if route["mode"] == "graph":
                tokens = {self._ids(r)[2]: 1 for r in route["requests"]}
                graph = {"capture_id": "capture:" + str(rank), "graph_generation": 1,
                         "graph_id": "graph:" + str(rank), "wrapper_uid": "wrapper:" + str(rank),
                         "batch": {"num_tokens": 26, "num_reqs": 26, "uniform": False,
                                   "has_lora": False, "num_active_loras": 0}, "rank": rank}
                # Warmup capture legitimately has no scheduler/execute join.
                records = [
                    ("capture_success", {**graph, "engine_request_ids": []}),
                    ("scheduler_begin", {"execute_call_id": "execute:" + str(rank),
                                         "request_tokens": tokens}),
                    ("replay_success", {**graph, "engine_request_ids": list(tokens),
                                        "execute_call_id": "execute:" + str(rank)}),
                ]
            self._stream(route, "worker", records, rank)

    def _run(self, paths=None):
        return consumer.summarize(self.config_path, list(self.events) if paths is None else paths)

    def _partial(self, rid="m27-graph", paths=None):
        if paths is None:
            paths = [path for path, rows in self.events.items() if rows[0]["route_id"] == rid]
        return consumer.summarize(self.config_path, paths, partial_route=rid)

    def _assert_route_only(self, report):
        self.assertEqual(report["scope"], "route_only")
        for field in ("observer_complete", "graph_observation_complete", "full_matrix_complete",
                      "full_matrix_observer_complete", "full_matrix_graph_observation_complete"):
            self.assertIs(report[field], False)
        self.assertTrue(all(value is False for value in report["acceptance"].values()))

    def _mutate(self, role, kind, change, rank=None, rid="m27-graph"):
        path = self.streams[rid, role, rank]
        event = next(e for e in self.events[path] if e["kind"] == kind)
        change(event)
        self._save(path)

    def _rejected(self, report=None):
        report = self._run() if report is None else report
        self.assertFalse(report["observer_complete"])
        self.assertFalse(report["graph_observation_complete"])
        self.assertTrue(report["errors"])
        self.assertTrue(all(value is False for value in report["acceptance"].values()))
        return report

    def test_complete_synthetic_joins_never_grant_acceptance(self):
        report = self._run()
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["observer_complete"])
        self.assertTrue(report["graph_observation_complete"])
        self.assertEqual(report["counts"]["logical_results"], 104)
        self.assertEqual(report["counts"]["scenario_results"], 20)
        self.assertEqual(report["counts"]["graph_request_joins"], 52)
        self.assertEqual(report["counts"]["graph_request_rank_joins"], 208)
        self.assertEqual(report["counts"]["processes"], 32)
        self.assertEqual(report["graph_ranks_complete"], [0, 1, 2, 3])
        for join in report["graph_request_joins"]:
            self.assertEqual([row["rank"] for row in join["rank_evidence"]], [0, 1, 2, 3])
        self.assertTrue(all(value is False for value in report["acceptance"].values()))
        self.assertEqual(len(report["scenario_output_fingerprint_claims"]), 20)
        self.assertEqual(len(report["logical_response_engine_joins"]), 104)
        self.assertEqual(set(report["process_identities"]),
                         {route["route_id"] for route in self.config["routes"]})
        self.assertTrue(all(len(rows) == 8 for rows in report["process_identities"].values()))
        for rows in report["process_identities"].values():
            self.assertEqual({row["loaded_test_modules"][0] for row in rows if row["role"] == "client"},
                             set(consumer.CLIENT_SCOPES))
        self.assertEqual(len(report["input_fingerprints"]), 33)
        for fp in report["input_fingerprints"]:
            raw = Path(fp["path"]).read_bytes()
            self.assertEqual(fp["bytes"], len(raw))
            self.assertEqual(fp["sha256"], hashlib.sha256(raw).hexdigest())

    def test_no_events_cannot_pass(self):
        self._rejected(self._run([]))

    def test_partial_graph_route_closes_only_twenty_six_requests(self):
        report = self._partial()
        self.assertEqual(report["errors"], [])
        self._assert_route_only(report)
        self.assertIs(report["route_observer_complete"], True)
        self.assertIs(report["route_graph_observation_complete"], True)
        self.assertEqual(report["route_graph_observation_applicability"], "applicable")
        self.assertEqual(report["counts"]["manifest_routes"], 4)
        self.assertEqual(report["counts"]["manifest_logical_results"], 104)
        self.assertEqual(report["counts"]["logical_results"], 26)
        self.assertEqual(report["counts"]["scenario_results"], 5)
        self.assertEqual(report["counts"]["graph_request_joins"], 26)
        self.assertEqual(report["counts"]["graph_request_rank_joins"], 104)
        self.assertEqual(report["graph_ranks_complete"], [0, 1, 2, 3])
        self.assertEqual(report["counts"]["processes"], 8)
        self.assertEqual(len(report["input_fingerprints"]), 9)
        self.assertEqual(set(report["process_identities"]), {"m27-graph"})
        self.assertEqual(len(report["logical_response_engine_joins"]), 26)
        self.assertEqual(len(report["scenario_output_fingerprint_claims"]), 5)

    def test_partial_eager_route_graph_observation_is_not_applicable(self):
        report = self._partial("m35-eager")
        self.assertEqual(report["errors"], [])
        self._assert_route_only(report)
        self.assertIs(report["route_observer_complete"], True)
        self.assertIs(report["route_graph_observation_complete"], False)
        self.assertEqual(report["route_graph_observation_applicability"], "not_applicable")
        self.assertEqual(report["counts"]["logical_results"], 26)
        self.assertEqual(report["counts"]["graph_request_joins"], 0)
        self.assertEqual(report["counts"]["graph_request_rank_joins"], 0)
        self.assertEqual(report["graph_ranks_complete"], [])

    def test_partial_route_still_requires_original_full_manifest(self):
        self.config["routes"] = [self.config["routes"][1]]
        self._save_config()
        report = self._partial()
        self._assert_route_only(report)
        self.assertIs(report["route_observer_complete"], False)
        self.assertEqual(report["errors"][0]["stage"], "config")

    def test_partial_route_rejects_any_surplus_other_route_trace(self):
        selected = [p for p, rows in self.events.items() if rows[0]["route_id"] == "m27-graph"]
        selected.append(self.streams["m35-graph", "worker", 0])
        report = self._partial(paths=selected)
        self._assert_route_only(report)
        self.assertIs(report["route_observer_complete"], False)
        self.assertEqual(report["errors"][0]["stage"], "events")
        self.assertIn("unknown route", report["errors"][0]["message"])

    def test_partial_route_requires_selected_events_and_known_single_route(self):
        for rid, paths in (("m27-graph", []), ("absent", []), (["m27-graph", "m35-graph"], [])):
            report = consumer.summarize(self.config_path, paths, partial_route=rid)
            self._assert_route_only(report)
            self.assertIs(report["route_observer_complete"], False)
            self.assertTrue(report["errors"])

    def test_partial_route_rejects_full_route_selection_argument(self):
        report = consumer.summarize(self.config_path, [], selected_routes=["m27-graph"],
                                    partial_route="m27-graph")
        self._assert_route_only(report)
        self.assertIs(report["route_observer_complete"], False)
        self.assertTrue(report["errors"])

    def test_partial_route_requires_api_and_each_client_telemetry_end(self):
        keys = [("m27-graph", "api", None)] + [
            ("m27-graph", "client", module) for module in consumer.CLIENT_SCOPES]
        for key in keys:
            path = self.streams[key]
            end = self.events[path].pop()
            self._save(path)
            report = self._partial()
            self._assert_route_only(report)
            self.assertIs(report["route_observer_complete"], False)
            self.assertEqual(report["errors"][0]["stage"], "processes")
            self.events[path].append(end)
            self._save(path)

    def test_partial_graph_route_rejects_missing_rank_or_replay_join(self):
        excluded = self.streams["m27-graph", "worker", 3]
        selected = [p for p, rows in self.events.items()
                    if rows[0]["route_id"] == "m27-graph" and p != excluded]
        report = self._partial(paths=selected)
        self.assertIs(report["route_observer_complete"], False)
        self.assertTrue(report["errors"])
        self._mutate("worker", "replay_success", lambda e: e["data"].update(graph_generation=99), 3)
        report = self._partial()
        self.assertIs(report["route_graph_observation_complete"], False)
        self.assertTrue(report["errors"])

    def test_partial_route_keeps_quality_and_source_requirements(self):
        self._mutate("client", "logical_result", lambda e: e["data"]["checks"].update(non_empty=False))
        self.assertIs(self._partial()["route_observer_complete"], False)
        self._mutate("client", "logical_result", lambda e: e["data"]["checks"].update(non_empty=True))
        self._mutate("worker", "activation", lambda e: e["data"].update(sources_verified=False), 0)
        self.assertIs(self._partial()["route_observer_complete"], False)

    def test_partial_cli_exits_zero_without_matrix_claims(self):
        selected = self.root / "selected-traces"
        selected.mkdir()
        for path, rows in self.events.items():
            if rows[0]["route_id"] == "m27-graph":
                (selected / path.name).write_bytes(path.read_bytes())
        output = self.root / "route-summary.json"
        args = ["--expected-manifest", str(self.config_path), "--trace-dir", str(selected),
                "--partial-route", "m27-graph", "--output", str(output)]
        self.assertEqual(consumer.main(args), 0)
        report = json.loads(output.read_bytes())
        self._assert_route_only(report)
        self.assertIs(report["route_observer_complete"], True)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(consumer.main(args), 2)
        for bad_args in (args + ["--route", "m27-graph"],
                         args + ["--partial-route", "m27-graph"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ex:
                consumer.main(bad_args)
            self.assertEqual(ex.exception.code, 2)

    def test_full_matrix_default_completeness_remains_unchanged(self):
        report = self._run()
        self.assertEqual(report["scope"], "full_matrix")
        self.assertIs(report["observer_complete"], True)
        self.assertIs(report["graph_observation_complete"], True)
        self.assertIs(report["full_matrix_complete"], True)
        self.assertIs(report["full_matrix_observer_complete"], True)
        self.assertIs(report["full_matrix_graph_observation_complete"], True)
        self.assertIs(report["route_observer_complete"], False)
        self.assertEqual(report["counts"]["logical_results"], 104)
        self.assertEqual(report["counts"]["scenario_results"], 20)
        self.assertEqual(report["counts"]["graph_request_joins"], 52)
        self.assertEqual(report["counts"]["graph_request_rank_joins"], 208)

    def test_gate_kit_requires_exact_nineteen_original_names(self):
        original = copy.deepcopy(self.config["gate_kit_files"])
        for changed in ({name: fp for name, fp in original.items() if name != "images/image_08.png"},
                        dict(original, unexpected=next(iter(original.values())))):
            self.config["gate_kit_files"] = changed
            self._save_config()
            report = self._rejected()
            self.assertEqual(report["errors"][0]["stage"], "config")

    def test_gate_kit_fingerprint_fields_are_strict(self):
        self.config["gate_kit_files"]["README.md"]["bytes"] = True
        self._save_config()
        self._rejected()

    def test_gate_kit_member_and_total_caps_are_bound(self):
        self.config["gate_kit_files"]["README.md"]["bytes"] = 1024**2 + 1
        self._save_config()
        self._rejected()
        for fp in self.config["gate_kit_files"].values():
            fp["bytes"] = 1024**2
        self._save_config()
        report = self._rejected()
        self.assertIn("total byte cap", report["errors"][0]["message"])

    def test_gate_module_sources_must_equal_bound_kit(self):
        self.config["sources"]["test_image"]["sha256"] = "c" * 64
        self._save_config()
        report = self._rejected()
        self.assertIn("differs from kit", report["errors"][0]["message"])

    def test_client_required_source_profile_cannot_omit_unloaded_gate_file(self):
        self.config["required_role_sources"]["client"].remove("test_mix_text_image")
        self._save_config()
        self._rejected()

    def test_each_client_acknowledges_entire_gate_kit(self):
        self._mutate("client", "activation",
                     lambda e: e["data"]["gate_kit_files"]["images/image_08.png"].update(sha256="c" * 64),
                     "test_image")
        report = self._rejected()
        self.assertIn("Gate kit fingerprints mismatch", report["errors"][0]["message"])

    def test_collected_test_module_is_exact_singleton(self):
        for loaded in ([], list(consumer.CLIENT_SCOPES), ["case_utils"], ["test_image", "test_image"]):
            self._mutate("client", "activation", lambda e: e["data"].update(loaded_test_modules=loaded),
                         "test_image")
            report = self._rejected()
            self.assertIn("exactly one", report["errors"][0]["message"])

    def test_missing_original_pytest_client_cannot_pass(self):
        excluded = self.streams["m27-graph", "client", "test_image"]
        report = self._rejected(self._run([p for p in self.events if p != excluded]))
        self.assertEqual(report["errors"][0]["stage"], "processes")

    def test_duplicate_client_scope_cannot_replace_other_original_test(self):
        self._mutate("client", "activation", lambda e: e["data"].update(loaded_test_modules=["test_text"]),
                     "test_image")
        report = self._rejected()
        self.assertIn("duplicate client module scope", report["errors"][0]["message"])

    def test_logical_result_must_belong_to_actual_collected_test_scope(self):
        foreign = self.config["routes"][1]["requests"][0]["logical_id"]
        self._mutate("client", "logical_result", lambda e: e["data"].update(logical_id=foreign),
                     "test_image")
        report = self._rejected()
        self.assertIn("escaped collected test scope", report["errors"][0]["message"])

    def test_scenario_result_must_belong_to_actual_collected_test_scope(self):
        self._mutate("client", "scenario_result", lambda e: e["data"].update(scenario="text_single"),
                     "test_image")
        report = self._rejected()
        self.assertIn("escaped collected test scope", report["errors"][0]["message"])

    def test_sdk_send_diagnostic_is_also_bound_to_collected_test_scope(self):
        foreign = self.config["routes"][1]["requests"][0]["logical_id"]
        self._insert("client", "observed_send_call", {"logical_id": foreign}, "test_image")
        report = self._rejected()
        self.assertIn("escaped collected test scope", report["errors"][0]["message"])

    def test_each_client_scope_must_cover_its_expected_logicals_exactly_once(self):
        path = self.streams["m27-graph", "client", "test_text"]
        rows = [e for e in self.events[path] if e["kind"] == "logical_result"]
        rows[1]["data"]["logical_id"] = rows[0]["data"]["logical_id"]
        self._save(path)
        report = self._rejected()
        self.assertIn("incomplete or duplicated", report["errors"][0]["message"])

    def test_missing_one_worker_rank_cannot_pass(self):
        excluded = self.streams["m27-graph", "worker", 3]
        self._rejected(self._run([p for p in self.events if p != excluded]))

    def test_source_acknowledgment_is_required(self):
        self._mutate("worker", "activation", lambda e: e["data"].update(sources_verified=False), 0)
        self._rejected()

    def test_source_digest_mismatch(self):
        self._mutate("api", "api_response", lambda e: e.update(observer_source_sha256="c" * 64))
        self._rejected()

    def test_binding_id_requires_digest(self):
        self.config["source_binding_id"] = "an-unmeasured-label"
        self._save_config()
        self._rejected()

    def test_normal_diagnostic_events_do_not_replace_graph_proof(self):
        lid = self.config["routes"][1]["requests"][0]["logical_id"]
        self._insert("api", "generate_call", {"diagnostic": True})
        for kind in ("scheduler_return", "graph_call", "native_capture_unassociated",
                     "native_replay_unassociated"):
            self._insert("worker", kind, {"diagnostic": True}, 0)
        self._insert("client", "observed_send_call", {"logical_id": lid, "send_ordinal": 0})
        report = self._run()
        self.assertTrue(report["observer_complete"], report["errors"])
        self.assertEqual(report["counts"]["graph_request_rank_joins"], 208)
        self.assertEqual(report["kind_counts"]["observed_send_call"], 1)
        path = self.streams["m27-graph", "worker", 0]
        replay = next(e for e in self.events[path] if e["kind"] == "replay_success")
        for diagnostic in ("native_capture_unassociated", "native_replay_unassociated"):
            replay.update(kind=diagnostic, data={})
            self._save(path)
            self._rejected()

    def test_diagnostic_event_role_and_error_status_are_enforced(self):
        self._insert("api", "observed_send_call", {"logical_id": "unused"})
        self._rejected()
        path = self.streams["m27-graph", "api", None]
        self.events[path][-2].update(kind="generate_call", status="error")
        self._save(path)
        self._rejected()

    def test_namespace_and_monotonic_identity_are_enforced(self):
        self._mutate("worker", "replay_success", lambda e: e.update(pid_namespace="pid:[20000]"), 0)
        self._rejected()
        self._mutate("worker", "replay_success", lambda e: e.update(pid_namespace="pid:[10000]", monotonic_ns=0), 0)
        self._rejected()

    def test_nullable_batch_request_count_preserves_exact_shape(self):
        for kind in ("capture_success", "replay_success"):
            self._mutate("worker", kind, lambda e: e["data"]["batch"].update(num_reqs=None), 0)
        report = self._run()
        self.assertTrue(report["observer_complete"], report["errors"])
        self._mutate("worker", "replay_success", lambda e: e["data"]["batch"].update(num_reqs=False), 0)
        self._rejected()

    def test_empty_scheduler_is_diagnostic_but_not_replay_proof(self):
        self._insert("worker", "scheduler_begin", {"execute_call_id": "empty", "request_tokens": {}}, 0)
        self.assertTrue(self._run()["observer_complete"])
        self._mutate("worker", "replay_success", lambda e: e["data"].update(execute_call_id="empty"), 0)
        self._rejected()

    def test_declared_observer_members_bind_canonical_digest(self):
        self.config["observer_sources"]["trace.py"]["sha256"] = "b" * 64
        self._save_config()
        self._rejected()

    def test_activation_sources_must_match_exact_role_fingerprints(self):
        self._mutate("worker", "activation", lambda e: e["data"]["sources"]["worker-source"].update(sha256="b" * 64), 0)
        self._rejected()

    def test_activation_cannot_omit_or_add_hooks(self):
        self._mutate("worker", "activation", lambda e: e["data"].update(hooks=["different-hook"]), 0)
        self._rejected()

    def test_original_request_ids_are_unique_within_scenario(self):
        requests = self.config["routes"][0]["requests"]
        requests[2]["original_request_id"] = requests[1]["original_request_id"]
        self._save_config()
        self._rejected()

    def test_sequence_gap_and_raw_order_are_rejected(self):
        path = self.streams["m27-graph", "worker", 0]
        self.events[path][2]["seq"] = 99
        self._save(path)
        self._rejected()
        self.events[path][2]["seq"] = 3
        self.events[path][1], self.events[path][2] = self.events[path][2], self.events[path][1]
        self._save(path)
        self._rejected()

    def test_duplicate_event_id(self):
        path = self.streams["m27-graph", "worker", 0]
        self.events[path][2]["event_id"] = self.events[path][1]["event_id"]
        self._save(path)
        self._rejected()

    def test_missing_telemetry_end(self):
        path = self.streams["m27-graph", "worker", 0]
        self.events[path].pop()
        self._save(path)
        self._rejected()

    def test_telemetry_error_and_cap_are_rejected(self):
        self._mutate("worker", "telemetry_end", lambda e: e["data"].update(errors=["synthetic error"]), 0)
        self._rejected()
        self._mutate("worker", "telemetry_end", lambda e: e["data"].update(errors=[], cap_exceeded=True), 0)
        self._rejected()

    def test_unknown_kind_or_status_cannot_be_ignored(self):
        path = self.streams["m27-graph", "worker", 0]
        self.events[path][1]["kind"] = "UNSUPPORTED"
        self._save(path)
        self._rejected()
        self.events[path][1]["kind"] = "capture_success"
        self.events[path][1]["status"] = "UNKNOWN"
        self._save(path)
        self._rejected()

    def test_generation_must_match_actual_capture(self):
        self._mutate("worker", "replay_success", lambda e: e["data"].update(graph_generation=2), 0)
        self._rejected()

    def test_wrapper_identity_and_batch_must_match(self):
        self._mutate("worker", "replay_success", lambda e: e["data"].update(wrapper_uid="different"), 0)
        self._rejected()
        self._mutate("worker", "replay_success", lambda e: e["data"].update(wrapper_uid="wrapper:0"), 0)
        self._mutate("worker", "replay_success", lambda e: e["data"]["batch"].update(num_tokens=27), 0)
        self._rejected()

    def test_replay_requires_same_process_scheduler(self):
        self._mutate("worker", "replay_success", lambda e: e["data"].update(execute_call_id="execute:1"), 0)
        self._rejected()

    def test_scheduler_request_ids_must_equal_replay_ids(self):
        self._mutate("worker", "replay_success", lambda e: e["data"]["engine_request_ids"].pop(), 0)
        self._rejected()

    def test_unjoined_response_cannot_use_client_send_as_substitute(self):
        self._mutate("client", "logical_result", lambda e: e["data"].update(response_id="unknown-response"))
        self._rejected()

    def test_logical_response_and_api_call_joins_must_be_bijections(self):
        path = self.streams["m27-graph", "client", None]
        logicals = [e for e in self.events[path] if e["kind"] == "logical_result"]
        logicals[1]["data"]["response_id"] = logicals[0]["data"]["response_id"]
        self._save(path)
        report = self._rejected()
        self.assertIn("exactly once", report["errors"][0]["message"])
        logicals[1]["data"]["response_id"] = self._ids(self.config["routes"][1]["requests"][1])[1]
        self._save(path)
        path = self.streams["m27-graph", "api", None]
        responses = [e for e in self.events[path] if e["kind"] == "api_response"]
        responses[1]["data"]["api_call_id"] = responses[0]["data"]["api_call_id"]
        self._save(path)
        report = self._rejected()
        self.assertIn("exactly once", report["errors"][0]["message"])

    def test_config_ordinals_are_zero_based_per_route(self):
        self.config["routes"][0]["requests"][0]["ordinal"] = 26
        self._save_config()
        self._rejected()

    def test_input_hash_is_bound_to_config(self):
        self._mutate("client", "logical_result", lambda e: e["data"].update(request_input_sha256="c" * 64))
        self._rejected()

    def test_original_returned_row_digest_is_not_whole_scenario_file_digest(self):
        report = self._run()
        self.assertTrue(report["observer_complete"], report["errors"])
        files = {(row["route_id"], row["scenario"]): row["output_fingerprint"]["sha256"]
                 for row in report["scenario_output_fingerprint_claims"]}
        self.assertTrue(all(row["original_result_sha256"] != files[row["route_id"], row["scenario"]]
                            for row in report["logical_response_engine_joins"]))

    def test_original_returned_row_digest_and_error_claim_types(self):
        self._mutate("client", "logical_result", lambda e: e["data"].update(original_result_sha256="bad"))
        self._rejected()
        self._mutate("client", "logical_result", lambda e: e["data"].update(original_result_sha256="c" * 64,
                                                                                original_error=""))
        self._rejected()

    def test_checks_require_exact_names_boolean_true_and_passed(self):
        self._mutate("client", "logical_result", lambda e: e["data"]["checks"].update(non_empty=1))
        self._rejected()
        self._mutate("client", "logical_result", lambda e: e["data"]["checks"].update(non_empty=False))
        self._rejected()
        self._mutate("client", "logical_result", lambda e: e["data"]["checks"].update(non_empty=True, extra=True))
        self._rejected()

    def test_scenario_complete_ids_and_fingerprint_required(self):
        self._mutate("client", "scenario_result", lambda e: e["data"].update(expected_logical_ids=[]))
        self._rejected()

    def test_config_matrix_and_typed_ranks(self):
        self.config["routes"][0]["expected_ranks"] = [False, 1, 2, 3]
        self._save_config()
        self._rejected()

    def test_duplicate_json_keys_are_rejected(self):
        path = self.streams["m27-graph", "worker", 0]
        raw = path.read_bytes().replace(b'"schema_version": 1,', b'"schema_version": 1, "schema_version": 1,', 1)
        path.write_bytes(raw)
        self._rejected()

    def test_nonfinite_constants_and_overflow_are_rejected(self):
        path = self.streams["m27-graph", "worker", 0]
        self.events[path][0]["data"]["overflow"] = "TOKEN"
        self._save(path)
        raw = path.read_bytes()
        for number in (b"NaN", b"Infinity", b"1e999"):
            path.write_bytes(raw.replace(b'"overflow": "TOKEN"', b'"overflow": ' + number, 1))
            self._rejected()

    def test_raw_newline_blank_line_and_missing_eof_are_rejected(self):
        path = self.streams["m27-graph", "worker", 0]
        raw = path.read_bytes()
        for corrupted in (raw[:-1], raw + b"\n", raw.replace(b"synthetic-run", b"synthetic\nrun", 1)):
            path.write_bytes(corrupted)
            self._rejected()

    def test_input_file_total_and_file_count_caps(self):
        with mock.patch.object(consumer, "MAX_FILE_BYTES", 100):
            self._rejected()
        with mock.patch.object(consumer, "MAX_TOTAL_BYTES", self.config_path.stat().st_size + 1):
            self._rejected()
        self._rejected(self._run([next(iter(self.events))] * 129))

    def test_noncanonical_and_nonregular_inputs_are_rejected(self):
        directory = self.root / "nested"
        directory.mkdir()
        self._rejected(consumer.summarize(directory / ".." / "config.json", list(self.events)))
        self._rejected(consumer.summarize(directory, list(self.events)))

    def test_input_mutation_during_read_is_rejected(self):
        path = self.root / "mutable.json"
        path.write_bytes(b"{}\n")
        original_read = consumer.os.read
        mutated = False

        def mutate_after_read(fd, amount):
            nonlocal mutated
            raw = original_read(fd, amount)
            if not mutated:
                mutated = True
                path.write_bytes(b"{\"changed\": true}\n")
            return raw

        with mock.patch.object(consumer.os, "read", side_effect=mutate_after_read):
            with self.assertRaisesRegex(consumer.ContractError, "input changed during read"):
                consumer._read(path, [0])

    def test_cli_output_is_exclusive_and_has_consumer_only_exit(self):
        output = self.root / "observer-summary.json"
        argv = ["--config", str(self.config_path), "--events", *map(str, self.events), "--output", str(output)]
        self.assertEqual(consumer.main(argv), 0)
        original = output.read_bytes()
        self.assertFalse(json.loads(original)["acceptance"]["Gate"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(consumer.main(argv), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_trace_directory_cli_requires_all_routes_and_is_not_recursive(self):
        nested = self.root / "ignored-subdirectory"
        nested.mkdir()
        (nested / "invalid.jsonl").write_bytes(b"not a trace\n")
        output = self.root / "observer-summary.json"
        base = ["--expected-manifest", str(self.config_path), "--trace-dir", str(self.root)]
        routes = [r["route_id"] for r in self.config["routes"]]
        full = base + [item for rid in routes for item in ("--route", rid)]
        self.assertEqual(consumer.main(full + ["--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_bytes())["counts"]["event_files"], 32)
        for index, selected in enumerate((routes[:3], routes[:3] + [routes[0]])):
            args = base + [item for rid in selected for item in ("--route", rid)]
            destination = self.root / ("rejected-" + str(index) + ".json")
            self.assertEqual(consumer.main(args + ["--output", str(destination)]), 1)
            self._rejected(json.loads(destination.read_bytes()))

    def test_trace_directory_rejects_nonregular_jsonl(self):
        (self.root / "not-a-file.jsonl").mkdir()
        with self.assertRaisesRegex(consumer.ContractError, "nonregular"):
            consumer._trace_paths(self.root)


if __name__ == "__main__":
    unittest.main()

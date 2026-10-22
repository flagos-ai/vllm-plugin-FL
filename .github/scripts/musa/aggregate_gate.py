"""Consume original Gate outputs and independent graph-observer evidence."""

import argparse
import base64
import csv
import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import (
    CHECKS,
    GATE_FILES,
    ROUTES,
    SCENARIOS,
    TEST_FILES,
    bound_json,
    fingerprint,
    fingerprint_streaming,
    read_bytes,
    read_json,
    require,
    write_json,
)


def original_scenario(report, scenario):
    expected = SCENARIOS[scenario]
    require(
        report["schema_version"] == 1 and report["case"]["scenario"] == scenario,
        "Wrong original scenario",
    )
    summary = report["summary"]
    require(
        type(summary["request_count"]) is int and summary["request_count"] == expected,
        "Logical request count changed",
    )
    require(
        type(summary["passed_count"]) is int
        and type(summary["failed_count"]) is int
        and summary["passed_count"] == expected
        and summary["failed_count"] == 0
        and summary["passed"] is True,
        "Original scenario reported failures",
    )
    require(
        type(report["output"]) is list
        and len(report["output"]) == expected
        and type(report["input"]) is list
        and len(report["input"]) == expected
        and type(report["case"]["concurrency"]) is int
        and report["case"]["concurrency"] == expected,
        "Missing original logical outputs",
    )
    ids = []
    response_ids = []
    for request, output in zip(report["input"], report["output"]):
        require(
            output["passed"] is True and output.get("error") in (None, ""),
            "Logical output failed",
        )
        require(
            set(output["checks"]) == CHECKS
            and all(value is True for value in output["checks"].values()),
            "The ten original quality checks did not all pass",
        )
        require(
            type(output["request_id"]) is str and output["request_id"],
            "Missing original logical ID",
        )
        require(
            request["request_id"] == output["request_id"],
            "Original input/output logical order changed",
        )
        ids.append(output["request_id"])
        response = output["response"]
        require(
            type(response) is dict
            and type(response.get("id")) is str
            and response["id"],
            "Missing SDK response ID",
        )
        response_ids.append(response["id"])
    require(
        len(set(ids)) == expected and len(set(response_ids)) == expected,
        "Duplicated logical or response ID",
    )
    if scenario == "text_single":
        require(
            report["input"][0]["min_length"] == 256
            and report["input"][0]["max_tokens"] == 512,
            "Original long-text request overrides changed",
        )
    return response_ids


def route_business(route, root, binding_id):
    require(route["source_binding_id"] == binding_id, "Route source binding changed")
    preflight = bound_json(route["preflight"], root)
    require(
        preflight["passed"] is True
        and preflight["source_binding_id"] == binding_id
        and preflight["logical_device_count"] == 4,
        "Real-device/source/cache preflight did not pass",
    )
    require(route["ready"]["ready"] is True, "Server was not ready")
    require(route["server"]["started"] is True, "Server was not started")
    require(
        route["cleanup"]["closed"] is True and not route["cleanup"]["errors"],
        "Owned server cleanup is incomplete",
    )
    require(
        [item["file"] for item in route["pytest"]] == list(TEST_FILES),
        "Not all three original pytest entries were attempted",
    )
    for item in route["pytest"]:
        require(
            item["attempted"] is True
            and item["actual_exit"] == 0
            and item["timed_out"] is False
            and not item["errors"],
            f"Original pytest entry failed: {item['file']}",
        )
        for key in ("stdout", "stderr", "junit"):
            ref = item[key]
            require(
                fingerprint(ref["path"], 64 * 1024 * 1024, root) == ref,
                f"Changed pytest {key}",
            )
        junit_raw = read_bytes(item["junit"]["path"], 16 * 1024 * 1024, root)
        require(
            b"<!DOCTYPE" not in junit_raw and b"<!ENTITY" not in junit_raw,
            "JUnit external/entity content is unsupported",
        )
        junit = ET.fromstring(junit_raw)
        cases = list(junit.iter("testcase"))
        expected_cases = 1 if item["file"] == "test_mix_text_image.py" else 2
        require(len(cases) == expected_cases, "Original pytest case collection changed")
        require(
            all(
                not list(case)
                or all(child.tag in ("system-out", "system-err") for child in case)
                for case in cases
            ),
            "Original pytest has a failure, error, skip or unknown result child",
        )
    for key in ("stdout", "stderr"):
        ref = route["cleanup"][key]
        require(
            fingerprint(ref["path"], 64 * 1024 * 1024, root) == ref,
            "Server raw output changed",
        )
    require(set(route["scenarios"]) == set(SCENARIOS), "Missing original JSON output")
    response_ids = {}
    for scenario, ref in route["scenarios"].items():
        report = bound_json(ref, root)
        require(
            report["case"]["served_model_name"] == route["served_model_name"]
            and report["case"]["model_path"] == route["model_path"],
            "Wrong model result",
        )
        require(str(report["case"]["port"]) == str(route["port"]), "Wrong route port")
        response_ids[scenario] = original_scenario(report, scenario)
    require(not route["errors"], "Route orchestration errors")
    return response_ids


# Match the bounded raw-input budgets of the companion observer consumer.
OBSERVER_RAW_MAX_FILE_BYTES = 2 * 1024**3
OBSERVER_RAW_MAX_TOTAL_BYTES = 12 * 1024**3


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def observer_route(observer, config, route, response_ids, root):
    rid = route["route_id"]
    expected_route = next(item for item in config["routes"] if item["route_id"] == rid)
    require(
        expected_route["model"] == route["model"]
        and expected_route["mode"] == route["mode"],
        "Observer model/mode changed",
    )
    claimed_outputs = [
        item
        for item in observer["scenario_output_fingerprint_claims"]
        if item["route_id"] == rid
    ]
    actual_outputs = {
        item["scenario"]: item["output_fingerprint"] for item in claimed_outputs
    }
    require(
        len(claimed_outputs) == 5
        and len(actual_outputs) == 5
        and actual_outputs == route["scenarios"],
        "Observer is not bound to the original result bytes",
    )
    attestations = bound_json(route["process_attestations"], root)
    require(
        attestations["server_identity"] == route["server"]["identity"],
        "Independent process ownership changed",
    )
    identities = {(p["pid"], p["starttime"]): p for p in attestations["processes"]}
    require(identities, "No independent process attestations")
    require(not attestations["errors"], "Independent process sampling had errors")
    process_keys = set()
    worker_ranks = set()
    api_count = 0
    client_entries = set()
    processes = observer["process_identities"][rid]
    worker_processes = {}
    for process in processes:
        ticks = process["starttime"]
        require(
            type(ticks) is str
            and ticks.isascii()
            and ticks.isdigit()
            and str(int(ticks)) == ticks,
            "Noncanonical observer starttime",
        )
        identity = identities.get((process["pid"], int(ticks)))
        require(
            identity is not None and identity["lifetime_stable"] is True,
            "Observer process has no independent PID/starttime ownership",
        )
        require(
            identity["namespaces"]["pid"] == process["pid_namespace"],
            "Observer PID namespace differs from independent process attestation",
        )
        if process["role"] == "client":
            require(
                identity.get("owner") == "pytest"
                and identity["entrypoint"] in TEST_FILES,
                "Observer client is not an original spawned pytest entry",
            )
            actual_entry = next(
                item
                for item in route["pytest"]
                if item["file"] == identity["entrypoint"]
            )
            require(
                actual_entry["identity"]["pid"] == process["pid"]
                and actual_entry["identity"]["starttime"] == int(ticks)
                and actual_entry["identity"]["sid"] == identity["sid"],
                "Observer client differs from its independent process lease",
            )
            require(
                process["loaded_test_modules"] == [Path(identity["entrypoint"]).stem],
                "Observer client test-module scope differs from original subprocess",
            )
            require(
                identity["entrypoint"] not in client_entries,
                "Duplicate observer pytest client",
            )
            client_entries.add(identity["entrypoint"])
        else:
            require(
                process["role"] in ("api", "worker")
                and identity["sid"] == route["server"]["identity"]["sid"],
                "Observer process is outside the owned server session",
            )
        require(
            process["process_key"] not in process_keys, "Duplicate observer process"
        )
        process_keys.add(process["process_key"])
        if process["role"] == "api":
            api_count += 1
        if process["role"] == "worker":
            worker_ranks.add(process["rank"])
            worker_processes[process["rank"]] = process
    require(
        api_count == 1
        and worker_ranks == {0, 1, 2, 3}
        and client_entries == set(TEST_FILES)
        and len(processes) == 8,
        "Missing API, TP4 worker or three original pytest client observations",
    )
    requests = [
        item
        for item in observer["logical_response_engine_joins"]
        if item["route_id"] == rid
    ]
    require(len(requests) == 26, "Missing observer logical joins")
    expected_requests = {item["ordinal"]: item for item in expected_route["requests"]}
    require(
        len(expected_requests) == 26 and set(expected_requests) == set(range(26)),
        "Observer config original request matrix is incomplete",
    )
    graph_requests = [
        item for item in observer["graph_request_joins"] if item["route_id"] == rid
    ]
    require(
        len(graph_requests) == (26 if route["mode"] == "graph" else 0),
        "Graph/eager route join count changed",
    )
    graph_by_logical = {item["logical_id"]: item for item in graph_requests}
    require(
        len(graph_by_logical) == len(graph_requests), "Duplicate graph logical join"
    )
    seen = set()
    route_ordinals = [
        (scenario, index, response_id)
        for scenario in SCENARIOS
        for index, response_id in enumerate(response_ids[scenario])
    ]
    for request in requests:
        require(
            type(request["ordinal"]) is int and 0 <= request["ordinal"] < 26,
            "Invalid whole-route observer ordinal",
        )
        scenario, local_ordinal, expected_id = route_ordinals[request["ordinal"]]
        key = (scenario, local_ordinal)
        require(key not in seen, "Duplicate observer logical join")
        seen.add(key)
        original = bound_json(route["scenarios"][scenario], root)
        input_row = original["input"][local_ordinal]
        output_row = original["output"][local_ordinal]
        expected = expected_requests[request["ordinal"]]
        require(
            request["scenario"] == scenario
            and request["response_id"] == expected_id
            and request["passed"] is True
            and request["original_error"] is None
            and request["logical_id"] == expected["logical_id"]
            and request["original_request_id"]
            == input_row["request_id"]
            == expected["original_request_id"]
            and request["request_input_sha256"]
            == canonical_sha(input_row)
            == expected["request_input_sha256"]
            and request["original_result_sha256"] == canonical_sha(output_row)
            and request["checks"] == output_row["checks"],
            "Observer response/original-quality join failed",
        )
        if route["mode"] == "graph":
            graph = graph_by_logical[request["logical_id"]]
            require(
                all(
                    graph[key] == request[key]
                    for key in ("response_id", "api_call_id", "engine_request_id")
                )
                and graph["ranks"] == [0, 1, 2, 3],
                "A graph logical request lacks capture/replay on all TP ranks",
            )
            require(
                len(graph["rank_evidence"]) == 4
                and [row["rank"] for row in graph["rank_evidence"]] == [0, 1, 2, 3],
                "Missing graph rank evidence",
            )
            for evidence in graph["rank_evidence"]:
                worker = worker_processes[evidence["rank"]]
                require(
                    all(
                        evidence[key] == worker[key]
                        for key in ("process_key", "pid", "starttime", "pid_namespace")
                    ),
                    "Graph evidence is outside an independently attested worker",
                )
                require(
                    type(evidence["graph_generation"]) is int
                    and evidence["graph_generation"] >= 1
                    and all(
                        type(evidence[key]) is int and evidence[key] >= 1
                        for key in ("capture_seq", "scheduler_seq", "replay_seq")
                    )
                    and evidence["capture_seq"] < evidence["replay_seq"]
                    and evidence["scheduler_seq"] < evidence["replay_seq"],
                    "Graph generation or event order is invalid",
                )
    require(
        seen == {(s, n) for s, count in SCENARIOS.items() for n in range(count)},
        "Observer logical request coverage is incomplete",
    )


def scenario_projection(route, root):
    result = {}
    for scenario in SCENARIOS:
        ref = route.get("scenarios", {}).get(scenario)
        if ref is None:
            result[scenario] = {"status": "NOT_RUN_OR_MISSING", "original_json": None}
            continue
        try:
            original = bound_json(ref, root)
            failures = [
                {
                    "request_id": row.get("request_id"),
                    "error": row.get("error"),
                    "failed_checks": [
                        name
                        for name, passed in row.get("checks", {}).items()
                        if passed is False
                    ],
                }
                for row in original["output"]
                if row.get("passed") is not True
            ]
            result[scenario] = {
                "status": "ORIGINAL_REPORTED",
                "original_json": ref,
                "original_summary": original["summary"],
                "original_failures": failures,
            }
        except (ValueError, KeyError, TypeError, OSError) as error:
            result[scenario] = {
                "status": "UNKNOWN",
                "original_json": ref,
                "error": f"{type(error).__name__}: {error}",
            }
    return result


def aggregate(manifest_path, observer_path=None):
    root = Path(manifest_path).resolve(strict=True).parent
    manifest = read_json(manifest_path, root=root)
    require(manifest["schema_version"] == 1, "Unknown route manifest schema")
    binding = bound_json(manifest["source_binding"])
    binding_id = manifest["source_binding"]["sha256"]
    require(manifest["source_binding_id"] == binding_id, "Changed source binding")
    require(
        binding["source_commit"] == manifest["source_commit"], "Changed source commit"
    )
    require(
        [r["route_id"] for r in manifest["routes"]] == [r[0] for r in ROUTES],
        "Four-route matrix is incomplete or reordered",
    )
    observer = None
    observer_error = None
    if observer_path is not None:
        try:
            config = bound_json(manifest["observer_config"])
            require(
                config["run_id"] == manifest["run_id"]
                and config["source_binding_id"] == binding_id
                and config["observer_source_sha256"]
                == manifest["observer_source_sha256"],
                "Observer readonly configuration does not bind this run/source",
            )
            require(
                [r["route_id"] for r in config["routes"]] == [r[0] for r in ROUTES],
                "Readonly observer configuration does not cover the original four routes",
            )
            require(
                set(config["gate_kit_files"]) == set(GATE_FILES),
                "Observer original kit count changed",
            )
            for name, ref in config["gate_kit_files"].items():
                require(
                    fingerprint(ref["path"]) == ref,
                    "Observer original Gate source changed",
                )
                expected_source = binding["original_gate"][name]
                require(
                    ref
                    == {
                        key: expected_source[key] for key in ("path", "bytes", "sha256")
                    },
                    "Observer Gate source is not bound to committed original blobs",
                )
            require(config["sources"], "No observer source origins")
            for ref in config["sources"].values():
                require(
                    fingerprint(ref["path"]) == ref,
                    "Observer API/vendor/Gate source changed",
                )
            recorded_paths = set()
            for name, record in config.get("source_records", {}).items():
                require(
                    record["record"] == binding["distributions"][name]["record"],
                    "Observer extra-source RECORD changed from normal-site preflight",
                )
                raw_record = read_bytes(record["record"]["path"], 64 * 1024 * 1024)
                require(
                    fingerprint(record["record"]["path"], 64 * 1024 * 1024)
                    == record["record"],
                    "Observer selected installed RECORD bytes changed",
                )
                rows = list(csv.reader(io.StringIO(raw_record.decode("utf-8"))))
                lookup = {row[0]: row[1:] for row in rows}
                require(len(lookup) == len(rows), "Duplicate installed RECORD member")
                for row in record["selected_source_record_verified"]:
                    ref = {key: row[key] for key in ("path", "bytes", "sha256")}
                    require(
                        fingerprint(ref["path"]) == ref,
                        "Observer extra-source body changed",
                    )
                    resolved = (
                        Path(record["record"]["path"]).parent.parent / row["member"]
                    ).resolve(strict=True)
                    require(
                        str(resolved) == ref["path"],
                        "Observer source/RECORD physical path mismatch",
                    )
                    encoded = (
                        base64.urlsafe_b64encode(bytes.fromhex(ref["sha256"]))
                        .decode()
                        .rstrip("=")
                    )
                    require(
                        lookup[row["member"]]
                        == ["sha256=" + encoded, str(ref["bytes"])],
                        "Observer source/RECORD body fingerprint mismatch",
                    )
                    recorded_paths.add(ref["path"])
            kit_paths = {ref["path"] for ref in config["gate_kit_files"].values()}
            require(
                recorded_paths
                == {ref["path"] for ref in config["sources"].values()} - kit_paths,
                "Observer vendor/upstream source RECORD coverage incomplete",
            )
            package_sources = binding["observer_sources"]
            require(
                package_sources is not None
                and set(package_sources) == set(config["observer_sources"]),
                "Observer installed source/RECORD origin is unbound",
            )
            for name, ref in package_sources.items():
                require(
                    fingerprint(ref["path"]) == ref
                    and config["observer_sources"][name]
                    == {key: ref[key] for key in ("bytes", "sha256")},
                    "Observer installed package body differs from origin binding/config",
                )
            require(
                canonical_sha(config["observer_sources"])
                == manifest["observer_source_sha256"],
                "Observer five-member source digest changed",
            )
            command = bound_json(manifest["observer_command"], root)
            require(
                command["attempted"] is True
                and command["actual_exit"] == 0
                and command["timed_out"] is False
                and not command["errors"]
                and command["cleanup"]["closed"] is True,
                "No successful actual independent observer invocation",
            )
            for key in ("stdout", "stderr"):
                ref = command[key]
                require(
                    fingerprint(ref["path"], 64 * 1024 * 1024, root) == ref,
                    "Observer command raw output changed",
                )
            observer = read_json(observer_path, 64 * 1024 * 1024, root)
            require(
                observer["schema_version"] == 1
                and observer["run_id"] == manifest["run_id"]
                and observer["source_binding_id"] == binding_id
                and observer["observer_source_sha256"]
                == manifest["observer_source_sha256"],
                "Observer source/run binding mismatch",
            )
            require(
                observer["observer_complete"] is True
                and observer["graph_observation_complete"] is True,
                "Observer marked telemetry or graph joins incomplete",
            )
            require(
                observer.get("scope") == "full_matrix"
                and observer.get("full_matrix_complete") is True
                and observer.get("full_matrix_observer_complete") is True
                and observer.get("full_matrix_graph_observation_complete") is True,
                "Partial route observation cannot replace a full Gate matrix",
            )
            require(
                set(observer["process_identities"]) == {r[0] for r in ROUTES},
                "Missing observer process routes",
            )
            require(
                observer["acceptance"]
                == {
                    "source": False,
                    "GPU": False,
                    "ABI": False,
                    "model": False,
                    "Gate": False,
                },
                "Observer attempted to self-certify acceptance",
            )
            for name, value in {
                "logical_results": 104,
                "scenario_results": 20,
                "api_responses": 104,
                "queued_requests": 104,
                "graph_request_joins": 52,
                "graph_request_rank_joins": 208,
            }.items():
                require(
                    observer["counts"][name] == value,
                    "Incomplete observer matrix counts",
                )
            require(
                len(observer["logical_response_engine_joins"]) == 104
                and len(observer["scenario_output_fingerprint_claims"]) == 20
                and len(observer["graph_request_joins"]) == 52
                and observer["graph_ranks_complete"] == [0, 1, 2, 3],
                "Incomplete flat observer joins",
            )
            require(observer.get("errors", []) == [], "Observer errors")
            inputs = observer["input_fingerprints"]
            require(
                len(inputs) == 33 and inputs[0] == manifest["observer_config"],
                "Observer config is not the first actual raw input",
            )
            require(
                len({ref["path"] for ref in inputs}) == len(inputs)
                and sum(ref["bytes"] for ref in inputs) <= OBSERVER_RAW_MAX_TOTAL_BYTES,
                "Observer raw input duplicate or total cap exceeded",
            )
            for ref in inputs[1:]:
                require(
                    fingerprint_streaming(
                        ref["path"], OBSERVER_RAW_MAX_FILE_BYTES, root
                    )
                    == ref,
                    "Observer raw events changed",
                )
        except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as error:
            observer_error = f"{type(error).__name__}: {error}"
            observer = None
    else:
        observer_error = "Independent observer summary is missing"
    results = []
    for index, route in enumerate(manifest["routes"]):
        errors = []
        business_passed = False
        graph_observed = False
        try:
            responses = route_business(route, root, binding_id)
            business_passed = True
            if observer is None:
                errors.append(observer_error)
            else:
                observer_route(observer, config, route, responses, root)
                graph_observed = route["mode"] == "graph"
        except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as error:
            errors.append(f"{type(error).__name__}: {error}")
        results.append(
            {
                "route_id": route["route_id"],
                "model": route["model"],
                "mode": route["mode"],
                "expected_logical_requests": 26,
                "business_passed": business_passed,
                "graph_observed": graph_observed,
                "passed": business_passed and not errors,
                "errors": errors,
                "original_scenarios": scenario_projection(route, root),
                "pytest_actual_exits": [
                    item.get("actual_exit") for item in route.get("pytest", [])
                ],
                "cleanup": route.get("cleanup"),
            }
        )
    passed = all(item["passed"] for item in results) and not manifest["errors"]
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "source_commit": manifest["source_commit"],
        "source_binding_id": binding_id,
        "route_manifest": fingerprint(manifest_path),
        "observer_summary": fingerprint(observer_path, 64 * 1024 * 1024)
        if observer_path is not None and Path(observer_path).is_file()
        else None,
        "routes": results,
        "expected_scenarios": 20,
        "expected_logical_requests": 104,
        "original_checks_per_logical_request": 10,
        "wire_attempt_count": "UNKNOWN: original SDK retry policy is unchanged",
        "gate_passed": passed,
        "errors": manifest["errors"],
        "scope": "Four original model/mode Gate routes; no claim of every vendor ABI",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-manifest", required=True)
    parser.add_argument("--observer-summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate(args.route_manifest, args.observer_summary)
    write_json(args.output, result)
    print(result["gate_passed"])
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

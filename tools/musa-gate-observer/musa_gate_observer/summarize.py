"""Strict stdlib event consumer. Joins never constitute GPU/model acceptance.

This module imports no framework or production package. Input fingerprints are
measured here; source acknowledgments and scenario fingerprints remain claims
from the separately bound producers, never independently verified source proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from collections import Counter, defaultdict

MAX_FILE_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
MAX_EVENT_FILES = 128
CHECKS = frozenset((
    "non_empty", "minimum_length", "expected_semantics", "expected_order",
    "no_bang_triplet", "no_mojibake", "no_control_characters",
    "no_long_character_run", "no_repeated_word_run", "no_repeated_phrase",
))
SCENARIOS = {
    "text_single": 1, "text_concurrent_8": 8, "image_single": 1,
    "image_concurrent_8": 8, "mixed_concurrent_8": 8,
}
CLIENT_SCOPES = {
    "test_text": frozenset(("text_single", "text_concurrent_8")),
    "test_image": frozenset(("image_single", "image_concurrent_8")),
    "test_mix_text_image": frozenset(("mixed_concurrent_8",)),
}
GATE_MODULES = frozenset(("case_utils", *CLIENT_SCOPES))
GATE_MEMBERS = frozenset((
    ".gitignore", "README.md", "case_utils.py", "generate_images.py", "run_serve.sh",
    "run_serve_eager.sh", "run_serve_graph.sh", "run_test.sh",
    "test_text.py", "test_image.py", "test_mix_text_image.py",
    *("images/image_%02d.png" % index for index in range(1, 9)),
))
EVENT_FIELDS = frozenset((
    "schema_version", "run_id", "route_id", "model", "mode", "event_id",
    "pid", "starttime", "pid_namespace", "time_ns", "monotonic_ns",
    "process_key", "seq", "kind", "status",
    "observer_source_sha256", "source_binding_id", "data",
))
KINDS = frozenset((
    "activation", "telemetry_end", "api_response", "engine_queued",
    "scheduler_begin", "capture_success", "replay_success", "logical_result",
    "scenario_result",
    "generate_call", "scheduler_return", "graph_call",
    "native_capture_unassociated", "native_replay_unassociated", "observed_send_call",
))
BATCH_FIELDS = frozenset((
    "num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras",
))
OBSERVER_MEMBERS = frozenset((
    "__init__.py", "trace.py", "server.py", "pytest_plugin.py", "summarize.py",
))


class ContractError(ValueError):
    """Malformed, ambiguous or incomplete observation contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def label(value):
    return (isinstance(value, str) and 0 < len(value) <= 1024
            and all(ord(c) >= 32 for c in value))


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ContractError("nonfinite JSON constant: " + value)


def _finite(value):
    # parse_constant does not cover a finite-looking exponent overflowing float.
    if isinstance(value, float):
        import math
        require(math.isfinite(value), "nonfinite JSON number")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _json(raw):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                       parse_constant=_constant)
    _finite(value)
    return value


def _identity(s, cross_method=False):
    # Windows lstat reports creation time as ctime; CRT fstat can report the
    # modification time there instead. Compare ctime within each method, and
    # compare the remaining stable fields when matching path to open handle.
    result = (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns,
              s.st_nlink, getattr(s, "st_rdev", None))
    if not (os.name == "nt" and cross_method):
        result += (s.st_ctime_ns,)
    return result


def _read(path, budget):
    p = Path(path).absolute()
    require(p.resolve(strict=True) == p, "noncanonical or symlink input: " + str(p))
    before = p.lstat()
    require(stat.S_ISREG(before.st_mode), "nonregular input: " + str(p))
    require(0 <= before.st_size <= MAX_FILE_BYTES, "input file cap exceeded")
    require(budget[0] + before.st_size <= MAX_TOTAL_BYTES, "total input cap exceeded")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(p, flags)
    try:
        opened = os.fstat(fd)
        require(_identity(before, True) == _identity(opened, True), "input changed before open")
        chunks, count = [], 0
        while True:
            chunk = os.read(fd, min(1024**2, MAX_FILE_BYTES + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            require(count <= MAX_FILE_BYTES, "input file cap exceeded during read")
        require(_identity(opened) == _identity(os.fstat(fd)), "input changed during read")
    finally:
        os.close(fd)
    require(_identity(before) == _identity(p.lstat()), "input changed after read")
    require(p.resolve(strict=True) == p, "input path changed after read")
    raw = b"".join(chunks)
    require(len(raw) == before.st_size, "input length changed")
    budget[0] += len(raw)
    return raw, {"path": str(p), "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}


def _config(c):
    require(isinstance(c, dict) and type(c.get("schema_version")) is int
            and c["schema_version"] == 1, "config schema")
    require(label(c.get("run_id")) and digest(c.get("source_binding_id")), "config labels/digest binding")
    require(digest(c.get("observer_source_sha256")), "config observer digest")
    own = c.get("observer_sources")
    require(isinstance(own, dict) and set(own) == OBSERVER_MEMBERS, "five observer source members")
    for fp in own.values():
        require(isinstance(fp, dict) and set(fp) == {"bytes", "sha256"}
                and integer(fp["bytes"], 1) and digest(fp["sha256"]), "observer source fingerprint")
    canonical = json.dumps(own, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    require(hashlib.sha256(canonical).hexdigest() == c["observer_source_sha256"],
            "observer source canonical digest mismatch")
    sources = c.get("sources")
    require(isinstance(sources, dict) and sources and all(label(k) for k in sources), "source table")
    for fp in sources.values():
        require(isinstance(fp, dict) and set(fp) == {"path", "bytes", "sha256"}, "source fingerprint fields")
        _fingerprint_claim(fp)
    kit = c.get("gate_kit_files")
    require(isinstance(kit, dict) and set(kit) == GATE_MEMBERS,
            "exact original nineteen-file Gate kit required")
    for fp in kit.values():
        require(isinstance(fp, dict) and set(fp) == {"path", "bytes", "sha256"},
                "Gate kit fingerprint fields")
        _fingerprint_claim(fp)
        require(fp["bytes"] <= 1024**2, "Gate kit member byte cap")
    require(sum(fp["bytes"] for fp in kit.values()) <= 4 * 1024**2, "Gate kit total byte cap")
    require(GATE_MODULES <= set(sources), "four original Gate module sources required")
    for name in GATE_MODULES:
        require(sources[name] == kit[name + ".py"], "Gate module source differs from kit: " + name)
    for name in ("required_role_sources", "required_role_hooks"):
        table = c.get(name)
        require(isinstance(table, dict) and set(table) == {"api", "worker", "client"}, name)
        for values in table.values():
            require(isinstance(values, list) and values and all(label(v) for v in values)
                    and len(values) == len(set(values)), name + " names")
            if name == "required_role_sources":
                require(set(values) <= set(sources), "unknown required source")
    require(GATE_MODULES | {"httpx._client", "openai._base_client"}
            <= set(c["required_role_sources"]["client"]), "complete Gate/SDK client source profile")
    routes = c.get("routes")
    require(isinstance(routes, list) and len(routes) == 4, "exactly four routes required")
    result, models, logical_ids = {}, defaultdict(set), set()
    for route in routes:
        require(isinstance(route, dict), "route is not an object")
        rid, model, mode = (route.get(k) for k in ("route_id", "model", "mode"))
        require(label(rid) and rid not in result and label(model), "route identity")
        require(mode in ("eager", "graph") and mode not in models[model], "route matrix")
        models[model].add(mode)
        ranks = route.get("expected_ranks")
        require(isinstance(ranks, list) and all(integer(r) for r in ranks)
                and len(ranks) == 4 and set(ranks) == {0, 1, 2, 3}, "rank contract")
        requests = route.get("requests")
        require(isinstance(requests, list) and len(requests) == 26, "26 requests required")
        ordinals, scenarios, expected, original_ids = set(), Counter(), {}, set()
        for request in requests:
            require(isinstance(request, dict), "request is not an object")
            lid, scenario, ordinal = (request.get(k) for k in ("logical_id", "scenario", "ordinal"))
            require(label(lid) and lid not in logical_ids, "logical id must be globally unique")
            logical_ids.add(lid)
            require(scenario in SCENARIOS and integer(ordinal)
                    and ordinal not in ordinals, "request scenario/ordinal")
            ordinals.add(ordinal)
            scenarios[scenario] += 1
            original = request.get("original_request_id")
            require(label(original) and (scenario, original) not in original_ids,
                    "original request id/scenario binding")
            original_ids.add((scenario, original))
            require(digest(request.get("request_input_sha256")), "request input digest")
            checks = request.get("required_checks")
            require(isinstance(checks, list) and len(checks) == 10
                    and all(isinstance(x, str) for x in checks)
                    and set(checks) == CHECKS, "exact ten required checks")
            expected[lid] = request
        require(ordinals == set(range(26)), "route ordinals must be exactly zero through 25")
        require(dict(scenarios) == SCENARIOS, "five original scenario counts")
        result[rid] = {"route": route, "requests": expected}
    require(len(models) == 2 and all(modes == {"eager", "graph"}
                                   for modes in models.values()), "two-model four-route matrix")
    return result


def _failure_data(value):
    if isinstance(value, dict):
        for key, item in value.items():
            lower = key.lower()
            if "cap_exceeded" in lower and item is not False:
                return True
            if lower in ("errors", "observer_errors", "telemetry_errors") and item != []:
                return True
            if lower in ("unsupported", "incomplete") and item is not False:
                return True
            if _failure_data(item):
                return True
    elif isinstance(value, list):
        return any(_failure_data(item) for item in value)
    return False


def _batch(batch):
    require(isinstance(batch, dict) and set(batch) == BATCH_FIELDS, "exact batch fields")
    require(integer(batch["num_tokens"])
            and (batch["num_reqs"] is None or integer(batch["num_reqs"]))
            and integer(batch["num_active_loras"]), "batch integers")
    require(type(batch["uniform"]) is bool and type(batch["has_lora"]) is bool,
            "batch booleans")


def _ids(ids, empty=False):
    require(isinstance(ids, list) and (empty or ids) and all(label(x) for x in ids)
            and len(ids) == len(set(ids)), "engine request ids")
    return set(ids)


def _fingerprint_claim(fp):
    require(isinstance(fp, dict) and label(fp.get("path"))
            and integer(fp.get("bytes"), 1) and digest(fp.get("sha256")),
            "scenario output fingerprint claim")


def _events(raw, c, routes, seen_ids):
    require(raw and raw.endswith(b"\n"), "empty or non-LF-terminated event file")
    events, identities = [], set()
    for line in raw.split(b"\n")[:-1]:
        require(line and b"\r" not in line, "blank line or raw CR in event JSONL")
        e = _json(line)
        require(isinstance(e, dict) and set(e) == EVENT_FIELDS, "exact event schema fields")
        require(type(e["schema_version"]) is int and e["schema_version"] == 1, "event schema version")
        for key in ("run_id", "observer_source_sha256", "source_binding_id"):
            require(e[key] == c[key], "event/config binding mismatch: " + key)
        rid = e["route_id"]
        require(isinstance(rid, str) and rid in routes, "unknown route")
        route = routes[rid]["route"]
        require(e["model"] == route["model"] and e["mode"] == route["mode"], "model/mode mismatch")
        require(label(e["event_id"]) and e["event_id"] not in seen_ids, "duplicate event id")
        seen_ids.add(e["event_id"])
        require(integer(e["pid"], 1) and label(e["starttime"])
                and re.fullmatch(r"[0-9]+", e["starttime"]) is not None
                and label(e["process_key"]), "process identity")
        require(isinstance(e["pid_namespace"], str)
                and re.fullmatch(r"pid:\[[0-9]+\]", e["pid_namespace"]) is not None,
                "PID namespace identity")
        require(integer(e["time_ns"]) and integer(e["monotonic_ns"]), "event timestamps")
        require(integer(e["seq"], 1), "event sequence")
        require(e["kind"] in KINDS and e["status"] == "ok", "unsupported/error/unknown event")
        require(isinstance(e["data"], dict) and not _failure_data(e["data"]),
                "event reports cap, error or incomplete telemetry")
        identities.add((rid, e["process_key"], e["pid"], e["starttime"], e["pid_namespace"]))
        events.append(e)
    require(len(identities) == 1, "event file must contain exactly one route/process")
    require([e["seq"] for e in events] == list(range(1, len(events) + 1)),
            "raw event file sequence must be contiguous in order")
    require(all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:])),
            "process monotonic timestamps decreased")
    return events


def _processes(events, routes, c):
    groups, labels = defaultdict(list), {}
    for event in events:
        key = (event["route_id"], event["process_key"])
        identity = (event["pid"], event["starttime"], event["pid_namespace"])
        require(key not in labels or labels[key] == identity, "process key identity reused")
        labels[key] = identity
        groups[key].append(event)
    roles, workers, client_scopes = defaultdict(lambda: defaultdict(list)), {}, {}
    for key, group in groups.items():
        # Sequence is a producer-local order, never a cross-process timestamp join.
        group.sort(key=lambda e: e["seq"])
        require([e["seq"] for e in group] == list(range(1, len(group) + 1)),
                "process sequence must start at one and be contiguous")
        require(group[0]["kind"] == "activation" and group[-1]["kind"] == "telemetry_end",
                "activation first and telemetry_end last required")
        require(sum(e["kind"] == "activation" for e in group) == 1
                and sum(e["kind"] == "telemetry_end" for e in group) == 1,
                "duplicate process boundary")
        activation, end = group[0]["data"], group[-1]["data"]
        role = activation.get("role")
        require(role in ("api", "worker", "client"), "activation role")
        require(activation.get("sources_verified") is True, "producer source acknowledgment missing")
        hooks = activation.get("hooks")
        require(isinstance(hooks, list) and hooks and all(label(h) for h in hooks)
                and len(hooks) == len(set(hooks)), "activation hooks")
        require(set(hooks) == set(c["required_role_hooks"][role]), "required role hooks mismatch")
        expected_sources = {name: c["sources"][name] for name in c["required_role_sources"][role]}
        require(activation.get("sources") == expected_sources, "required role source fingerprints mismatch")
        require(end.get("errors") == [] and isinstance(end.get("errors"), list), "telemetry errors")
        roles[key[0]][role].append(key)
        if role == "worker":
            rank = activation.get("rank")
            require(integer(rank) and rank in routes[key[0]]["route"]["expected_ranks"], "worker rank")
            rkey = (key[0], rank)
            require(rkey not in workers, "multiple processes claim one worker rank")
            workers[rkey] = key
        elif role == "client":
            loaded = activation.get("loaded_test_modules")
            require(isinstance(loaded, list) and len(loaded) == 1
                    and isinstance(loaded[0], str) and loaded[0] in CLIENT_SCOPES,
                    "client must acknowledge exactly one original collected test module")
            module = loaded[0]
            require(activation.get("gate_kit_files") == c["gate_kit_files"],
                    "client complete Gate kit fingerprints mismatch")
            require((key[0], module) not in client_scopes, "duplicate client module scope")
            client_scopes[key[0], module] = key
            scope = CLIENT_SCOPES[module]
            expected_logicals = {lid for lid, row in routes[key[0]]["requests"].items()
                                 if row["scenario"] in scope}
            actual_logicals, actual_scenarios = [], []
            for event in group[1:-1]:
                data, kind = event["data"], event["kind"]
                if kind in ("logical_result", "observed_send_call"):
                    lid = data.get("logical_id")
                    require(isinstance(lid, str) and lid in expected_logicals,
                            "client logical/send event escaped collected test scope")
                    if kind == "logical_result":
                        require(data.get("scenario") in scope,
                                "client logical scenario escaped collected test scope")
                        actual_logicals.append(lid)
                elif kind == "scenario_result":
                    scenario = data.get("scenario")
                    require(isinstance(scenario, str) and scenario in scope,
                            "client scenario event escaped collected test scope")
                    actual_scenarios.append(scenario)
            require(len(actual_logicals) == len(expected_logicals)
                    and set(actual_logicals) == expected_logicals,
                    "client scope logical results incomplete or duplicated")
            require(len(actual_scenarios) == len(scope) and set(actual_scenarios) == set(scope),
                    "client scope scenario results incomplete or duplicated")
        allowed = {"api": {"api_response", "engine_queued", "generate_call"},
                   "worker": {"scheduler_begin", "capture_success", "replay_success",
                              "scheduler_return", "graph_call", "native_capture_unassociated",
                              "native_replay_unassociated"},
                   "client": {"logical_result", "scenario_result", "observed_send_call"}}[role]
        require(all(e["kind"] in allowed for e in group[1:-1]), "event kind/role mismatch")
    for rid, details in routes.items():
        require(len(roles[rid]["api"]) == 1 and len(roles[rid]["client"]) == 3,
                "each route needs one API and three original pytest client processes")
        require({module for route, module in client_scopes if route == rid} == set(CLIENT_SCOPES),
                "each route needs all three original collected test module scopes")
        require(set(rank for route, rank in workers if route == rid)
                == set(details["route"]["expected_ranks"]), "missing worker activation rank")
    return groups, roles, workers


def _join(events, routes, groups, roles, workers, report):
    expected_logicals = sum(len(details["requests"]) for details in routes.values())
    expected_scenarios = len(routes) * len(SCENARIOS)
    expected_graphs = sum(len(details["requests"]) for details in routes.values()
                          if details["route"]["mode"] == "graph")
    expected_rank_joins = sum(len(details["requests"]) * len(details["route"]["expected_ranks"])
                              for details in routes.values() if details["route"]["mode"] == "graph")
    responses, queued, engines, logicals, scenarios = {}, {}, {}, {}, {}
    schedules, captures, replays = {}, {}, []
    for e in events:
        rid, data, kind = e["route_id"], e["data"], e["kind"]
        process = (rid, e["process_key"])
        if kind == "api_response":
            aid, response = data.get("api_call_id"), data.get("response_id")
            require(label(aid) and label(response), "API response ids")
            require((rid, response) not in responses, "ambiguous response id")
            responses[rid, response] = (aid, process)
        elif kind == "engine_queued":
            aid, engine, external = (data.get(k) for k in
                                     ("api_call_id", "engine_request_id", "external_req_id"))
            require(all(label(x) for x in (aid, engine, external)), "engine queued ids")
            require((rid, aid) not in queued and (rid, engine) not in engines,
                    "ambiguous API/engine queued id")
            queued[rid, aid] = (engine, process)
            engines[rid, engine] = aid
        elif kind == "scheduler_begin":
            execute, tokens = data.get("execute_call_id"), data.get("request_tokens")
            require(label(execute) and isinstance(tokens, dict)
                    and all(label(k) and integer(v) for k, v in tokens.items()), "scheduler tokens")
            require((process, execute) not in schedules, "duplicate execute id")
            schedules[process, execute] = e
        elif kind in ("capture_success", "replay_success"):
            _batch(data.get("batch"))
            for name in ("capture_id", "graph_id", "wrapper_uid"):
                require(label(data.get(name)), "graph identity: " + name)
            require(integer(data.get("graph_generation"), 1), "graph generation")
            rank = groups[process][0]["data"]["rank"]
            require(type(data.get("rank")) is int and data["rank"] == rank, "graph event rank")
            _ids(data.get("engine_request_ids"), empty=kind == "capture_success")
            key = (process, data["capture_id"], data["graph_generation"],
                   data["graph_id"], data["wrapper_uid"])
            if kind == "capture_success":
                require(key not in captures, "duplicate capture generation")
                captures[key] = e
            else:
                require(label(data.get("execute_call_id")), "replay needs execute call id")
                replays.append((key, e))
        elif kind == "logical_result":
            lid = data.get("logical_id")
            require(label(lid) and lid in routes[rid]["requests"] and (rid, lid) not in logicals,
                    "unknown/duplicate logical result")
            expected = routes[rid]["requests"][lid]
            for name in ("scenario", "ordinal", "request_input_sha256"):
                require(type(data.get(name)) is type(expected[name]) and data.get(name) == expected[name],
                        "logical input binding: " + name)
            require(label(data.get("response_id")), "logical response id")
            require(digest(data.get("original_result_sha256")), "original returned-row digest claim")
            require("original_error" in data and data["original_error"] is None,
                    "original returned row reports an error")
            checks = data.get("checks")
            require(isinstance(checks, dict) and set(checks) == CHECKS
                    and all(type(v) is bool for v in checks.values()), "ten boolean logical checks")
            require(data.get("passed") is True and all(checks.values()), "original logical quality failure")
            logicals[rid, lid] = e
        elif kind == "scenario_result":
            scenario = data.get("scenario")
            require(scenario in SCENARIOS and (rid, scenario) not in scenarios, "scenario identity")
            expected = [r["logical_id"] for r in routes[rid]["route"]["requests"]
                        if r["scenario"] == scenario]
            ids = data.get("expected_logical_ids")
            require(isinstance(ids, list) and ids == expected, "scenario expected logical ids/order")
            _fingerprint_claim(data.get("output_fingerprint"))
            scenarios[rid, scenario] = e
        elif kind == "observed_send_call":
            require(label(data.get("logical_id"))
                    and data["logical_id"] in routes[rid]["requests"],
                    "observed SDK send has unknown logical id")
    require(len(logicals) == expected_logicals and len(scenarios) == expected_scenarios,
            "all selected logical results and scenario fingerprints required")
    require(len(responses) == expected_logicals and len(queued) == expected_logicals,
            "all selected unique API responses and queued requests required")
    used_responses = {(rid, event["data"]["response_id"]) for (rid, _), event in logicals.items()}
    require(len(used_responses) == expected_logicals and used_responses == set(responses),
            "logical results must use each actual API response exactly once")
    response_calls = {(rid, aid) for (rid, _), (aid, _) in responses.items()}
    require(len(response_calls) == expected_logicals and response_calls == set(queued),
            "API responses must cover each queued API call exactly once")
    joins = defaultdict(dict)
    for key, replay in replays:
        data, process = replay["data"], key[0]
        require(key in captures, "replay lacks matching capture identity/generation")
        capture = captures[key]
        require(capture["seq"] < replay["seq"] and capture["data"]["batch"] == data["batch"],
                "capture/replay shape or order mismatch")
        execute = data["execute_call_id"]
        require((process, execute) in schedules, "replay lacks same-process scheduler")
        schedule = schedules[process, execute]
        require(schedule["data"]["request_tokens"]
                and all(integer(value, 1) for value in schedule["data"]["request_tokens"].values())
                and integer(data["batch"]["num_tokens"], 1), "replay has no actual positive scheduled tokens")
        require(schedule["seq"] < replay["seq"] and set(schedule["data"]["request_tokens"])
                == _ids(data["engine_request_ids"]), "scheduler/replay engine IDs mismatch")
        for engine in data["engine_request_ids"]:
            require((process[0], engine) in engines, "replay engine has no queued API request")
            evidence = joins[process[0], engine].get(data["rank"])
            if evidence is None:
                joins[process[0], engine][data["rank"]] = {
                    "rank": data["rank"], "process_key": replay["process_key"],
                    "pid": replay["pid"], "starttime": replay["starttime"],
                    "pid_namespace": replay["pid_namespace"],
                    "capture_event_id": capture["event_id"], "capture_seq": capture["seq"],
                    "scheduler_event_id": schedule["event_id"], "scheduler_seq": schedule["seq"],
                    "replay_event_id": replay["event_id"], "replay_seq": replay["seq"],
                    "capture_time_ns": capture["time_ns"], "replay_time_ns": replay["time_ns"],
                    "capture_monotonic_ns": capture["monotonic_ns"],
                    "replay_monotonic_ns": replay["monotonic_ns"],
                    "execute_call_id": execute, "capture_id": data["capture_id"],
                    "graph_generation": data["graph_generation"], "graph_id": data["graph_id"],
                    "wrapper_uid": data["wrapper_uid"], "batch": data["batch"],
                    "matching_replay_count": 1,
                }
            else:
                evidence["matching_replay_count"] += 1
    graph_joins, logical_joins = [], []
    for (rid, lid), logical in logicals.items():
        response = logical["data"]["response_id"]
        require((rid, response) in responses, "logical result has no actual API response")
        aid, api_process = responses[rid, response]
        require((rid, aid) in queued and queued[rid, aid][1] == api_process,
                "API response/engine queue process mismatch")
        engine = queued[rid, aid][0]
        expected = routes[rid]["requests"][lid]
        logical_joins.append({
            "route_id": rid, "logical_id": lid, "scenario": expected["scenario"],
            "ordinal": expected["ordinal"], "original_request_id": expected["original_request_id"],
            "request_input_sha256": expected["request_input_sha256"],
            "logical_event_id": logical["event_id"], "response_id": response,
            "api_call_id": aid, "engine_request_id": engine,
            "checks": logical["data"]["checks"], "passed": logical["data"]["passed"],
            "original_result_sha256": logical["data"]["original_result_sha256"],
            "original_error": logical["data"]["original_error"],
        })
        if routes[rid]["route"]["mode"] == "graph":
            require(set(joins[rid, engine]) == set(routes[rid]["route"]["expected_ranks"]),
                    "graph request missing replay join on every rank")
            graph_joins.append({"route_id": rid, "logical_id": lid,
                                "response_id": response, "api_call_id": aid,
                                "engine_request_id": engine, "ranks": sorted(joins[rid, engine]),
                                "rank_evidence": [joins[rid, engine][rank]
                                                  for rank in sorted(joins[rid, engine])]})
    require(len(graph_joins) == expected_graphs, "all selected graph logical-request joins required")
    actual_rank_joins = sum(len(join["ranks"]) for join in graph_joins)
    require(actual_rank_joins == expected_rank_joins, "all selected graph request/rank joins required")
    report["counts"].update(logical_results=len(logicals), scenario_results=len(scenarios),
                            api_responses=len(responses), queued_requests=len(queued),
                            graph_request_joins=len(graph_joins),
                            graph_request_rank_joins=actual_rank_joins,
                            captures=len(captures), replays=len(replays))
    report["graph_request_joins"] = graph_joins
    report["logical_response_engine_joins"] = sorted(logical_joins,
                                                     key=lambda row: (row["route_id"], row["ordinal"]))
    report["graph_ranks_complete"] = (sorted(set.intersection(
        *(set(join["ranks"]) for join in graph_joins))) if graph_joins else [])
    report["scenario_output_fingerprint_claims"] = [
        {"route_id": rid, "scenario": scenario,
         "output_fingerprint": event["data"]["output_fingerprint"]}
        for (rid, scenario), event in sorted(scenarios.items())]


def summarize(config_path, event_paths, selected_routes=None, partial_route=None):
    """Read and summarize. Never writes files or imports a production package."""
    report = {
        "schema_version": 1, "record_kind": "observer_event_contract_summary",
        "observer_complete": False, "graph_observation_complete": False,
        "scope": "route_only" if partial_route is not None else "full_matrix",
        "full_matrix_complete": False, "full_matrix_observer_complete": False,
        "full_matrix_graph_observation_complete": False,
        "route_observer_complete": False, "route_graph_observation_complete": False,
        "route_graph_observation_applicability": "unknown",
        "acceptance": {"source": False, "GPU": False, "ABI": False,
                       "model": False, "Gate": False},
        "limitations": ["Producer source acknowledgments are claims, not independent source proof.",
                        "Graph joins do not establish native completion, ABI, model or Gate acceptance.",
                        "Returned-row and scenario-file fingerprints are distinct producer claims; outputs are not reread here."],
        "input_fingerprints": [], "counts": {}, "errors": [], "graph_ranks_complete": [],
        "process_identities": {}, "logical_response_engine_joins": [],
    }
    budget, stage = [0], "config"
    try:
        raw, fp = _read(config_path, budget)
        report["input_fingerprints"].append(fp)
        c = _json(raw)
        routes = _config(c)
        report["counts"].update(manifest_routes=len(routes), manifest_logical_results=sum(
            len(details["requests"]) for details in routes.values()))
        if partial_route is not None:
            require(selected_routes is None and label(partial_route) and partial_route in routes,
                    "partial route must be exactly one configured route, without full-route selection")
            routes = {partial_route: routes[partial_route]}
            route = routes[partial_route]["route"]
            report.update(selected_route_id=partial_route, model=route["model"], mode=route["mode"],
                          route_graph_observation_applicability=(
                              "applicable" if route["mode"] == "graph" else "not_applicable"))
        if selected_routes is not None:
            require(isinstance(selected_routes, list) and len(selected_routes) == 4
                    and all(label(r) for r in selected_routes)
                    and len(set(selected_routes)) == 4 and set(selected_routes) == set(routes),
                    "selected routes must be exactly all four configured routes without duplicates")
        report.update(run_id=c["run_id"], observer_source_sha256=c["observer_source_sha256"],
                      source_binding_id=c["source_binding_id"])
        report["counts"].update(expected_routes=len(routes), expected_logical_results=sum(
            len(details["requests"]) for details in routes.values()),
            expected_scenarios=len(routes) * len(SCENARIOS), expected_graph_requests=sum(
                len(details["requests"]) for details in routes.values()
                if details["route"]["mode"] == "graph"))
        stage = "events"
        paths = list(event_paths)
        require(0 < len(paths) <= MAX_EVENT_FILES, "event file count must be 1..128")
        stage, events, seen_ids, seen_paths = "events", [], set(), set()
        for path in paths:
            canonical = str(Path(path).absolute())
            require(canonical not in seen_paths, "duplicate event input path")
            seen_paths.add(canonical)
            raw, fp = _read(path, budget)
            report["input_fingerprints"].append(fp)
            events.extend(_events(raw, c, routes, seen_ids))
        report["counts"].update(event_files=len(paths), events=len(events), input_bytes=budget[0])
        report["kind_counts"] = dict(sorted(Counter(e["kind"] for e in events).items()))
        stage = "processes"
        groups, roles, workers = _processes(events, routes, c)
        report["counts"]["processes"] = len(groups)
        report["process_identities"] = {
            rid: [{"role": group[0]["data"]["role"],
                   "rank": group[0]["data"].get("rank"), "pid": group[0]["pid"],
                   "starttime": group[0]["starttime"], "pid_namespace": group[0]["pid_namespace"],
                   "process_key": group[0]["process_key"],
                   **({"loaded_test_modules": group[0]["data"]["loaded_test_modules"]}
                      if group[0]["data"]["role"] == "client" else {})}
                  for key, group in sorted(groups.items()) if key[0] == rid]
            for rid in sorted(routes)
        }
        stage = "joins"
        _join(events, routes, groups, roles, workers, report)
        if partial_route is None:
            report["observer_complete"] = report["graph_observation_complete"] = True
            report["full_matrix_complete"] = report["full_matrix_observer_complete"] = True
            report["full_matrix_graph_observation_complete"] = True
        else:
            report["route_observer_complete"] = True
            report["route_graph_observation_complete"] = report["mode"] == "graph"
    except (ContractError, OSError, ValueError, TypeError, KeyError, RecursionError) as ex:
        report["errors"].append({"stage": stage, "type": type(ex).__name__,
                                 "message": str(ex)[:4096]})
    return report


def _write_exclusive(path, raw):
    p = Path(path).absolute()
    require(p.parent.resolve(strict=True) == p.parent, "noncanonical output parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(p, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd is not None:
            os.close(fd)


def _trace_paths(path):
    directory = Path(path).absolute()
    require(directory.resolve(strict=True) == directory, "noncanonical trace directory")
    before = directory.lstat()
    require(stat.S_ISDIR(before.st_mode), "non-directory trace input")
    paths = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith(".jsonl"):
                continue
            require(len(paths) < MAX_EVENT_FILES, "trace directory event file cap exceeded")
            require(stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode),
                    "nonregular or symlink trace file")
            paths.append(directory / entry.name)
    require(_identity(before) == _identity(directory.lstat()), "trace directory changed during enumeration")
    return sorted(paths), directory, before


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    config_args = parser.add_mutually_exclusive_group(required=True)
    config_args.add_argument("--expected-manifest")
    config_args.add_argument("--config", help="Diagnostic alias for the same strict four-route contract")
    input_args = parser.add_mutually_exclusive_group(required=True)
    input_args.add_argument("--trace-dir", help="Only direct regular *.jsonl files; no recursion")
    input_args.add_argument("--events", nargs="+", help="Diagnostic explicit-file input")
    route_args = parser.add_mutually_exclusive_group()
    route_args.add_argument("--route", action="append", help="Repeat once for each of the four configured routes")
    route_args.add_argument("--partial-route", action="append",
                            help="Observe exactly one route; never grants full-matrix completeness")
    parser.add_argument("--output", help="Create exclusively; existing output is never overwritten")
    args = parser.parse_args(argv)
    if args.partial_route is not None and len(args.partial_route) != 1:
        parser.error("--partial-route must occur exactly once")
    partial_route = args.partial_route[0] if args.partial_route is not None else None
    if args.expected_manifest:
        if (not args.trace_dir or (not args.route and args.partial_route is None) or not args.output):
            parser.error("--expected-manifest requires --trace-dir, four --route values or one --partial-route, and --output")
    elif args.trace_dir or args.route or args.partial_route is not None:
        parser.error("diagnostic --config requires --events and no --route")
    config_path = args.expected_manifest or args.config
    if args.trace_dir:
        try:
            paths, directory, before = _trace_paths(args.trace_dir)
            report = summarize(config_path, paths, args.route, partial_route)
            require(_identity(before) == _identity(directory.lstat())
                    and directory.resolve(strict=True) == directory,
                    "trace directory changed during summary")
        except (ContractError, OSError) as ex:
            report = summarize(config_path, [], partial_route=partial_route)
            report["errors"].append({"stage": "trace_directory", "type": type(ex).__name__,
                                     "message": str(ex)[:4096]})
    else:
        report = summarize(config_path, args.events)
    raw = (json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    if args.output:
        try:
            _write_exclusive(args.output, raw)
        except (ContractError, OSError) as ex:
            print("exclusive output failed: " + str(ex), file=sys.stderr)
            return 2
    else:
        sys.stdout.write(raw.decode("utf-8"))
    complete = (report["route_observer_complete"] if partial_route is not None
                else report["observer_complete"])
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

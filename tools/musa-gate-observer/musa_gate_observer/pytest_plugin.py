"""Observe the original Gate/SDK calls, without changing requests or checks."""

import functools
import hashlib
import os
import sys
import traceback
from pathlib import Path

from .trace import (Config, Trace, Unsupported, digest, logical_context,
                    read_original, scenario_context, strict_json, unique_id)

_state = {}
CHECKS = {"non_empty", "minimum_length", "expected_semantics", "expected_order",
          "no_bang_triplet", "no_mojibake", "no_control_characters",
          "no_long_character_run", "no_repeated_word_run", "no_repeated_phrase"}
HOOKS = ["case_utils.run_case", "case_utils.execute_request", "httpx.AsyncClient.send"]


def expected_request(config, scenario, request):
    rows = [row for row in config.route["requests"]
            if row["scenario"] == scenario
            and row["original_request_id"] == request["request_id"]]
    if len(rows) != 1 or digest(request) != rows[0]["request_input_sha256"]:
        raise Unsupported("Actual original request differs from bound request")
    return rows[0]


def observe_result(trace, expected, result):
    checks = result.get("checks")
    response = result.get("response", {})
    if not isinstance(checks, dict) or (checks and (set(checks) != CHECKS
            or not all(type(value) is bool for value in checks.values()))):
        raise Unsupported("Actual original check schema differs")
    trace.emit("logical_result", {
        "logical_id": expected["logical_id"], "scenario": expected["scenario"],
        "ordinal": expected["ordinal"],
        "request_input_sha256": expected["request_input_sha256"],
        "response_id": response.get("id"), "checks": checks,
        "passed": result.get("passed"), "original_error": result.get("error"),
        "original_result_sha256": digest(result),
    })


def scenario_original(config, gate, scenario):
    # Derive exactly the original case_utils.run_case destination; do not write it.
    port = os.environ["PORT"]
    model = os.environ.get("MODEL_PATH") or gate.resolve_served_model_name()
    root = Path(os.environ.get("RESULTS_DIR", str(gate.ROOT / "results")))
    path = root / gate.safe_slug(model) / f"port-{gate.safe_slug(port)}" / f"{scenario}.json"
    raw, fingerprint = read_original(path)
    document = strict_json(raw)
    if (document.get("schema_version") != 1
            or document.get("case", {}).get("scenario") != scenario
            or document["case"].get("port") != port
            or document["case"].get("model_path") != os.environ.get("MODEL_PATH")
            or document["case"].get("served_model_name") != gate.resolve_served_model_name()):
        raise Unsupported("Original scenario result identity differs")
    inputs, outputs = document["input"], document["output"]
    if len(inputs) != len(outputs) or document["case"].get("concurrency") != len(inputs):
        raise Unsupported("Original scenario result lengths differ")
    expected = [expected_request(config, scenario, request) for request in inputs]
    bound = [row for row in config.route["requests"] if row["scenario"] == scenario]
    if expected != bound or any(output["request_id"] != request["request_id"]
                               for request, output in zip(inputs, outputs)):
        raise Unsupported("Original scenario result input order differs")
    passed = sum(output.get("passed") is True for output in outputs)
    summary = {"request_count": len(inputs), "passed_count": passed,
               "failed_count": len(inputs) - passed, "passed": passed == len(inputs)}
    if document.get("summary") != summary:
        raise Unsupported("Original scenario summary differs")
    return expected, outputs, fingerprint


def install(collected_modules):
    config = Config()
    trace = Trace(config, "client")
    _state.update(config=config, trace=trace, send_counts={}, results={})
    try:
        minimum = {"case_utils", "test_text", "test_image", "test_mix_text_image",
                   "httpx._client", "openai._base_client"}
        if not minimum <= set(config.data["required_role_sources"]["client"]):
            raise Unsupported("Missing required original Gate/SDK source profile")
        if len(collected_modules) != 1 or collected_modules[0] not in {
                "test_text", "test_image", "test_mix_text_image"}:
            raise Unsupported("Expected one original test file per pytest invocation")
        gate_files = config.verify_gate_kit()
        sources = config.verify_role("client", static_names={
            "test_text", "test_image", "test_mix_text_image"})
        for name in ("case_utils", "test_text", "test_image", "test_mix_text_image"):
            if sources[name] != gate_files[f"{name}.py"]:
                raise Unsupported("Module source differs from complete original Gate kit")
        gate = sys.modules["case_utils"]
        httpx = sys.modules["httpx._client"]
        original_case = gate.run_case
        original_request = gate.execute_request
        original_send = httpx.AsyncClient.send
        if any(hasattr(value, "_musa_gate_original")
               for value in (original_case, original_request, original_send)):
            raise Unsupported("Client hooks already replaced")
        if set(config.data["required_role_hooks"]["client"]) != set(HOOKS):
            raise Unsupported("Client hook contract differs")

        @functools.wraps(original_request)
        async def request_wrapper(client, request, served_model_name):
            expected = None
            token = None
            try:
                expected = expected_request(config, scenario_context.get(), request)
                token = logical_context.set(expected)
            except Exception as error:
                trace.fail(error)
            try:
                result = await original_request(client, request, served_model_name)
                if expected is not None:
                    try:
                        observe_result(trace, expected, result)
                        if expected["logical_id"] in _state["results"]:
                            raise Unsupported("Duplicate original logical request")
                        _state["results"][expected["logical_id"]] = digest(result)
                    except Exception as error:
                        trace.fail(error)
                return result
            except BaseException:
                trace.emit("logical_exception", {"logical_id": expected["logical_id"]
                           if expected else None}, "error")
                raise
            finally:
                if token is not None:
                    logical_context.reset(token)

        @functools.wraps(original_case)
        def case_wrapper(scenario):
            token = scenario_context.set(scenario)
            try:
                return original_case(scenario)
            finally:
                try:
                    expected, outputs, fingerprint = scenario_original(config, gate, scenario)
                    for row, output in zip(expected, outputs):
                        if _state["results"].get(row["logical_id"]) != digest(output):
                            raise Unsupported("Original JSON output differs from actual returned row")
                    trace.emit("scenario_result", {"scenario": scenario,
                        "expected_logical_ids": [row["logical_id"] for row in expected],
                        "output_fingerprint": fingerprint})
                except Exception as error:
                    trace.fail(error)
                finally:
                    scenario_context.reset(token)

        @functools.wraps(original_send)
        async def send_wrapper(client, request, *args, **kwargs):
            logical = logical_context.get()
            data = {"logical_id": logical["logical_id"] if logical else None,
                    "send_call_id": unique_id(), "method": str(request.method),
                    "url_path": str(request.url.path),
                    "classification": "observed_httpx_AsyncClient_send_invocation"}
            if logical:
                key = logical["logical_id"]
                _state["send_counts"][key] = _state["send_counts"].get(key, 0) + 1
                data["send_ordinal"] = _state["send_counts"][key]
            try:
                data["already_buffered_request_bytes"] = len(request.content)
                data["already_buffered_request_sha256"] = hashlib.sha256(request.content).hexdigest()
            except Exception:
                data["request_body_state"] = "not_already_buffered_no_extra_read"
            try:
                result = await original_send(client, request, *args, **kwargs)
                data["status_code"] = result.status_code
                trace.emit("observed_send_call", data)
                return result
            except BaseException:
                trace.emit("observed_send_exception", data, "error")
                raise

        for owner, name, original, wrapped in (
                (gate, "execute_request", original_request, request_wrapper),
                (gate, "run_case", original_case, case_wrapper),
                (httpx.AsyncClient, "send", original_send, send_wrapper)):
            wrapped._musa_gate_original = original
            setattr(owner, name, wrapped)
            if getattr(owner, name) is not wrapped:
                raise Unsupported("Client hook not retained")
        for name in collected_modules:
            module = sys.modules[name]
            if module.run_case is not original_case:
                raise Unsupported("Original test run_case alias differs")
            module.run_case = case_wrapper
            if module.run_case is not case_wrapper:
                raise Unsupported("Test run_case alias replacement not retained")
        trace.activation(HOOKS, sources, client_scope={
            "loaded_test_modules": collected_modules, "gate_kit_files": gate_files})
    except Exception as error:
        if trace.seq == 0:
            trace.activation([], {}, "unsupported", str(error))
        trace.fail(error)


def pytest_collection_modifyitems(session, config, items):
    """Called after original test modules load; no eager SDK/test import."""
    if not _state:
        try:
            install(sorted({item.module.__name__ for item in items}))
        except Exception:
            sys.stderr.write("MUSA_GATE_OBSERVER_UNSUPPORTED\n" + traceback.format_exc())


def pytest_sessionfinish(session, exitstatus):
    if _state.get("trace") is not None:
        _state["trace"].close()

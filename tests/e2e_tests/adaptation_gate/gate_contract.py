# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared, CPU-testable contracts for the Qwen3.6 adaptation Gate."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCENARIOS: tuple[tuple[str, int], ...] = (
    ("text_single", 1),
    ("text_concurrent_8", 8),
    ("image_single", 1),
    ("image_concurrent_8", 8),
    ("mixed_concurrent_8", 8),
)

QUALITY_CHECKS = frozenset(
    {
        "non_empty",
        "minimum_length",
        "expected_semantics",
        "expected_order",
        "no_bang_triplet",
        "no_mojibake",
        "no_control_characters",
        "no_long_character_run",
        "no_repeated_word_run",
        "no_repeated_phrase",
    }
)

_MISSING = object()
_LOCAL_MEDIA_FLAG = "--allowed-local-media-path"


def normalized_text(text: str) -> str:
    """Normalize model output for semantic checks."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    for word, digit in {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }.items():
        normalized = re.sub(rf"\b{word}\b", digit, normalized)
    return normalized


def _has_repeated_phrase(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for size in range(2, min(13, len(tokens) // 3 + 1)):
        for start in range(0, len(tokens) - size * 3 + 1):
            phrase = tokens[start : start + size]
            if (
                phrase
                == tokens[start + size : start + size * 2]
                == tokens[start + size * 2 : start + size * 3]
            ):
                return True
    return False


def quality_checks(
    text: str, expected: list[str], min_length: int = 1
) -> dict[str, bool]:
    """Apply the ten adaptation checks to one configured chat case."""
    normalized = normalized_text(text)
    positions: list[int] = []
    cursor = 0
    for term in expected:
        position = normalized.find(str(term).lower(), cursor)
        positions.append(position)
        if position >= 0:
            cursor = position + len(str(term))
    mojibake_markers = (
        "\ufffd",
        chr(0x00C3),
        chr(0x00C2),
        "\u00e2\u20ac",
        "\u00ef\u00bf\u00bd",
    )
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {
        "non_empty": bool(normalized),
        "minimum_length": len(text.strip()) >= min_length,
        "expected_semantics": all(position >= 0 for position in positions),
        "expected_order": positions == sorted(positions)
        and all(position >= 0 for position in positions),
        "no_bang_triplet": "!!!" not in text,
        "no_mojibake": not any(marker in text for marker in mojibake_markers),
        "no_control_characters": not any(
            ord(char) < 32 and char not in "\n\r\t" for char in text
        ),
        "no_long_character_run": re.search(r"([^\s])\1{7,}", text) is None,
        "no_repeated_word_run": not any(
            tokens[index] == tokens[index + 1] == tokens[index + 2]
            for index in range(len(tokens) - 2)
        ),
        "no_repeated_phrase": not _has_repeated_phrase(text),
    }


def safe_component(value: str) -> str:
    """Return one filesystem-safe path component."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return component or "unknown"


def gate_results_root(
    repo_root: Path,
    *,
    run_id: str,
    platform: str,
    device: str,
    model: str,
    case: str,
) -> Path:
    """Build a per-run result root without allowing values to add path levels."""
    return (
        repo_root
        / "test-results"
        / "gate"
        / safe_component(run_id)
        / safe_component(platform)
        / safe_component(device)
        / safe_component(model)
        / safe_component(case)
    ).resolve()


def gate_run_id(environment: Mapping[str, str]) -> str:
    """Return the framework run id, or create one for direct pytest runs."""
    return environment.get("FL_RUN_ID") or uuid.uuid4().hex


def append_loopback_no_proxy(value: str | None) -> str:
    """Preserve proxy exclusions and add both loopback host spellings."""
    entries = [item.strip() for item in (value or "").split(",") if item.strip()]
    return ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost"]))


def resolve_request_timeout(environment: Mapping[str, str]) -> str:
    """Resolve the Gate client timeout while keeping the case source unchanged."""
    value = environment.get(
        "FL_ADAPTATION_GATE_REQUEST_TIMEOUT",
        environment.get("REQUEST_TIMEOUT", "300"),
    )
    try:
        if float(value) <= 0:
            raise ValueError
    except ValueError as error:
        raise ValueError(
            "adaptation Gate request timeout must be a positive number"
        ) from error
    return value


@contextmanager
def temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    """Apply environment updates and exactly restore the previous state."""
    previous = {name: os.environ.get(name, _MISSING) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)


def validate_scenario_documents(
    directory: Path, scenarios: tuple[tuple[str, int], ...]
) -> dict[str, int | bool]:
    """Validate one selected group of adaptation Gate scenario documents."""
    directory = Path(directory)
    documents = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    by_scenario: dict[str, dict[str, Any]] = {}
    for document in documents:
        scenario = document.get("case", {}).get("scenario")
        if not isinstance(scenario, str) or scenario in by_scenario:
            raise AssertionError("Gate results contain a missing or duplicate scenario")
        by_scenario[scenario] = document

    expected_scenarios = dict(scenarios)
    if len(documents) != len(scenarios) or set(by_scenario) != set(expected_scenarios):
        raise AssertionError(
            "Gate produced an unexpected scenario set: "
            f"expected={sorted(expected_scenarios)}, actual={sorted(by_scenario)}"
        )

    request_count = 0
    check_count = 0
    passed_checks = 0
    passed_requests = 0
    for scenario, expected_count in scenarios:
        document = by_scenario[scenario]
        inputs = document.get("input")
        outputs = document.get("output")
        if not isinstance(inputs, list) or len(inputs) != expected_count:
            raise AssertionError(
                f"{scenario} must contain {expected_count} input requests"
            )
        if not isinstance(outputs, list) or len(outputs) != expected_count:
            raise AssertionError(
                f"{scenario} must contain {expected_count} output responses"
            )

        scenario_passed = 0
        for output in outputs:
            checks = output.get("checks") if isinstance(output, dict) else None
            if not isinstance(checks, dict) or set(checks) != QUALITY_CHECKS:
                raise AssertionError(
                    f"{scenario} response has missing or unexpected quality checks"
                )
            if any(type(value) is not bool for value in checks.values()):
                raise AssertionError(
                    f"{scenario} quality check values must all be bool"
                )
            row_passed = output.get("passed") is True and all(checks.values())
            scenario_passed += int(row_passed)
            passed_requests += int(row_passed)
            check_count += len(checks)
            passed_checks += sum(checks.values())

        summary = document.get("summary")
        expected_summary = {
            "request_count": expected_count,
            "passed_count": scenario_passed,
            "failed_count": expected_count - scenario_passed,
            "passed": scenario_passed == expected_count,
        }
        if not isinstance(summary, dict) or any(
            summary.get(name) != value for name, value in expected_summary.items()
        ):
            raise AssertionError(f"{scenario} summary does not match its responses")
        request_count += expected_count

    expected_request_count = sum(expected_scenarios.values())
    expected_check_count = expected_request_count * len(QUALITY_CHECKS)
    result = {
        "logical_requests": request_count,
        "checks": check_count,
        "passed_checks": passed_checks,
        "passed_requests": passed_requests,
        "passed": (
            request_count == expected_request_count
            and check_count == expected_check_count
            and passed_checks == expected_check_count
            and passed_requests == expected_request_count
        ),
    }
    if not result["passed"]:
        raise AssertionError(f"Adaptation Gate quality contract failed: {result}")
    return result


def validate_gate_documents(directory: Path) -> dict[str, int | bool]:
    """Validate the original five-scenario, 26-request, 260-check contract."""
    return validate_scenario_documents(directory, SCENARIOS)


# ---------------------------------------------------------------------------
# Gate execution helpers shared by the serving and concurrent runners.
# These stay free of any concrete request data: runners pass configured
# entries in, and every response is graded with the ten quality checks.
# ---------------------------------------------------------------------------


def apply_local_media_flag(extra_args: list[str], media_dir: Path | None) -> list[str]:
    """Append ``--allowed-local-media-path`` once when local images are used.

    Raises when the flag is already present: user configuration must not
    race with the runner-owned argument.
    """
    if any(
        arg == _LOCAL_MEDIA_FLAG or str(arg).startswith(_LOCAL_MEDIA_FLAG + "=")
        for arg in extra_args
    ):
        raise ValueError(
            f"{_LOCAL_MEDIA_FLAG} must be supplied only by the gate runner, "
            "not by model or platform configuration"
        )
    if media_dir is None:
        return list(extra_args)
    return [*extra_args, _LOCAL_MEDIA_FLAG, str(Path(media_dir).resolve())]


def ensure_gate_server_alive(server: Any, phase: str) -> None:
    """Fail immediately when the shared gate server has exited."""
    process = getattr(server, "_process", None)
    if process is None or process.poll() is not None:
        details = server.preserve_failure_logs(
            f"vLLM service exited before the {phase} gate scenario"
        )
        raise RuntimeError(details)


def ensure_clean_gate_directory(directory: Path) -> Path:
    """Create the per-job document directory and refuse stale results."""
    directory = Path(directory)
    if directory.exists() and any(directory.glob("*.json")):
        raise RuntimeError(f"Refusing to reuse stale gate results: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def evaluate_gate_entry(
    entry: Mapping[str, Any],
    text: str,
    *,
    error: str | None = None,
    latency_seconds: float | None = None,
) -> dict[str, Any]:
    """Grade one response with exactly the ten quality checks.

    Failed or errored requests still yield ten booleans (all False) so the
    JSON document stays contract-valid and the failure is visible in the
    error field and in the counters.
    """
    if error:
        checks: dict[str, bool] = {name: False for name in QUALITY_CHECKS}
    else:
        checks = quality_checks(
            text, list(entry["expected"]), min_length=int(entry["min_length"])
        )
    output: dict[str, Any] = {
        "request": entry["name"],
        "latency_seconds": (
            round(latency_seconds, 3) if latency_seconds is not None else None
        ),
        "text": text,
        "checks": checks,
        "passed": not error and all(checks.values()),
    }
    if error:
        output["error"] = error
    return output


async def execute_gate_entries(
    client: Any, served_model_name: str, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run all entries of one scenario through a single asyncio.gather."""
    api = getattr(getattr(client, "chat", None), "completions", None)
    if api is None or not hasattr(api, "create"):
        raise TypeError("client must expose chat.completions.create()")

    async def run_one(entry: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            completion = await api.create(
                model=served_model_name,
                messages=entry["messages"],
                max_tokens=int(entry["max_tokens"]),
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = completion.choices[0].message.content or ""
            return evaluate_gate_entry(
                entry,
                text,
                latency_seconds=time.monotonic() - started,
            )
        except Exception as error:  # noqa: BLE001 - recorded, never raised
            return evaluate_gate_entry(
                entry,
                "",
                error=f"{type(error).__name__}: {error}",
                latency_seconds=time.monotonic() - started,
            )

    return await asyncio.gather(*(run_one(entry) for entry in entries))


def write_gate_document(
    directory: Path,
    scenario: str,
    *,
    model_path: str,
    served_model_name: str,
    port: Any,
    entries: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> Path:
    """Write one scenario document atomically, compatible with the validator."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    passed_count = sum(
        1
        for output in outputs
        if output.get("passed") is True and all(output.get("checks", {}).values())
    )
    document = {
        "schema_version": 1,
        "case": {
            "scenario": scenario,
            "model_path": model_path,
            "served_model_name": served_model_name,
            "port": str(port),
            "concurrency": len(entries),
        },
        "input": [
            {
                "request": entry["name"],
                "messages": entry["messages"],
                "expected": list(entry["expected"]),
                "min_length": int(entry["min_length"]),
                "max_tokens": int(entry["max_tokens"]),
            }
            for entry in entries
        ],
        "output": outputs,
        "summary": {
            "request_count": len(entries),
            "passed_count": passed_count,
            "failed_count": len(entries) - passed_count,
            "passed": passed_count == len(entries),
        },
    }
    destination = directory / f"{scenario}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def gate_failure_details(outputs: list[dict[str, Any]], limit: int = 160) -> str:
    """Render failed requests as compact diagnostic lines."""
    lines = []
    for output in outputs:
        if output.get("passed"):
            continue
        error = output.get("error")
        failed_checks = [
            name for name, passed in output.get("checks", {}).items() if not passed
        ]
        detail = f"failed checks: {', '.join(failed_checks) or 'unknown'}"
        if error:
            detail = f"{error}; {detail}"
        text = " ".join(str(output.get("text", "")).split())
        if text:
            detail += f"; output: {text[:limit]!r}"
        lines.append(f"    - {output.get('request', '<unnamed>')}: {detail}")
    return "\n".join(lines)


def format_gate_scenario_log(
    phase: str, scenario: str, outputs: list[dict[str, Any]]
) -> str:
    """One grep-friendly log line with the real per-scenario counters."""
    passed_requests = sum(1 for output in outputs if output.get("passed") is True)
    passed_checks = sum(
        1
        for output in outputs
        for passed in output.get("checks", {}).values()
        if passed
    )
    total_checks = len(outputs) * len(QUALITY_CHECKS)
    return (
        f"[Gate] {phase} scenario={scenario} "
        f"requests={passed_requests}/{len(outputs)} "
        f"checks={passed_checks}/{total_checks}"
    )


def format_gate_total_log(phase: str, summary: Mapping[str, Any]) -> str:
    """Job-total log line computed from validate_scenario_documents output."""
    status = "PASS" if summary["passed"] else "FAIL"
    return (
        f"[Gate] {phase} total: {status}; "
        f"requests={summary['passed_requests']}/{summary['logical_requests']}; "
        f"checks={summary['passed_checks']}/{summary['checks']}"
    )


def create_gate_client(*, base_url: str, api_key: str, timeout: str | float) -> Any:
    """Build the gate HTTP client pair; the proxy bypass is explicit."""
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(trust_env=False)
    client = AsyncOpenAI(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        timeout=float(timeout),
        http_client=http_client,
    )
    return client, http_client


def run_concurrent_scenario(
    entries: list[dict[str, Any]],
    *,
    served_model_name: str,
    base_url: str,
    api_key: str,
    timeout: str | float,
    document_dir: Path,
    scenario: str,
    model_path: str,
    port: Any,
) -> list[dict[str, Any]]:
    """Execute one scenario: one client, one gather, one JSON document."""

    async def execute() -> list[dict[str, Any]]:
        client, _http_client = create_gate_client(
            base_url=base_url, api_key=api_key, timeout=timeout
        )
        try:
            return await execute_gate_entries(client, served_model_name, entries)
        finally:
            # AsyncOpenAI owns and closes the supplied httpx.AsyncClient.
            # Closing it again here is both unnecessary and error-prone:
            # httpx.AsyncClient exposes aclose(), not close().
            await client.close()

    outputs = asyncio.run(execute())
    write_gate_document(
        document_dir,
        scenario,
        model_path=model_path,
        served_model_name=served_model_name,
        port=port,
        entries=entries,
        outputs=outputs,
    )
    return outputs

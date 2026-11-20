# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared, CPU-testable contracts for the Qwen3.6 adaptation Gate."""

from __future__ import annotations

import json
import os
import re
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


def safe_component(value: str) -> str:
    """Return one filesystem-safe path component."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return component or "unknown"


def gate_results_root(
    repo_root: Path,
    *,
    platform: str,
    device: str,
    model: str,
    case: str,
) -> Path:
    """Build a stable result root without allowing values to add path levels."""
    return (
        repo_root
        / "test-results"
        / "gate"
        / safe_component(platform)
        / safe_component(device)
        / safe_component(model)
        / safe_component(case)
    ).resolve()


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


def summarize_gate_progress(
    results_dir: Path, scenario: str | None = None
) -> dict[str, int | bool]:
    """Return non-asserting counts suitable for live CI progress logs."""
    selected = SCENARIOS
    if scenario is not None:
        selected = tuple(item for item in SCENARIOS if item[0] == scenario)
        if not selected:
            raise ValueError(f"Unknown adaptation Gate scenario: {scenario}")

    expected_scenarios = len(selected)
    expected_requests = sum(count for _, count in selected)
    expected_checks = expected_requests * len(QUALITY_CHECKS)
    documents = []
    for name, _ in selected:
        path = results_dir / f"{name}.json"
        if path.is_file():
            documents.append(json.loads(path.read_text(encoding="utf-8")))

    outputs = [
        output
        for document in documents
        for output in document.get("output", [])
        if isinstance(output, dict)
    ]
    checks = [
        passed for output in outputs for passed in (output.get("checks") or {}).values()
    ]
    checks_have_valid_shape = all(
        isinstance(output.get("checks"), dict)
        and set(output["checks"]) == QUALITY_CHECKS
        and all(type(value) is bool for value in output["checks"].values())
        for output in outputs
    )
    passed_requests = sum(output.get("passed") is True for output in outputs)
    passed_checks = sum(passed is True for passed in checks)
    return {
        "expected_scenarios": expected_scenarios,
        "observed_scenarios": len(documents),
        "expected_requests": expected_requests,
        "observed_requests": len(outputs),
        "passed_requests": passed_requests,
        "expected_checks": expected_checks,
        "observed_checks": len(checks),
        "passed_checks": passed_checks,
        "passed": (
            len(documents) == expected_scenarios
            and len(outputs) == expected_requests
            and passed_requests == expected_requests
            and len(checks) == expected_checks
            and passed_checks == expected_checks
            and checks_have_valid_shape
        ),
    }


def format_gate_progress(label: str, summary: Mapping[str, int | bool]) -> str:
    """Format Gate counts as one grep-friendly CI log line."""
    status = "PASS" if summary["passed"] else "FAIL"
    return (
        f"[Gate] {label}: {status}; "
        f"scenarios={summary['observed_scenarios']}/{summary['expected_scenarios']}; "
        f"requests={summary['passed_requests']}/{summary['expected_requests']} passed "
        f"({summary['observed_requests']} observed); "
        f"checks={summary['passed_checks']}/{summary['expected_checks']} passed "
        f"({summary['observed_checks']} observed)"
    )


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


def build_server_args(config: Any, image_dir: Path) -> list[str]:
    """Combine model and serving arguments, then allow the Gate fixtures."""
    serve = config.serve
    args = [*config.serve_args(**serve.extra_engine), *serve.extra_args]
    if any(
        arg == _LOCAL_MEDIA_FLAG or str(arg).startswith(_LOCAL_MEDIA_FLAG + "=")
        for arg in args
    ):
        raise ValueError(
            f"{_LOCAL_MEDIA_FLAG} must be supplied only by the adaptation Gate"
        )
    return [*args, _LOCAL_MEDIA_FLAG, str(image_dir.resolve())]


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

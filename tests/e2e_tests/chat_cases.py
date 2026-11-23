# Copyright (c) 2026 BAAI. All rights reserved.

"""Generic helpers for task-tagged entries in ``serve.chat_cases``.

Chat cases may opt into the quality-gate execution path by setting
``task: serving`` or ``task: concurrent``.  This module only routes,
groups, builds, and validates those entries: it must never embed concrete
test prompts, expected answers, or image tables.  The sole source of
request data is the ``serve.chat_cases`` list in each model YAML, and
runners built on these helpers must not import the tools case pack at
runtime.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from tests.e2e_tests.adaptation_gate.gate_contract import apply_local_media_flag

TASKS = ("serving", "concurrent")
# Chat cases without a ``task`` field keep the legacy serving behaviour.
LEGACY_TASK = "serving"

# Contract: scenario names, their owning task, and per-scenario request
# counts.  Scenario *contents* (prompts, expected answers, image files)
# live only in the model YAML; nothing here may duplicate them.
GATE_SCENARIO_CONTRACT: dict[str, tuple[str, int]] = {
    "text_single": ("serving", 1),
    "image_single": ("serving", 1),
    "text_concurrent_8": ("concurrent", 8),
    "image_concurrent_8": ("concurrent", 8),
    "mixed_concurrent_8": ("concurrent", 8),
}
CONCURRENT_REQUESTS_PER_SCENARIO = 8


def _case_name(case: dict[str, Any]) -> str:
    """Return the display name used in every validation error message."""
    return str(case.get("name", "<unnamed>"))


def _require_unique_names(chat_cases: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for case in chat_cases:
        name = _case_name(case)
        if name in seen:
            raise ValueError(f"Duplicate chat case name {name!r} in serve.chat_cases")
        seen.add(name)


def chat_cases_for_task(
    chat_cases: list[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    """Return the chat cases that belong to ``task``.

    Cases without a ``task`` field count as legacy serving cases so that
    existing model YAML keeps its behaviour.  An unknown ``task`` value on
    any case is rejected regardless of which task is being selected.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    selected: list[dict[str, Any]] = []
    for case in chat_cases:
        case_task = case.get("task", LEGACY_TASK)
        if case_task not in TASKS:
            raise ValueError(
                f"Chat case {_case_name(case)!r} has unknown task {case_task!r}; "
                f"expected one of {TASKS}"
            )
        if case_task == task:
            selected.append(case)
    return selected


def group_concurrent_cases(
    chat_cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group concurrent chat cases by scenario, preserving YAML order.

    Scenario keys appear in first-appearance order, and each group keeps
    the order of the ``serve.chat_cases`` list.  Every concurrent case
    must declare a non-empty ``scenario``, names must be unique across the
    whole list, and each scenario group must hold exactly eight explicit
    requests (a scenario selector entry does not satisfy the contract).
    """
    _require_unique_names(chat_cases)
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in chat_cases_for_task(chat_cases, "concurrent"):
        scenario = case.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError(
                f"Concurrent chat case {_case_name(case)!r} needs a non-empty scenario"
            )
        groups.setdefault(scenario, []).append(case)
    for scenario, cases in groups.items():
        if len(cases) != CONCURRENT_REQUESTS_PER_SCENARIO:
            raise ValueError(
                f"Concurrent scenario {scenario!r} must list exactly "
                f"{CONCURRENT_REQUESTS_PER_SCENARIO} explicit chat cases, "
                f"got {len(cases)}"
            )
    return groups


def resolve_case_image(case: dict[str, Any], repo_root: Path | str) -> Path:
    """Resolve one chat case ``image_path`` to a file inside the repository."""
    raw = case.get("image_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Chat case {_case_name(case)!r} needs a non-empty image_path")
    root = Path(repo_root).resolve()
    raw_path = Path(raw)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        candidate = candidate.resolve()
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Chat case {_case_name(case)!r} image_path escapes the repository "
            f"root: {raw!r}"
        ) from error
    if not candidate.is_file():
        raise ValueError(
            f"Chat case {_case_name(case)!r} image_path is not a file: {raw!r}"
        )
    return candidate


def build_chat_messages(
    case: dict[str, Any], repo_root: Path | str
) -> list[dict[str, Any]]:
    """Build OpenAI chat messages for one chat case.

    Text cases return a copy of their ``messages``.  Local image cases
    combine ``prompt`` with a ``file://`` URI built from ``image_path``;
    the path must exist, be a file, and stay inside the repository root.
    """
    if case.get("image_path"):
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Image chat case {_case_name(case)!r} needs a non-empty prompt"
            )
        image_uri = resolve_case_image(case, repo_root).as_uri()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ]
    messages = case.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"Chat case {_case_name(case)!r} needs non-empty messages "
            "(or image_path + prompt for an image case)"
        )
    return copy.deepcopy(messages)


def local_media_directory(
    cases: list[dict[str, Any]], repo_root: Path | str
) -> Path | None:
    """Return the single directory holding every case image, or ``None``.

    Used later to append ``--allowed-local-media-path`` exactly once.
    Refuses to silently pick one directory when cases span several.
    """
    directories: set[Path] = set()
    for case in cases:
        if not case.get("image_path"):
            continue
        directories.add(resolve_case_image(case, repo_root).parent)
    if not directories:
        return None
    if len(directories) > 1:
        listing = ", ".join(sorted(str(item) for item in directories))
        raise ValueError(
            "Local image cases must share one directory so a single "
            f"--allowed-local-media-path can cover them; got: {listing}"
        )
    return directories.pop()


def validate_chat_case_tasks(chat_cases: list[dict[str, Any]]) -> None:
    """Reject unknown ``task`` values anywhere in ``serve.chat_cases``."""
    for case in chat_cases:
        case_task = case.get("task", LEGACY_TASK)
        if case_task not in TASKS:
            raise ValueError(
                f"Chat case {_case_name(case)!r} has unknown task {case_task!r}; "
                f"expected one of {TASKS}"
            )


def resolve_execution_mode(
    chat_cases: list[dict[str, Any]],
    verification_configured: bool,
    task: str,
) -> str:
    """Return ``"gate"``, ``"verification"``, or ``"legacy"`` for one runner.

    Explicit ``task``-tagged cases for ``task`` win over the legacy
    verification pack; without either, runners keep their historical
    behaviour.  Unknown task values must be rejected by
    :func:`validate_chat_case_tasks` before calling this.
    """
    if any(case.get("task") == task for case in chat_cases):
        return "gate"
    if verification_configured:
        return "verification"
    return "legacy"


def validate_gate_case(case: dict[str, Any], task: str) -> None:
    """Structurally validate one explicit gate chat case for ``task``."""
    name = _case_name(case)
    if case.get("task") != task:
        raise ValueError(f"Gate chat case {name!r} must declare task: {task}")
    scenario = case.get("scenario")
    contract = (
        GATE_SCENARIO_CONTRACT.get(scenario) if isinstance(scenario, str) else None
    )
    if contract is None:
        raise ValueError(
            f"Gate chat case {name!r} references unknown scenario {scenario!r}; "
            f"expected one of {sorted(GATE_SCENARIO_CONTRACT)}"
        )
    expected_task, _required = contract
    if expected_task != task:
        raise ValueError(
            f"Gate chat case {name!r} assigns scenario {scenario!r} to task "
            f"{task!r}; the contract requires {expected_task!r}"
        )
    expected = case.get("expected")
    if (
        not isinstance(expected, list)
        or not expected
        or not all(isinstance(term, str) for term in expected)
    ):
        raise ValueError(
            f"Gate chat case {name!r} needs a non-empty expected list of strings"
        )
    for field_name in ("min_length", "max_tokens"):
        value = case.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Gate chat case {name!r} needs an integer {field_name} >= 1"
            )
    if case.get("image_path"):
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Image gate chat case {name!r} needs a non-empty prompt")
    else:
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Text gate chat case {name!r} needs non-empty messages")


def gate_entry(case: dict[str, Any], repo_root: Path | str) -> dict[str, Any]:
    """Build one execution entry: messages, grading thresholds, and name."""
    return {
        "name": case["name"],
        "scenario": case["scenario"],
        "messages": build_chat_messages(case, repo_root),
        "expected": list(case["expected"]),
        "min_length": int(case["min_length"]),
        "max_tokens": int(case["max_tokens"]),
    }


def build_request_body(
    messages: list[dict[str, Any]], max_tokens: int
) -> dict[str, Any]:
    """Raw ``/chat/completions`` JSON body for one gate request.

    Sampling is part of the gate contract: deterministic temperature and
    thinking disabled, regardless of platform ``serve`` sampling settings.
    """
    return {
        "messages": copy.deepcopy(messages),
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_stream_kwargs(
    messages: list[dict[str, Any]], max_tokens: int, model: str
) -> dict[str, Any]:
    """OpenAI SDK create() kwargs for one streaming gate request."""
    return {
        "model": model,
        "messages": copy.deepcopy(messages),
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def assemble_gate_extra_args(
    base_args: list[str],
    cases: list[dict[str, Any]],
    repo_root: Path | str,
) -> list[str]:
    """Append ``--allowed-local-media-path`` once when gate cases need images.

    ``base_args`` must already contain the ModelConfig-derived serving
    arguments; the media directory is derived from the cases themselves.
    """
    return apply_local_media_flag(
        list(base_args), local_media_directory(cases, repo_root)
    )


def validate_gate_chat_cases(
    chat_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a gate model's ``serve.chat_cases`` against the contract.

    The contract fixes scenario names, owning tasks, and per-scenario
    request counts (1/1/8/8/8 = 26).  Request contents are only checked
    structurally: unique names, expected/min_length/max_tokens present,
    messages for text cases, image_path + prompt for image cases.
    """
    _require_unique_names(chat_cases)
    counts: dict[str, int] = {}
    for case in chat_cases:
        name = _case_name(case)
        task = case.get("task")
        if task not in TASKS:
            raise ValueError(
                f"Gate chat case {name!r} needs an explicit task from {TASKS}"
            )
        scenario = case.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError(f"Gate chat case {name!r} needs a non-empty scenario")
        contract = GATE_SCENARIO_CONTRACT.get(scenario)
        if contract is None:
            raise ValueError(
                f"Gate chat case {name!r} references unknown scenario {scenario!r}; "
                f"expected one of {sorted(GATE_SCENARIO_CONTRACT)}"
            )
        expected_task, _required = contract
        if task != expected_task:
            raise ValueError(
                f"Gate chat case {name!r} assigns scenario {scenario!r} to task "
                f"{task!r}; the contract requires {expected_task!r}"
            )
        expected = case.get("expected")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(term, str) for term in expected)
        ):
            raise ValueError(
                f"Gate chat case {name!r} needs a non-empty expected list of strings"
            )
        for field_name in ("min_length", "max_tokens"):
            value = case.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Gate chat case {name!r} needs an integer {field_name} >= 1"
                )
        if case.get("image_path"):
            prompt = case.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Image gate chat case {name!r} needs a non-empty prompt"
                )
        else:
            messages = case.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Text gate chat case {name!r} needs non-empty messages"
                )
        counts[scenario] = counts.get(scenario, 0) + 1

    problems = []
    for scenario, (_task, required) in GATE_SCENARIO_CONTRACT.items():
        observed = counts.get(scenario, 0)
        if observed != required:
            problems.append(f"{scenario}: {observed} (contract requires {required})")
    if problems:
        raise ValueError(
            "Gate chat case counts do not satisfy the 26-request contract: "
            + "; ".join(problems)
        )
    return {"total": sum(counts.values()), "scenarios": counts}

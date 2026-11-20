# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for generic chat-case helpers and the Qwen3.6 YAML transcription.

The assertion cores are plain functions so they can also run without the
full pytest/conftest environment (``tests/conftest.py`` imports torch).
Only these tests may touch ``tools/adaptation-gate-cases``: production
helpers and runners must never import it at CI runtime.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import yaml

from tests.e2e_tests.chat_cases import (
    build_chat_messages,
    chat_cases_for_task,
    group_concurrent_cases,
    local_media_directory,
    validate_gate_chat_cases,
)
from tests.utils.model_config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_MODEL = "qwen3_6"
TEMPLATE_CASE = "27b_tp2_eager"
TOOLS_CASE_UTILS = REPO_ROOT / "tools/adaptation-gate-cases/case_utils.py"
TOOLS_IMAGE_DIR = REPO_ROOT / "tools/adaptation-gate-cases/images"

SCENARIO_ORDER = (
    "text_single",
    "image_single",
    "text_concurrent_8",
    "image_concurrent_8",
    "mixed_concurrent_8",
)

# Distinctive fragments of the gate prompts/answers/image names.  Used to
# prove that no production module secretly embeds the request table.
REQUEST_DATA_MARKERS = (
    "Compute 19",
    "capital of France",
    "opposite of hot",
    "one week",
    "chemical formula",
    "largest ocean",
    "multiplied by 6",
    "mixing blue and yellow",
    "quadrant colors",
    "three shapes",
    "How many triangles",
    "visible character",
    "red circle",
    "color is the triangle",
    "blue circles",
    "contain visible text",
    "Introduce large language models",
    "image_0",
)


def _expect_value_error(function, *args):
    try:
        function(*args)
    except ValueError as error:
        return error
    raise AssertionError(f"{function.__name__} did not raise ValueError")


def _template_chat_cases():
    config = ModelConfig.load(TEMPLATE_MODEL, TEMPLATE_CASE)
    return config.serve.chat_cases


def _all_gate_case_names():
    """Every qwen3_6 case YAML; each must carry the full 26-case inventory."""
    return sorted(
        path.stem for path in (REPO_ROOT / "tests/models/qwen3_6").glob("*.yaml")
    )


def _make_concurrent_scenario(scenario: str, prefix: str, count: int = 8):
    return [
        {
            "name": f"{prefix}-{index:02d}",
            "task": "concurrent",
            "scenario": scenario,
            "messages": [{"role": "user", "content": f"generic probe {index}"}],
            "expected": ["generic"],
            "min_length": 1,
            "max_tokens": 8,
        }
        for index in range(1, count + 1)
    ]


# ---------------------------------------------------------------------------
# Assertion cores
# ---------------------------------------------------------------------------


def _assert_helper_behavior():
    legacy = {
        "name": "legacy",
        "messages": [{"role": "user", "content": "hi"}],
    }
    serving_case = {
        "name": "serve-1",
        "task": "serving",
        "scenario": "text_single",
        "messages": [{"role": "user", "content": "hi"}],
        "expected": ["x"],
        "min_length": 1,
        "max_tokens": 8,
    }
    concurrent_case = {
        "name": "conc-1",
        "task": "concurrent",
        "scenario": "text_concurrent_8",
        "messages": [{"role": "user", "content": "hi"}],
        "expected": ["x"],
        "min_length": 1,
        "max_tokens": 8,
    }
    cases = [legacy, serving_case, concurrent_case]

    # Cases without task default to serving (backward compatibility).
    assert chat_cases_for_task(cases, "serving") == [legacy, serving_case]
    assert chat_cases_for_task(cases, "concurrent") == [concurrent_case]
    _expect_value_error(chat_cases_for_task, cases, "benchmark")
    _expect_value_error(
        chat_cases_for_task, [{"name": "bad", "task": "typo"}], "serving"
    )

    # Grouping preserves first-appearance scenario order and YAML order.
    text_group = _make_concurrent_scenario("text_concurrent_8", "t")
    image_group = _make_concurrent_scenario("image_concurrent_8", "i")
    interleaved = []
    for index in range(8):
        interleaved.append(text_group[index])
        interleaved.append(image_group[index])
    groups = group_concurrent_cases(interleaved + [legacy])
    assert list(groups) == ["text_concurrent_8", "image_concurrent_8"]
    assert [case["name"] for case in groups["text_concurrent_8"]] == [
        case["name"] for case in text_group
    ]
    assert all(case["task"] == "concurrent" for case in groups["image_concurrent_8"])

    _expect_value_error(
        group_concurrent_cases,
        [{"name": "bad", "task": "concurrent", "scenario": "", "messages": []}],
    )
    _expect_value_error(
        group_concurrent_cases,
        [{"name": "bad", "task": "concurrent", "messages": []}],
    )
    duplicate = dict(text_group[0])
    duplicate["messages"] = [{"role": "user", "content": "dup"}]
    _expect_value_error(group_concurrent_cases, text_group + [duplicate])
    _expect_value_error(group_concurrent_cases, text_group[:7])

    # file:// conversion for a repository image (path mechanics only).
    sample_image = sorted(TOOLS_IMAGE_DIR.glob("image_*.png"))[0]
    relative = sample_image.relative_to(REPO_ROOT).as_posix()
    image_case = {
        "name": "img",
        "task": "serving",
        "scenario": "image_single",
        "image_path": relative,
        "prompt": "generic probe",
        "expected": ["generic"],
        "min_length": 1,
        "max_tokens": 8,
    }
    messages = build_chat_messages(image_case, REPO_ROOT)
    assert messages[0]["content"][0] == {"type": "text", "text": "generic probe"}
    assert messages[0]["content"][1]["image_url"]["url"] == sample_image.as_uri()
    # Text cases pass their messages through unchanged.
    assert build_chat_messages(legacy, REPO_ROOT) == legacy["messages"]

    # Missing file, non-file path, and repository escape all fail loudly.
    missing = dict(image_case, image_path="tools/adaptation-gate-cases/images/nope.png")
    _expect_value_error(build_chat_messages, missing, REPO_ROOT)
    directory_case = dict(image_case, image_path="tools/adaptation-gate-cases/images")
    _expect_value_error(build_chat_messages, directory_case, REPO_ROOT)
    outside = dict(image_case, image_path="/etc/hostname")
    _expect_value_error(build_chat_messages, outside, REPO_ROOT)

    # Media directory: None without images, one shared directory, multi fails.
    assert local_media_directory([legacy], REPO_ROOT) is None
    assert local_media_directory([image_case], REPO_ROOT) == sample_image.parent
    text_file = REPO_ROOT / "tests" / "README.md"
    other_dir_case = dict(image_case, name="img2", image_path=text_file.as_posix())
    _expect_value_error(local_media_directory, [image_case, other_dir_case], REPO_ROOT)

    # build_chat_messages must not mutate the YAML case dict.
    assert image_case["image_path"] == relative


def _assert_template_contract():
    """Every qwen3_6 YAML must carry the explicit 26-case inventory."""
    for case_name in _all_gate_case_names():
        config = ModelConfig.load(TEMPLATE_MODEL, case_name)
        cases = config.serve.chat_cases
        summary = validate_gate_chat_cases(cases)
        assert summary["total"] == 26, case_name
        assert summary["scenarios"] == {
            "text_single": 1,
            "image_single": 1,
            "text_concurrent_8": 8,
            "image_concurrent_8": 8,
            "mixed_concurrent_8": 8,
        }, case_name
        assert len(chat_cases_for_task(cases, "serving")) == 2, case_name
        assert len(chat_cases_for_task(cases, "concurrent")) == 24, case_name
        names = [case["name"] for case in cases]
        assert len(names) == len(set(names)) == 26, case_name
        # The legacy verification compatibility path is fully removed.
        assert not hasattr(config, "verification"), case_name

    # A handful of entries cannot satisfy the explicit 26-request contract
    # (e.g. one entry per scenario instead of eight explicit requests).
    cases = ModelConfig.load(TEMPLATE_MODEL, TEMPLATE_CASE).serve.chat_cases
    selector_only = [
        case for case in cases if case.get("scenario") in ("text_single", "image_single")
    ] + [
        dict(case, name="selector-" + case["scenario"])
        for case in [cases[2], cases[10], cases[18]]
    ]
    assert len(selector_only) == 5
    _expect_value_error(validate_gate_chat_cases, selector_only)


def _load_tools_case_utils():
    """Import tools case_utils for the transcription check (tests only).

    A stub stands in for ``openai`` when the local environment lacks it:
    only the pure request builders run here, never the HTTP client.
    """
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("openai")

        class _StubAsyncOpenAI:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("HTTP client must not be constructed in tests")

        stub.AsyncOpenAI = _StubAsyncOpenAI
        sys.modules.setdefault("openai", stub)
    spec = importlib.util.spec_from_file_location(
        "_gate_case_utils_baseline", TOOLS_CASE_UTILS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_tools_request(request: dict) -> dict:
    image_path = request.get("image_path")
    return {
        "messages": request["messages"],
        "expected": list(request["required_terms"]),
        "min_length": int(request.get("min_length", 1)),
        "max_tokens": int(request.get("max_tokens", 128)),
        "image": Path(image_path).name if image_path else None,
    }


def _normalize_yaml_case(case: dict, repo_root: Path) -> dict:
    image_path = case.get("image_path")
    return {
        "messages": build_chat_messages(case, repo_root),
        "expected": list(case["expected"]),
        "min_length": int(case["min_length"]),
        "max_tokens": int(case["max_tokens"]),
        "image": Path(image_path).name if image_path else None,
    }


def _assert_tools_transcription_consistency():
    """Deep-equality between tools scenario_requests() and EVERY qwen3_6 YAML."""
    case_utils = _load_tools_case_utils()
    previous_max_tokens = os.environ.pop("MAX_TOKENS", None)
    try:
        baseline = {
            scenario: case_utils.scenario_requests(scenario)
            for scenario in SCENARIO_ORDER
        }
    finally:
        if previous_max_tokens is not None:
            os.environ["MAX_TOKENS"] = previous_max_tokens

    report = {}
    for case_name in _all_gate_case_names():
        cases = ModelConfig.load(TEMPLATE_MODEL, case_name).serve.chat_cases
        per_scenario = {}
        for scenario in SCENARIO_ORDER:
            yaml_cases = [case for case in cases if case.get("scenario") == scenario]
            tools_requests = baseline[scenario]
            assert len(yaml_cases) == len(tools_requests), (
                f"{case_name}/{scenario}: YAML has {len(yaml_cases)} cases, "
                f"tools defines {len(tools_requests)} requests"
            )
            for case, request in zip(yaml_cases, tools_requests):
                expected = _normalize_tools_request(request)
                actual = _normalize_yaml_case(case, REPO_ROOT)
                assert actual == expected, (
                    f"{case_name}: case {case['name']!r} ({scenario}) diverges "
                    f"from tools:\n  YAML : {actual}\n  tools: {expected}"
                )
            per_scenario[scenario] = len(yaml_cases)
        assert sum(per_scenario.values()) == 26, case_name
        report[case_name] = per_scenario
    assert sorted(report) == _all_gate_case_names()
    return report


def _assert_no_hidden_request_table():
    production_sources = (
        REPO_ROOT / "tests/e2e_tests/chat_cases.py",
        REPO_ROOT / "tests/e2e_tests/adaptation_gate/gate_contract.py",
    )
    for path in production_sources:
        text = path.read_text(encoding="utf-8")
        for marker in REQUEST_DATA_MARKERS:
            assert marker not in text, (
                f"{path.name} must not embed request data (found {marker!r})"
            )
        for forbidden in ("case_utils", "adaptation-gate-cases"):
            assert forbidden not in text, (
                f"{path.name} must not reference tools ({forbidden!r} found)"
            )


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


def test_helper_behavior():
    _assert_helper_behavior()


def test_template_contract():
    _assert_template_contract()


def test_tools_transcription_consistency():
    _assert_tools_transcription_consistency()


def test_no_hidden_request_table():
    _assert_no_hidden_request_table()

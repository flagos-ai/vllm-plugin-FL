# Copyright (c) 2025 BAAI. All rights reserved.

import pytest

from tests.utils.concurrent_cases import (
    DEFAULT_TEXT_CASES,
    matches_expected,
    normalize_text_cases,
)


def test_default_cases_are_short_completion_prompts():
    assert len(DEFAULT_TEXT_CASES) == 8
    for case in DEFAULT_TEXT_CASES:
        assert case["expected"]
        assert "?" not in case["text"]


def test_normalize_text_cases_accepts_strings_and_expected_string():
    cases = normalize_text_cases(
        [
            "free-form prompt",
            {"text": "The capital of France is", "expected": "Paris"},
        ]
    )

    assert cases == [
        {"text": "free-form prompt", "expected": []},
        {"text": "The capital of France is", "expected": ["Paris"]},
    ]


@pytest.mark.parametrize("output", ["Jupiter", "The answer is JUPITER."])
def test_matches_expected_is_case_insensitive(output):
    case = {"text": "The largest planet is", "expected": ["jupiter"]}

    assert matches_expected(case, output)


def test_matches_expected_accepts_unvalidated_custom_prompt():
    assert matches_expected({"text": "Tell me a story", "expected": []}, "Once")

# Copyright (c) 2025 BAAI. All rights reserved.

"""Prompt and validation helpers for the concurrent generation smoke test."""

from __future__ import annotations

from typing import Any


# ``LLM.generate`` receives raw completion prompts, not chat messages. Keep the
# expected answer at the start of each completion so reasoning models do not
# spend the smoke test's small token budget on an answer preamble.
DEFAULT_TEXT_CASES = [
    {"text": "The United States has exactly", "expected": ["50"]},
    {"text": "The capital of France is", "expected": ["paris"]},
    {
        "text": "The largest planet in the solar system is",
        "expected": ["jupiter"],
    },
    {
        "text": "The play Romeo and Juliet was written by",
        "expected": ["shakespeare"],
    },
    {"text": "The product of 17 and 13 is", "expected": ["221"]},
    {
        "text": "The three primary colors of light are",
        "expected": ["red", "green", "blue"],
    },
    {"text": "World War II ended in", "expected": ["1945"]},
    {
        "text": "Gravity is the force that causes objects with mass to",
        "expected": ["attract"],
    },
]


def normalize_text_cases(raw_cases: list[Any]) -> list[dict[str, Any]]:
    """Normalize string and mapping cases to one validation shape."""
    cases = []
    for item in raw_cases:
        if isinstance(item, str):
            cases.append({"text": item, "expected": []})
            continue
        expected = item.get("expected", [])
        if isinstance(expected, str):
            expected = [expected]
        cases.append({"text": item["text"], "expected": expected})
    return cases


def matches_expected(case: dict[str, Any], text: str) -> bool:
    """Return whether output contains any configured marker, case-insensitively."""
    expected = case["expected"]
    if not expected:
        return True
    lower = text.lower()
    return any(item.lower() in lower for item in expected)

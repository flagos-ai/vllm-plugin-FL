"""Explainable quality checks for adaptation Gate responses; standard library only."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

ORACLE_VERSION = "2.1-llm-concepts"
LLM_EXPLANATION_PROFILE = "llm-explanation-v1"
LLM_REQUIRED_TERMS = ("large language model", "training", "inference", "limitations")

# This is a bounded English concept matcher, not a general semantic evaluator.
# Each category needs an affirmative limitation statement, not a bare heading.
_CATEGORIES = {
    "factual_reliability": r"\b(?:hallucinat\w*|false information|incorrect (?:facts|answers|information)|fabricat\w*|invent\w* (?:facts|information))\b",
    "bias": r"\b(?:bias(?:es|ed)?|discrimination|stereotypes?)\b",
    "understanding_reasoning": r"\b(?:understanding|reasoning|reason|consciousness)\b",
    "knowledge_freshness": r"\b(?:outdated|out.of.date|knowledge cutoff|current information|recent events|real.time (?:knowledge|data|awareness))\b",
}
_POSITIVE_PATTERNS = {
    "factual_reliability": (
        r"\b(?:can|may|might|sometimes|often)\s+(?:(?!not\b|never\b)\w+\s+){0,4}(?:hallucinate|fabricate|invent)\b[^.!?;]{0,80}",
        r"\b(?:suffer(?:s)? from|(?:are|is) (?:prone|susceptible|vulnerable) to)\s+hallucinations?\b",
        r"\b(?:can|may|might)\s+(?:(?!not\b|never\b)\w+\s+){0,4}(?:produce|generate|present|give|contain)\s+[^.!?;]{0,60}\b(?:false|incorrect|fabricated|inaccurate)\s+(?:facts?|answers?|information)\b",
    ),
    "bias": (
        r"\bbias(?:es)?\s+(?:inherited|learned)\s+from\s+(?:their\s+)?(?:training\s+)?data\b",
        r"\b(?:can|may|might|sometimes|often)\s+(?:(?!not\b|never\b)\w+\s+){0,4}(?:reflect|reproduce|inherit|amplify|perpetuate|exhibit)\b[^.!?;]{0,80}\b(?:bias(?:es|ed)?|discrimination|stereotypes?)\b",
    ),
    "understanding_reasoning": (
        r"\b(?:understanding|reasoning)\s+(?:is|can be|may be|remains?)\s+(?:limited|unreliable|imperfect)\b",
        r"\b(?:lack(?:s)?|do not have|does not have)\s+(?:(?:true|real|genuine|reliable|human|deep|robust)\s+)*(?:understanding|reasoning|consciousness)\b",
        r"\b(?:have|has|demonstrate(?:s)?|suffer(?:s)? from)\s+(?:a\s+)?lack of\s+(?:(?:true|real|genuine|reliable|human|deep|robust)\s+)*(?:understanding|reasoning|consciousness)\b",
        r"\b(?:cannot reliably|cannot truly|do not truly|does not truly)\s+(?:understand|reason)\b",
    ),
    "knowledge_freshness": (
        r"\b(?:knowledge|information)\b[^.!?;]{0,40}\b(?:is|can be|may be|remains?)\s+(?:outdated|out.of.date|limited)\b",
        r"\b(?:cannot|do not|does not)\s+[^.!?;]{0,40}\b(?:access|know|cover)\b[^.!?;]{0,40}\b(?:current information|recent events|real.time (?:knowledge|data|awareness))\b",
        r"\b(?:have|has)\s+(?:a\s+)?knowledge cutoff\b",
    ),
}
_LIST_CONTEXT = re.compile(
    r"\b(?:limitations?|weaknesses?|drawbacks?|risks?)\s+"
    r"(?:such as|include|includes|involve|involves|persist,?\s+such as|remain,?\s+such as)\b[^.!?;]{0,240}|"
    # A comma-separated example needs a finite affirmative predicate; a bare
    # "Limitations, such as ..." heading is not explanatory evidence.
    r"\b(?:face(?:s)?|have|has)\s+(?:(?!no\b|not\b|never\b)\w+\s+){0,3}"
    r"(?:limitations?|weaknesses?|drawbacks?|risks?)\s*,\s*such as\b[^.!?;]{0,240}"
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|free of|cannot|can't|don't|doesn't)\b"
)
_DENIED_CONTEXT = re.compile(
    r"\b(?:not (?:true|correct|accurate)|no (?:evidence|proof|indication))\b"
    r"[^.!?;]{0,100}\bthat\b|"
    r"\b(?:neither|none of (?:these|those))\b[^.!?;]{0,100}"
    r"\b(?:problems?|risks?|limitations?|appl\w*|affect\w*)\b|"
    r"\b(?:myth|untrue|false claim)\b[^.!?;]{0,60}\bthat\b"
)


def normalized_text(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    number_words = {
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
    }
    for word, digit in number_words.items():
        normalized = re.sub(rf"\b{word}\b", digit, normalized)
    return normalized


def has_repeated_phrase(text: str) -> bool:
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


def has_repeated_word_run(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return any(
        tokens[index] == tokens[index + 1] == tokens[index + 2]
        for index in range(len(tokens) - 2)
    )


def llm_explanation_evidence(text: str) -> dict[str, Any]:
    """Return reviewable spans in normalized text for two distinct limitations.

    A noun list is accepted only in a statement such as "limitations include ...".
    Explicitly negated risk statements do not count. The limitation evidence must
    follow the training and inference anchors to satisfy the ordered explanation.
    """
    normalized = normalized_text(text)
    evidence: dict[tuple[str, int, int], dict[str, Any]] = {}
    for sentence in re.finditer(r"[^.!?;]+", normalized):
        body = sentence.group()
        if _DENIED_CONTEXT.search(body):
            continue
        for category, topic in _CATEGORIES.items():
            candidates = list(_LIST_CONTEXT.finditer(body))
            candidates += [
                match
                for pattern in _POSITIVE_PATTERNS[category]
                for match in re.finditer(pattern, body)
            ]
            for match in sorted(candidates, key=lambda item: item.start()):
                if not re.search(topic, match.group()):
                    continue
                # "never suffer from hallucinations" and "no limitations include"
                # must not become affirmative evidence by matching a later verb.
                before = body[: match.start()]
                prefix = " ".join(re.findall(r"[\w']+", before)[-3:])
                prefix = re.sub(r"\bnot only\b", "", prefix)
                if _NEGATION.search(prefix):
                    continue
                # A list denying any listed problem is not limitation evidence.
                if match.re is _LIST_CONTEXT and _NEGATION.search(match.group()):
                    continue
                start, end = (
                    sentence.start() + match.start(),
                    sentence.start() + match.end(),
                )
                evidence[(category, start, end)] = {
                    "category": category,
                    "start": start,
                    "end": end,
                    "text": match.group(),
                }
    positions = []
    cursor = 0
    for term in LLM_REQUIRED_TERMS[:3]:
        position = normalized.find(term, cursor)
        positions.append(position)
        if position >= 0:
            cursor = position + len(term)
    categories = {row["category"] for row in evidence.values()}
    ordered_categories = sorted(
        {
            row["category"]
            for row in evidence.values()
            if positions[-1] >= 0 and row["start"] >= cursor
        }
    )
    return {
        "profile": LLM_EXPLANATION_PROFILE,
        "offsets": "normalized_text",
        "anchor_terms": list(LLM_REQUIRED_TERMS[:3]),
        "anchor_positions": positions,
        "limitations": sorted(evidence.values(), key=lambda row: row["start"]),
        "distinct_categories": sorted(categories),
        "ordered_categories": ordered_categories,
        "semantics": all(position >= 0 for position in positions)
        and len(categories) >= 2,
        "ordered": all(position >= 0 for position in positions)
        and positions == sorted(positions)
        and len(ordered_categories) >= 2,
    }


def quality_checks(
    text: str,
    required_terms: list[str],
    min_length: int = 1,
    *,
    semantic_profile: str | None = None,
) -> dict[str, bool]:
    normalized = normalized_text(text)
    positions: list[int] = []
    cursor = 0
    for term in required_terms:
        position = normalized.find(term.lower(), cursor)
        positions.append(position)
        if position >= 0:
            cursor = position + len(term)
    semantics = all(position >= 0 for position in positions)
    order = positions == sorted(positions) and semantics
    if semantic_profile is not None:
        if (
            semantic_profile != LLM_EXPLANATION_PROFILE
            or tuple(required_terms) != LLM_REQUIRED_TERMS
        ):
            raise ValueError("Unknown or mismatched Gate semantic profile")
        evidence = llm_explanation_evidence(text)
        semantics, order = evidence["semantics"], evidence["ordered"]
    mojibake_markers = (
        "\ufffd",
        chr(0x00C3),
        chr(0x00C2),
        chr(0x00E2) + chr(0x20AC),
        chr(0x00EF) + chr(0x00BF) + chr(0x00BD),
    )
    return {
        "non_empty": bool(normalized),
        "minimum_length": len(text.strip()) >= min_length,
        "expected_semantics": semantics,
        "expected_order": order,
        "no_bang_triplet": "!!!" not in text,
        "no_mojibake": not any(marker in text for marker in mojibake_markers),
        "no_control_characters": not any(
            ord(char) < 32 and char not in "\n\r\t" for char in text
        ),
        "no_long_character_run": re.search(r"([^\s])\1{7,}", text) is None,
        "no_repeated_word_run": not has_repeated_word_run(text),
        "no_repeated_phrase": not has_repeated_phrase(text),
    }

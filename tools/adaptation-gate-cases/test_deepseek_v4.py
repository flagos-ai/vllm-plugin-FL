# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""DS text-only service gate for release/0.2; connect to an existing endpoint."""

import concurrent.futures
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pytest
import requests

TASKS = [
    ("Compute 19 + 23. Reply with the number only.", "42"),
    ("Name the capital of France. Reply with the city only.", "paris"),
    ("Give the opposite of hot. Reply with one word.", "cold"),
    ("How many days are in one week? Reply with the number only.", "7"),
    ("Give the chemical formula for water. Reply with the formula only.", "h2o"),
    ("Name Earth's largest ocean. Reply with the ocean name only.", "pacific"),
    ("Compute 5 multiplied by 6. Reply with the number only.", "30"),
    (
        "What color results from mixing blue and yellow paint? Reply with one word.",
        "green",
    ),
]


def quality(text, required, minimum):
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    words = re.findall(r"[a-z0-9]+", normalized)
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
    return {
        "minimum_length": len(text.strip()) >= minimum,
        "expected_semantics": all(term in normalized for term in required),
        "no_mojibake": "\ufffd" not in text,
        "no_bang_triplet": "!!!" not in text,
        "no_control_characters": not any(
            ord(c) < 32 and c not in "\n\r\t" for c in text
        ),
        "no_long_character_run": re.search(r"(\S)\1{7,}", text) is None,
        "no_repeated_word_run": not any(
            a == b == c for a, b, c in zip(words, words[1:], words[2:])
        ),
        "no_repeated_phrase": not any(
            words[i : i + n] == words[i + n : i + 2 * n] == words[i + 2 * n : i + 3 * n]
            for n in range(2, 13)
            for i in range(len(words) - 3 * n + 1)
        ),
    }


def request_case(prompt, required, minimum=1, max_tokens=128):
    payload = {
        "model": os.environ["SERVED_MODEL_NAME"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 1234,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.perf_counter()
    response = requests.post(
        os.environ["BASE_URL"].rstrip("/") + "/chat/completions",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    text = choice["message"].get("content") or ""
    checks = quality(text, required, minimum)
    checks["not_truncated"] = choice.get("finish_reason") != "length"
    return {
        "input": payload,
        "output": data,
        "checks": checks,
        "seconds": time.perf_counter() - started,
        "passed": all(checks.values()),
    }


def run_case(name, cases):
    # A service or transport exception fails the test; it is never retried.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(request_case, *case) for case in cases]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                results.append(
                    {"passed": False, "error": f"{type(error).__name__}: {error}"}
                )
    destination = (
        Path(os.environ["RESULTS_DIR"])
        / os.environ["EXECUTION_MODE"]
        / (name + ".json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"case": name, "results": results}, indent=2) + "\n"
    )
    assert all(item["passed"] for item in results), f"Inspect {destination}"


@pytest.fixture(scope="module", autouse=True)
def settings():
    for name in ("BASE_URL", "SERVED_MODEL_NAME", "RESULTS_DIR", "EXECUTION_MODE"):
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is required")
    if os.environ["EXECUTION_MODE"] not in ("eager", "decode_graph"):
        raise ValueError("EXECUTION_MODE must be eager or decode_graph")


def test_text_single():
    run_case(
        "text_single",
        [
            (
                (
                    "Introduce large language models in one paragraph of 80-100 words. Explain "
                    "training, inference, capabilities and limitations, using at least 256 characters."
                ),
                ["large language model", "training", "inference", "limitations"],
                256,
                1536,
            )
        ],
    )


def test_text_concurrent_8():
    run_case("text_concurrent_8", [(q, [answer]) for q, answer in TASKS])


def long_case(repeats, marker):
    padding = "Background record: a brown fox crossed a quiet road.\n" * repeats
    return (
        padding + f"Final record: verification code is {marker}. "
        "Return the verification code only.",
        [marker.lower()],
        1,
        128,
    )


def test_text_long():
    # Approximately 16K prompt tokens; actual token count remains in the API output.
    run_case("text_long", [long_case(1500, "DsvLong42")])


def test_mixed_lengths_8():
    cases = [
        long_case(n, f"DsvMixed{i}Z")
        for i, n in enumerate([0, 1, 20, 100, 300, 500, 1000, 1500])
    ]
    run_case("mixed_lengths_8", cases)


def test_decode_buckets_64():
    # Distinct requests prevent prefix-only smoke coverage. The API outputs and
    # server capture log together establish the graph/metadata checks.
    cases = [
        (f"Verification code: DsvBatch{i}Z. Return this code only.", [f"dsvbatch{i}z"])
        for i in range(64)
    ]
    run_case("decode_buckets_64", cases)

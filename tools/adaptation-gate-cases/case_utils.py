"""Shared requests, checks, and JSON reporting for adaptation gate tests."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"

TEXT_TASKS = (
    ("Compute 19 + 23. Reply with the number only.", ("42",)),
    ("Name the capital of France. Reply with the city only.", ("paris",)),
    ("Give the opposite of hot. Reply with one word.", ("cold",)),
    ("How many days are in one week? Reply with the number only.", ("7",)),
    ("Give the chemical formula for water. Reply with the formula only.", ("h2o",)),
    ("Name Earth's largest ocean. Reply with the ocean name only.", ("pacific",)),
    ("Compute 5 multiplied by 6. Reply with the number only.", ("30",)),
    (
        "What color results from mixing blue and yellow paint? Reply with one word.",
        ("green",),
    ),
)

LLM_DETAILS_TASK = (
    (
        "Introduce large language models (LLMs) in detail. Explain what they are, "
        "how training and inference work, typical capabilities, limitations, and "
        "responsible use. Write a coherent response of at least 256 characters. "
        "Do not repeat words or phrases unnecessarily."
    ),
    ("large language model", "training", "inference", "limitations"),
)

IMAGE_TASKS = (
    (
        (
            "Name the four quadrant colors in this order: top-left, top-right, "
            "bottom-left, bottom-right. Reply with color names only."
        ),
        ("red", "green", "blue", "yellow"),
    ),
    (
        (
            "Name the four quadrant colors in this order: top-left, top-right, "
            "bottom-left, bottom-right. Reply with color names only."
        ),
        ("blue", "yellow", "red", "green"),
    ),
    (
        "Name the three shapes from left to right. Reply with shape names only.",
        ("circle", "square", "triangle"),
    ),
    (
        "How many triangles are in the image? Reply with the number only.",
        ("3",),
    ),
    (
        (
            "Read the two words. The second word has four letters. Copy "
            "every visible character exactly and reply with the text only."
        ),
        ("hello", "vllm"),
    ),
    (
        "What shape is inside the red circle? Reply with the shape name only.",
        ("square",),
    ),
    (
        "What color is the triangle? Reply with the color name only.",
        ("yellow",),
    ),
    (
        "How many blue circles are in the image? Reply with the number only.",
        ("4",),
    ),
)


def text_request(request_id: str, index: int) -> dict[str, Any]:
    prompt, required = TEXT_TASKS[index]
    return {
        "request_id": request_id,
        "kind": "text",
        "messages": [{"role": "user", "content": prompt}],
        "required_terms": list(required),
    }


def llm_details_request() -> dict[str, Any]:
    prompt, required = LLM_DETAILS_TASK
    return {
        "request_id": "text-llm-details",
        "kind": "text",
        "messages": [{"role": "user", "content": prompt}],
        "required_terms": list(required),
        "min_length": 256,
        "max_tokens": 512,
    }


def image_request(
    request_id: str,
    index: int,
    task: tuple[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    prompt, required = task or IMAGE_TASKS[index]
    image_path = IMAGE_DIR / f"image_{index + 1:02d}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing test image: {image_path}")
    return {
        "request_id": request_id,
        "kind": "image",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
                ],
            }
        ],
        "image_path": str(image_path),
        "required_terms": list(required),
    }


def scenario_requests(scenario: str) -> list[dict[str, Any]]:
    if scenario == "text_single":
        return [llm_details_request()]
    if scenario == "text_concurrent_8":
        return [text_request(f"text-{index + 1:02d}", index) for index in range(8)]
    if scenario == "image_single":
        return [image_request("image-05", 4)]
    if scenario == "image_concurrent_8":
        requests = [
            image_request(f"image-{index + 1:02d}", index) for index in range(8)
        ]
        requests[4] = image_request(
            "image-05",
            4,
            ("Does this image contain visible text? Reply yes or no.", ("yes",)),
        )
        return requests
    if scenario == "mixed_concurrent_8":
        requests = [text_request(f"text-{index + 1:02d}", index) for index in range(4)]
        requests += [
            image_request(f"image-{index + 1:02d}", index) for index in range(4)
        ]
        return requests
    raise ValueError(f"Unknown scenario: {scenario}")


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


def quality_checks(
    text: str, required_terms: list[str], min_length: int = 1
) -> dict[str, bool]:
    normalized = normalized_text(text)
    positions: list[int] = []
    cursor = 0
    for term in required_terms:
        position = normalized.find(term.lower(), cursor)
        positions.append(position)
        if position >= 0:
            cursor = position + len(term)
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
        "expected_semantics": all(position >= 0 for position in positions),
        "expected_order": positions == sorted(positions)
        and all(position >= 0 for position in positions),
        "no_bang_triplet": "!!!" not in text,
        "no_mojibake": not any(marker in text for marker in mojibake_markers),
        "no_control_characters": not any(
            ord(char) < 32 and char not in "\n\r\t" for char in text
        ),
        "no_long_character_run": re.search(r"([^\s])\1{7,}", text) is None,
        "no_repeated_word_run": not has_repeated_word_run(text),
        "no_repeated_phrase": not has_repeated_phrase(text),
    }


def resolve_served_model_name() -> str:
    served_model_name = os.environ.get("SERVED_MODEL_NAME")
    if served_model_name:
        return served_model_name
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        return model_path
    raise RuntimeError("SERVED_MODEL_NAME or MODEL_PATH is required")


async def execute_request(
    client: AsyncOpenAI,
    request: dict[str, Any],
    served_model_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completion = await client.chat.completions.create(
            model=served_model_name,
            messages=request["messages"],
            max_tokens=int(
                request.get("max_tokens", os.environ.get("MAX_TOKENS", "128"))
            ),
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = completion.choices[0].message.content or ""
        checks = quality_checks(
            text,
            request["required_terms"],
            min_length=int(request.get("min_length", 1)),
        )
        return {
            "request_id": request["request_id"],
            "kind": request["kind"],
            "latency_seconds": round(time.perf_counter() - started, 3),
            "text": text,
            "checks": checks,
            "passed": all(checks.values()),
            "response": completion.model_dump(mode="json"),
        }
    except Exception as error:
        return {
            "request_id": request["request_id"],
            "kind": request["kind"],
            "latency_seconds": round(time.perf_counter() - started, 3),
            "text": "",
            "checks": {},
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.rstrip("/").split("/")[-1])
    return slug.strip("-") or "model"


def compact_text(value: str, limit: int = 300) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3] + "..."


def failure_message(
    scenario: str,
    served_model_name: str,
    port: str,
    destination: Path,
    outputs: list[dict[str, Any]],
) -> str:
    failed_outputs = [output for output in outputs if not output["passed"]]
    base_url = os.environ.get("BASE_URL", f"http://127.0.0.1:{port}/v1")
    lines = [
        "Adaptation gate failed",
        f"  Scenario: {scenario}",
        f"  Model: {served_model_name}",
        f"  Endpoint: {base_url}",
        f"  Result: {destination}",
        f"  Failed requests: {len(failed_outputs)}/{len(outputs)}",
    ]
    for output in failed_outputs:
        request_id = output["request_id"]
        if error := output.get("error"):
            lines.append(f"    - {request_id}: {compact_text(error)}")
            continue
        failed_checks = [
            name for name, passed in output.get("checks", {}).items() if not passed
        ]
        detail = f"failed checks: {', '.join(failed_checks) or 'unknown'}"
        if text := compact_text(output.get("text", ""), limit=160):
            detail += f"; output: {text!r}"
        lines.append(f"    - {request_id}: {detail}")
    return "\n".join(lines)


async def execute_scenario(
    requests: list[dict[str, Any]], served_model_name: str
) -> list[dict[str, Any]]:
    port = os.environ["PORT"]
    client = AsyncOpenAI(
        api_key=os.environ.get("API_KEY", "EMPTY"),
        base_url=os.environ.get("BASE_URL", f"http://127.0.0.1:{port}/v1"),
        timeout=float(os.environ.get("REQUEST_TIMEOUT", "300")),
    )
    try:
        return await asyncio.gather(
            *(
                execute_request(client, request, served_model_name)
                for request in requests
            )
        )
    finally:
        await client.close()


def run_case(scenario: str) -> None:
    served_model_name = resolve_served_model_name()
    model_path = os.environ.get("MODEL_PATH")
    port = os.environ.get("PORT")
    if not port:
        raise RuntimeError("PORT is required; use run_test.sh or set it directly")
    model_label = model_path or served_model_name
    requests = scenario_requests(scenario)
    started_at = datetime.now(timezone.utc)
    outputs = asyncio.run(execute_scenario(requests, served_model_name))
    passed_count = sum(bool(output["passed"]) for output in outputs)
    document = {
        "schema_version": 1,
        "case": {
            "scenario": scenario,
            "model_path": model_path,
            "served_model_name": served_model_name,
            "port": port,
            "concurrency": len(requests),
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input": requests,
        "output": outputs,
        "summary": {
            "request_count": len(requests),
            "passed_count": passed_count,
            "failed_count": len(requests) - passed_count,
            "passed": passed_count == len(requests),
        },
    }
    results_root = Path(os.environ.get("RESULTS_DIR", str(ROOT / "results")))
    destination = (
        results_root
        / safe_slug(model_label)
        / f"port-{safe_slug(port)}"
        / f"{scenario}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    if not document["summary"]["passed"]:
        message = failure_message(
            scenario, served_model_name, port, destination, outputs
        )
        print(message, file=sys.stderr, flush=True)
        raise AssertionError(f"Quality gate failed; inspect {destination}")

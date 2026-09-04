#!/usr/bin/env python3
"""Run a synchronized concurrent completion batch with exact token counts."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SOURCE_TEXT = (
    "A distributed inference service receives varied technical questions. "
    "Explain the design, verify assumptions, compare alternatives, and provide "
    "a precise implementation with deterministic validation steps. "
)


def post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error


def build_prompt(base_url: str, model: str, target_tokens: int) -> list[int]:
    repetitions = 32
    while True:
        result = post_json(
            f"{base_url}/tokenize",
            {
                "model": model,
                "prompt": SOURCE_TEXT * repetitions,
                "add_special_tokens": True,
            },
            300,
        )
        tokens = result.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"/tokenize did not return a token list: {result}")
        if len(tokens) >= target_tokens:
            return [int(token) for token in tokens[:target_tokens]]
        repetitions *= 2
        if repetitions > 1024:
            raise RuntimeError("unable to construct the requested prompt length")


def run_batch(
    base_url: str,
    request_body: dict[str, Any],
    concurrency: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    barrier = threading.Barrier(concurrency)

    def send(index: int) -> dict[str, Any]:
        barrier.wait()
        started = time.monotonic()
        response = post_json(f"{base_url}/v1/completions", request_body, timeout)
        return {
            "index": index,
            "client_latency_seconds": time.monotonic() - started,
            "response": response,
        }

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send, index) for index in range(concurrency)]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["index"])
    return rows, time.monotonic() - started


def validate_usage(
    rows: list[dict[str, Any]], input_tokens: int, output_tokens: int
) -> None:
    errors: list[str] = []
    for row in rows:
        usage = row["response"].get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens != input_tokens or completion_tokens != output_tokens:
            errors.append(
                f"request {row['index']}: prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}"
            )
    if errors:
        raise RuntimeError("unexpected token counts:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--prompt-output", type=Path)
    parser.add_argument("--prompt-input", type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = str(config["model"])
    concurrency = int(config["concurrency"])
    input_tokens = int(config["input_tokens"])
    output_tokens = int(config["output_tokens"])
    if args.prompt_input:
        prompt_tokens = json.loads(args.prompt_input.read_text(encoding="utf-8"))
    else:
        prompt_tokens = build_prompt(args.base_url, model, input_tokens)
    if len(prompt_tokens) != input_tokens:
        raise RuntimeError(
            f"prompt has {len(prompt_tokens)} tokens, expected {input_tokens}"
        )
    if args.prompt_output:
        args.prompt_output.write_text(
            json.dumps(prompt_tokens, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    request_body = {
        "model": model,
        "prompt": prompt_tokens,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "seed": int(config.get("seed", 0)),
    }
    rows, batch_seconds = run_batch(
        args.base_url, request_body, concurrency, args.timeout_seconds
    )
    validate_usage(rows, input_tokens, output_tokens)
    args.responses.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    latencies = [float(row["client_latency_seconds"]) for row in rows]
    metrics = {
        "model": model,
        "concurrency": concurrency,
        "input_tokens_per_request": input_tokens,
        "output_tokens_per_request": output_tokens,
        "request_count": len(rows),
        "total_input_tokens": input_tokens * len(rows),
        "total_output_tokens": output_tokens * len(rows),
        "batch_wall_time_seconds": batch_seconds,
        "min_request_latency_seconds": min(latencies),
        "max_request_latency_seconds": max(latencies),
        "avg_request_latency_seconds": sum(latencies) / len(latencies),
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

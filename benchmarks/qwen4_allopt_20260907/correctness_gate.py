#!/usr/bin/env python3
"""Small deterministic API smoke gate; not a model numerical-correctness test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any
import urllib.error
import urllib.request


def request_json(url: str, payload: dict[str, Any] | None = None) -> tuple[int, bytes]:
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()
    except urllib.error.URLError:
        return 0, b""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "19821")))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "Qwen3.8-Flash-Next"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    errors: list[str] = []
    result: dict[str, Any] = {
        "gate_type": "api_smoke",
        "model": args.model,
        "base_url": base,
        "models_http_status": None,
        "completion_http_status": None,
    }

    models_status, models_body = request_json(f"{base}/v1/models")
    result["models_http_status"] = models_status
    if models_status != 200:
        errors.append(f"/v1/models returned HTTP {models_status}")
    else:
        try:
            models = json.loads(models_body)
            ids = [row.get("id") for row in models.get("data", []) if isinstance(row, dict)]
            result["served_model_ids"] = ids
            if args.model not in ids:
                errors.append(f"served model name absent from /v1/models: {args.model}")
        except (TypeError, ValueError):
            errors.append("/v1/models was not valid JSON")

    prompt = "Reply with exactly OK."
    request_payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 0,
        "seed": 12345,
        "ignore_eos": True,
        "stream": False,
    }
    completion_status, completion_body = request_json(
        f"{base}/v1/completions", request_payload
    )
    result["completion_http_status"] = completion_status
    result["completion_body_sha256"] = hashlib.sha256(completion_body).hexdigest()
    result["completion_body_bytes"] = len(completion_body)
    if completion_status != 200:
        errors.append(f"/v1/completions returned HTTP {completion_status}")
    else:
        try:
            completion = json.loads(completion_body)
            choices = completion.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                errors.append("completion has no first choice")
            else:
                text = choices[0].get("text")
                result["choice_text_bytes"] = len(text.encode("utf-8")) if isinstance(text, str) else 0
                if not isinstance(text, str) or not text:
                    errors.append("completion first choice is empty")
            usage = completion.get("usage")
            if isinstance(usage, dict):
                result["usage"] = {
                    key: usage.get(key)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
        except (TypeError, ValueError):
            errors.append("completion was not valid JSON")

    result["status"] = "pass" if not errors else "fail"
    result["errors"] = errors
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if errors:
        print("API smoke gate: FAIL")
        return 1
    print("API smoke gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2026 FlagOS Contributors
"""Collect one image05 reproduction from an already running local vLLM service.

This stdlib-only script was prepared offline and has not been GPU-rerun.
Default baseline preserves historical R1: file URI, exact shared Gate image05
prompt, temperature0, disabled thinking, max128, and no token-ID overrides.
Explicit --diagnostic-token-ids instead matches R2's diagnostic parameters:
max8, return_token_ids=True, skip_special_tokens=False. R1 observed empty128;
R2 separately observed248320 eight times. These historical observations are
not claimed as results of this newly organized script. --image-transport=data-url
is an additional portable variant, also not GPU-rerun; PNG SHA256 is recorded.

No server/device/package lifecycle operations are performed. Exit1 means an image
symptom was observed; exit2 means identity, transport or response collection failed.
This is a diagnostic, not the complete shared quality/performance Gate.
"""

import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

IMAGE_PROMPT = (
    "Read the two words. The second word has four letters. Copy "
    "every visible character exactly and reply with the text only."
)
TEXT_PROMPT = "Compute 19 + 23. Reply with the number only."


def local_endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/v1"}
        or parsed.port is None
        or not 1 <= parsed.port <= 65535
    ):
        raise ValueError(
            "Endpoint must be an explicit localhost HTTP(S) host:port[/v1]"
        )
    return f"{parsed.scheme}://{parsed.netloc}/v1"


def image_payload(
    served_name, image_path, image_bytes, transport="file", diagnostic=False
):
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("--image must be the repository image05 PNG")
    if transport == "file":
        url = Path(image_path).resolve().as_uri()
    elif transport == "data-url":
        url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    else:
        raise ValueError("Image transport must be file or data-url")
    payload = {
        "model": served_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 8 if diagnostic else 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if diagnostic:
        payload.update(return_token_ids=True, skip_special_tokens=False)
    return payload


def verify_model(document, model, served_name):
    cards = [row for row in document.get("data", []) if row.get("id") == served_name]
    if len(cards) != 1 or cards[0].get("root") != model:
        raise ValueError(
            "/v1/models does not uniquely match both --served-name and --model root"
        )


def vocabulary_size(model, explicit):
    if explicit is not None:
        if type(explicit) is not int or explicit <= 0:
            raise ValueError("--vocab-size must be a positive integer")
        return explicit
    config = Path(model) / "config.json"
    if not config.is_file():
        return None
    document = json.loads(config.read_text(encoding="utf-8"))
    value = document.get("text_config", {}).get(
        "vocab_size", document.get("vocab_size")
    )
    if type(value) is not int or value <= 0:
        raise ValueError(
            "Local model config has no positive vocab_size; pass --vocab-size"
        )
    return value


def completion_summary(
    document, vocab_size, require_ids=False, expected_terms=("hello", "vllm")
):
    choices = document.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Expected exactly one completion choice")
    choice = choices[0]
    ids = choice.get("token_ids")
    if (require_ids and ids is None) or (
        ids is not None
        and (not isinstance(ids, list) or any(type(token) is not int for token in ids))
    ):
        raise ValueError("Expected the official vLLM choice.token_ids integer list")
    text = choice.get("message", {}).get("content") or ""
    if not isinstance(text, str):
        raise ValueError("Expected a text completion")
    invalid = [
        token
        for token in (ids or [])
        if token < 0 or (vocab_size is not None and token >= vocab_size)
    ]
    normalized = text.casefold()
    cursor, positions = 0, []
    for term in expected_terms:
        position = normalized.find(term.casefold(), cursor)
        positions.append(position)
        if position >= 0:
            cursor = position + len(term)
    return {
        "content": text,
        "empty_content": not text.strip(),
        "finish_reason": choice.get("finish_reason"),
        "usage": document.get("usage"),
        "token_ids": ids,
        "token_ids_available": ids is not None,
        "vocab_size": vocab_size,
        "upper_token_bound_checked": ids is not None and vocab_size is not None,
        "invalid_token_ids": invalid if ids is not None else None,
        "expected_terms": list(expected_terms),
        "expected_words_in_order": all(position >= 0 for position in positions),
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def capture(url, payload, timeout, output, label):
    """Preserve exact response body bytes, including HTTP error bodies."""
    encoded = None if payload is None else json.dumps(payload).encode("utf-8")
    if payload is not None:
        (output / f"{label}.request.json").write_bytes(encoded)
    request = Request(url, data=encoded, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        body = response.read()
        status = response.status
        headers = dict(response.headers.items())
    (output / f"{label}.response.body").write_bytes(body)
    metadata = {
        "url": url,
        "http_status": status,
        "headers": headers,
        "seconds": time.monotonic() - started,
    }
    (output / f"{label}.http.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if status != 200:
        raise ValueError(f"{label}: HTTP {status}; raw body was saved")
    return json.loads(body)


def run(args):
    endpoint = local_endpoint(args.endpoint)
    if not args.model.strip() or not args.served_name.strip():
        raise ValueError("Model root and served name must be nonempty")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    image = Path(args.image).read_bytes()
    payload = image_payload(
        args.served_name,
        args.image,
        image,
        args.image_transport,
        args.diagnostic_token_ids,
    )
    vocab = vocabulary_size(args.model, args.vocab_size)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "preparation_validation": "Offline CPU only; this derived variant was not GPU-rerun during preparation",
        "mode": "token-diagnostic-r2" if args.diagnostic_token_ids else "baseline-r1",
        "image_transport": args.image_transport,
        "endpoint": endpoint,
        "model_root": args.model,
        "served_name": args.served_name,
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "vocab_size": vocab,
    }
    try:
        models = capture(endpoint + "/models", None, args.timeout, output, "models")
        verify_model(models, args.model, args.served_name)
        receipt["model_identity_verified"] = True
        if args.text_control:
            control = {
                "model": args.served_name,
                "messages": [{"role": "user", "content": TEXT_PROMPT}],
                "temperature": 0.0,
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if args.diagnostic_token_ids:
                control.update(return_token_ids=True, skip_special_tokens=False)
            receipt["text_control"] = completion_summary(
                capture(
                    endpoint + "/chat/completions",
                    control,
                    args.timeout,
                    output,
                    "text-control",
                ),
                vocab,
                require_ids=args.diagnostic_token_ids,
                expected_terms=("42",),
            )
        receipt["image"] = completion_summary(
            capture(
                endpoint + "/chat/completions", payload, args.timeout, output, "image"
            ),
            vocab,
            require_ids=args.diagnostic_token_ids,
        )
        result = receipt["image"]
        receipt["symptom_detected"] = (
            result["empty_content"]
            or bool(result["invalid_token_ids"])
            or not result["expected_words_in_order"]
        )
        code = 1 if receipt["symptom_detected"] else 0
    except Exception as error:
        receipt["collection_error"] = f"{type(error).__name__}: {error}"
        code = 2
    (output / "summary.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--model", required=True, help="Exact model root reported by /v1/models"
    )
    parser.add_argument("--served-name", required=True)
    parser.add_argument(
        "--image", required=True, help="tools/adaptation-gate-cases/images/image_05.png"
    )
    parser.add_argument(
        "--output-dir", required=True, help="A new directory for raw HTTP artifacts"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Optional explicit bound if model config is unavailable locally",
    )
    parser.add_argument(
        "--text-control",
        action="store_true",
        help="Also send one max8 text control before the image",
    )
    parser.add_argument(
        "--diagnostic-token-ids",
        action="store_true",
        help="R2 diagnostic: max8, return_token_ids=True, skip_special_tokens=False",
    )
    parser.add_argument(
        "--image-transport",
        choices=["file", "data-url"],
        default="file",
        help="Default file URI follows R1/R2; data-url is a portable unrerun variant",
    )
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

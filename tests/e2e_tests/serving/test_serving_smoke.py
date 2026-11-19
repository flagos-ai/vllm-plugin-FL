# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unified serving smoke test — auto-generated from model YAML configs.

Driven by environment variables set by ``tests/run.py``:

- ``FL_TEST_MODEL``: Model family (e.g. ``qwen3``, ``minicpm``)
- ``FL_TEST_CASE``:  Case name within the family (e.g. ``next_tp8``, ``o45_tp2``)
- ``FL_TEST_PLATFORM``/``FL_TEST_DEVICE``: Apply platform device overrides
Loads ``tests/models/<model>/<case>.yaml``, starts a vLLM server with the
engine config, and validates configured endpoints (completion, chat, embedding).

Supports both non-streaming (raw requests) and streaming (OpenAI SDK) modes,
controlled by the ``serve.stream`` flag in the model YAML.
"""

import base64
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import pytest
import requests

from tests.e2e_tests.adaptation_gate.gate_contract import (
    build_server_args,
    format_gate_progress,
    summarize_gate_progress,
    validate_scenario_documents,
)
from tests.e2e_tests.serving.server_helper import _NO_PROXY, VllmServer
from tests.e2e_tests.verification_cases import (
    case_pack_environment,
    ensure_server_alive,
    load_case_pack,
    validate_case_selection,
    verification_results_root,
)
from tests.utils.model_config import ModelConfig

# ---------------------------------------------------------------------------
# Load config from environment (injected by run.py)
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("FL_TEST_MODEL", "")
_CASE = os.environ.get("FL_TEST_CASE", "")
_PLATFORM = os.environ.get("FL_TEST_PLATFORM", "")
_DEVICE = os.environ.get("FL_TEST_DEVICE", "")

if not _MODEL or not _CASE:
    pytest.skip(
        "FL_TEST_MODEL and FL_TEST_CASE must be set (injected by run.py)",
        allow_module_level=True,
    )

_CFG = ModelConfig.load(_MODEL, _CASE, platform=_PLATFORM, device=_DEVICE)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VERIFICATION = _CFG.verification
_CASE_UTILS = None
if _VERIFICATION.source and _VERIFICATION.serving_scenarios:
    validate_case_selection(_VERIFICATION.source, _VERIFICATION.serving_scenarios)
    _CASE_UTILS = load_case_pack(_REPO_ROOT, _VERIFICATION.source)

if not os.path.exists(_CFG.model):
    pytest.fail(
        f"Model not found: {_CFG.model}",
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Start vLLM server with model config and optional serve overrides."""
    serve = _CFG.serve

    # Build extra_args from engine config + serve-specific overrides
    if _CASE_UTILS is None:
        extra_args = [*_CFG.serve_args(**serve.extra_engine), *serve.extra_args]
    else:
        extra_args = build_server_args(_CFG, Path(_CASE_UTILS.IMAGE_DIR))

    with VllmServer(
        model=_CFG.model,
        tp_size=_CFG.engine.get("tensor_parallel_size", 1),
        api_key=serve.api_key,
        served_model_name=serve.served_model_name,
        max_retries=serve.startup_retries,
        extra_args=extra_args,
    ) as srv:
        yield srv


@pytest.fixture
def base_url(server):
    return server.base_url


@pytest.fixture
def headers():
    serve = _CFG.serve
    h: dict[str, str] = {"Content-Type": "application/json"}
    if serve.api_key:
        h["Authorization"] = f"Bearer {serve.api_key}"
    return h


@pytest.fixture(scope="module")
def verification_context(server):
    """Share one serving process and environment across selected scenarios."""
    if _CASE_UTILS is None:
        yield None
        return
    results_root = verification_results_root(
        _REPO_ROOT,
        phase="serving",
        platform=_PLATFORM,
        device=_DEVICE,
        model=_MODEL,
        case=_CASE,
    )
    scenarios = tuple(
        (name, len(_CASE_UTILS.scenario_requests(name)))
        for name in _VERIFICATION.serving_scenarios
    )
    with case_pack_environment(
        case_utils=_CASE_UTILS,
        config=_CFG,
        server=server,
        results_root=results_root,
    ) as document_dir:
        yield document_dir
        with server.log_on_failure("Serving verification contract failed"):
            summary = validate_scenario_documents(document_dir, scenarios)
        print(
            f"[Gate] serving total: PASS; requests={summary['passed_requests']}/"
            f"{summary['logical_requests']} passed; checks={summary['passed_checks']}/"
            f"{summary['checks']} passed",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_REQUEST_MODEL = _CFG.serve.request_model(_CFG.model)


@pytest.mark.e2e
def test_model_list(server, base_url, headers):
    """Service must expose the loaded model in /v1/models."""
    with server.log_on_failure("Serving /v1/models validation failed"):
        response = requests.get(
            f"{base_url}/models", headers=headers, proxies=_NO_PROXY
        )
        assert response.status_code == 200
        models = response.json()["data"]
        assert any(m["id"] == _REQUEST_MODEL for m in models)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "scenario",
    _VERIFICATION.serving_scenarios or [None],
    ids=_VERIFICATION.serving_scenarios or ["not-configured"],
)
def test_verification_case(scenario, server, verification_context):
    """Run configured single-request cases in the standard serving job."""
    if scenario is None or _CASE_UTILS is None:
        pytest.skip("No external serving verification cases configured")
    with server.log_on_failure(f"Serving verification scenario '{scenario}' failed"):
        ensure_server_alive(server, "serving")
        try:
            _CASE_UTILS.run_case(scenario)
        finally:
            print(
                format_gate_progress(
                    f"serving/{scenario}",
                    summarize_gate_progress(verification_context, scenario),
                ),
                flush=True,
            )


# ---------------------------------------------------------------------------
# Endpoint runners
# ---------------------------------------------------------------------------


def _run_completion(base_url: str, headers: dict) -> None:
    """Validate /v1/completions endpoint."""
    serve = _CFG.serve
    payload = {
        "model": _REQUEST_MODEL,
        "prompt": serve.completion_prompt,
        "max_tokens": serve.max_tokens,
        **serve.sampling,
    }

    response = requests.post(
        f"{base_url}/completions",
        headers=headers,
        json=payload,
        proxies=_NO_PROXY,
    )

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

    generated_text = data["choices"][0]["text"]
    assert len(generated_text.strip()) > 0, "Generated text is empty"
    print(f"\nGenerated text: {generated_text}")


def _run_chat(base_url: str, headers: dict) -> None:
    """Validate /v1/chat/completions endpoint.

    When ``serve.stream`` is True, uses the OpenAI SDK for streaming.
    Otherwise, uses a plain requests POST.
    """
    serve = _CFG.serve
    chat_cases = serve.chat_cases or [
        {
            "name": "default",
            "messages": serve.chat_messages or [{"role": "user", "content": "Hello"}],
        }
    ]

    for chat_case in chat_cases:
        case_name = chat_case.get("name", "unnamed")
        if chat_case.get("generated_image"):
            messages = _generated_image_messages(
                chat_case.get("prompt", "Describe this image.")
            )
        else:
            messages = chat_case.get("messages", [])

        assert messages, f"Chat case '{case_name}' has no messages"
        print(f"\nRunning chat case: {case_name}")
        if serve.stream:
            response_text = _run_chat_streaming(base_url, serve, messages)
        else:
            response_text = _run_chat_non_streaming(base_url, headers, serve, messages)

        expected = chat_case.get("expected")
        if expected:
            expected_values = expected if isinstance(expected, list) else [expected]
            normalized_response = response_text.casefold()
            missing = [
                value
                for value in expected_values
                if value.casefold() not in normalized_response
            ]
            is_correct = not missing
            print(f"Answer validation: {'CORRECT' if is_correct else 'INCORRECT'}")
            assert is_correct, f"Chat case '{case_name}' is missing expected: {missing}"


def _generated_image_messages(prompt: str) -> list[dict]:
    """Create the local image payload used by the Qwen multimodal smoke case."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 250, 150), fill="blue")
    draw.text((90, 80), "Hello VLM", fill="yellow")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    print(
        "[image-input] "
        + json.dumps(
            {
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "bytes": len(image_bytes),
                "width": image.width,
                "height": image.height,
                "format": "JPEG",
                "mode": image.mode,
            },
            sort_keys=True,
        )
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _log_chat_metadata(finish_reason, usage) -> None:
    """Log only completion status and fixed token counts, never request data."""
    known_reasons = {"stop", "length", "tool_calls", "content_filter", "function_call"}
    reason = finish_reason if isinstance(finish_reason, str) else None
    if reason is not None and reason not in known_reasons:
        reason = "unrecognized"
    counts = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = (
            usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        )
        if type(value) is int and value >= 0:
            counts[name] = value
    print(
        "[chat-result] "
        + json.dumps(
            {"finish_reason": reason, "usage": counts or None},
            sort_keys=True,
        )
    )


def _run_chat_non_streaming(
    base_url: str,
    headers: dict,
    serve,
    messages: list[dict],
) -> str:
    """Non-streaming chat completions via raw requests."""
    payload: dict = {
        "model": _REQUEST_MODEL,
        "messages": messages,
        "max_tokens": serve.max_tokens,
        **serve.sampling,
    }
    if serve.extra_body:
        payload.update(serve.extra_body)
    if serve.chat_template_kwargs:
        payload["chat_template_kwargs"] = serve.chat_template_kwargs

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        proxies=_NO_PROXY,
    )
    if response.status_code != 200:
        # Diagnostic only — pass/fail semantics below are unchanged. vLLM puts
        # the server-side exception text into the JSON error body, which the CI
        # job log otherwise never shows (the pytest traceback stops at the
        # assertion on the next line).
        print(f"\nHTTP {response.status_code} from /chat/completions:", flush=True)
        print(response.text[:8000], flush=True)
    assert response.status_code == 200

    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

    content = data["choices"][0]["message"]["content"]
    assert len(content.strip()) > 0, "Assistant message is empty"
    _log_chat_metadata(data["choices"][0].get("finish_reason"), data.get("usage"))
    print(f"\nResponse: {content}")
    return content


def _run_chat_streaming(
    base_url: str,
    serve,
    messages: list[dict],
) -> str:
    """Streaming chat completions via OpenAI SDK."""
    import httpx
    from openai import OpenAI

    # trust_env=False prevents httpx from reading HTTP_PROXY env vars
    client = OpenAI(
        api_key=serve.api_key or "EMPTY",
        base_url=base_url,
        http_client=httpx.Client(trust_env=False),
    )

    create_kwargs: dict = {
        "model": _REQUEST_MODEL,
        "messages": messages,
        "max_tokens": serve.max_tokens,
        "stream": True,
        **serve.sampling,
    }
    if serve.extra_body:
        create_kwargs["extra_body"] = serve.extra_body
    if serve.chat_template_kwargs:
        create_kwargs["extra_body"] = {
            **create_kwargs.get("extra_body", {}),
            "chat_template_kwargs": serve.chat_template_kwargs,
        }

    response = client.chat.completions.create(**create_kwargs)

    text = ""
    finish_reason = None
    usage = None
    for chunk in response:
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta.content:
                text += choice.delta.content
            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage

    assert len(text.strip()) > 0, "Streaming response is empty"
    _log_chat_metadata(finish_reason, usage)
    print(f"\nStreaming response: {text}")
    return text


def _run_embedding(base_url: str, headers: dict) -> None:
    """Validate /v1/embeddings endpoint."""
    serve = _CFG.serve
    payload = {
        "model": _REQUEST_MODEL,
        "input": serve.embedding_input or "Hello world",
    }

    response = requests.post(
        f"{base_url}/embeddings",
        headers=headers,
        json=payload,
        proxies=_NO_PROXY,
    )
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    assert len(data["data"][0]["embedding"]) > 0
    print(f"\nEmbedding dimension: {len(data['data'][0]['embedding'])}")


_ENDPOINT_RUNNERS = {
    "completion": _run_completion,
    "chat": _run_chat,
    "embedding": _run_embedding,
}


@pytest.mark.e2e
@pytest.mark.parametrize("endpoint", _CFG.serve.endpoints, ids=_CFG.serve.endpoints)
def test_endpoint(endpoint: str, server, base_url, headers):
    """Validate a serving endpoint configured in the model YAML."""
    if _CASE_UTILS is not None:
        pytest.skip("External verification cases replace standard chat cases")
    with server.log_on_failure(f"Serving endpoint '{endpoint}' validation failed"):
        runner = _ENDPOINT_RUNNERS.get(endpoint)
        assert runner is not None, f"Unknown endpoint type: {endpoint}"
        runner(base_url, headers)

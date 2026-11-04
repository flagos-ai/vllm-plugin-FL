# Copyright 2026 FlagOS Contributors
"""Offline contracts: fake all HTTP and never import a GPU framework."""

import ast
import base64
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
IMAGE = REPO / "tools/adaptation-gate-cases/images/image_05.png"


@pytest.fixture
def repro():
    spec = importlib.util.spec_from_file_location(
        "repro_27b_image", HERE / "repro_27b_image.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r1_baseline_matches_actual_shared_prompt_without_token_overrides(repro):
    tree = ast.parse(
        (REPO / "tools/adaptation-gate-cases/case_utils.py").read_text(encoding="utf-8")
    )
    tasks = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "IMAGE_TASKS"
            for target in node.targets
        )
    )
    payload = repro.image_payload("served", IMAGE, IMAGE.read_bytes())
    assert payload == {
        "model": "served",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": tasks[4][0]},
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE.resolve().as_uri()},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_r2_diagnostic_is_explicit_max8_without_logprobs(repro):
    baseline = repro.image_payload("served", IMAGE, IMAGE.read_bytes())
    diagnostic = repro.image_payload(
        "served", IMAGE, IMAGE.read_bytes(), diagnostic=True
    )
    expected = dict(
        baseline, max_tokens=8, return_token_ids=True, skip_special_tokens=False
    )
    assert diagnostic == expected
    assert (
        "logprobs" not in diagnostic and "return_tokens_as_token_ids" not in diagnostic
    )


def test_portable_transport_preserves_png_bytes(repro):
    payload = repro.image_payload(
        "served", IMAGE, IMAGE.read_bytes(), transport="data-url"
    )
    url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == IMAGE.read_bytes()
    assert payload["max_tokens"] == 128


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8000", "http://localhost:8000/v1"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
    ],
)
def test_local_endpoint_normalization(repro, url, expected):
    assert repro.local_endpoint(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:8000",
        "http://127.0.0.1",
        "http://user@localhost:8000",
        "http://localhost:8000?x=1",
        "http://localhost:8000/#fragment",
        "http://localhost:8000/api",
        "file://localhost:8000",
        "http://localhost:0",
        "http://localhost:65536",
    ],
)
def test_reject_remote_or_ambiguous_endpoint(repro, url):
    with pytest.raises(ValueError):
        repro.local_endpoint(url)


def response(content="HELLO VLLM", ids=None):
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": "stop", "token_ids": ids}
        ],
        "usage": {"prompt_tokens": 1062},
    }


def test_baseline_does_not_invent_missing_token_ids(repro):
    result = repro.completion_summary(response(""), 248320)
    assert result["empty_content"] is True
    assert result["token_ids_available"] is False
    assert result["invalid_token_ids"] is None
    assert result["upper_token_bound_checked"] is False
    with pytest.raises(ValueError, match="token_ids"):
        repro.completion_summary(response(""), 248320, require_ids=True)


def test_reports_one_past_vocab_and_negative_ids_without_mutation(repro):
    document = response("", [248319, 248320, -1])
    result = repro.completion_summary(document, 248320, require_ids=True)
    assert result["invalid_token_ids"] == [248320, -1]
    assert document["choices"][0]["token_ids"] == [248319, 248320, -1]
    unknown = repro.completion_summary(document, None, require_ids=True)
    assert unknown["invalid_token_ids"] == [-1]
    assert unknown["upper_token_bound_checked"] is False


def test_optional_vocab_bound_comes_from_explicit_or_local_config(repro, tmp_path):
    assert repro.vocabulary_size("absent-model", None) is None
    assert repro.vocabulary_size("absent-model", 248320) == 248320
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"vocab_size": 248320}})
    )
    assert repro.vocabulary_size(str(tmp_path), None) == 248320
    with pytest.raises(ValueError):
        repro.vocabulary_size(str(tmp_path), 0)


def args(tmp_path, **kwargs):
    values = dict(
        endpoint="http://localhost:8000",
        model="model-root",
        served_name="served",
        image=str(IMAGE),
        output_dir=str(tmp_path / "raw"),
        vocab_size=248320,
        timeout=300,
        text_control=False,
        image_transport="file",
        diagnostic_token_ids=False,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_default_sends_one_image_only_after_root_and_name_match(
    repro, monkeypatch, tmp_path
):
    calls = []

    def fake(url, payload, timeout, output, label):
        calls.append((label, payload))
        return (
            {"data": [{"id": "served", "root": "model-root"}]}
            if label == "models"
            else response()
        )

    monkeypatch.setattr(repro, "capture", fake)
    assert repro.run(args(tmp_path)) == 0
    assert [name for name, _ in calls] == ["models", "image"]
    assert calls[1][1]["max_tokens"] == 128
    assert "return_token_ids" not in calls[1][1]


@pytest.mark.parametrize(
    "card",
    [{"id": "other", "root": "model-root"}, {"id": "served", "root": "other-root"}],
)
def test_wrong_service_never_receives_image(repro, monkeypatch, tmp_path, card):
    calls = []

    def fake(url, payload, timeout, output, label):
        calls.append(label)
        return {"data": [card]}

    monkeypatch.setattr(repro, "capture", fake)
    assert repro.run(args(tmp_path)) == 2
    assert calls == ["models"]


def test_optional_control_adds_exactly_one_request_and_diagnostic_flags(
    repro, monkeypatch, tmp_path
):
    calls = []

    def fake(url, payload, timeout, output, label):
        calls.append((label, payload))
        return (
            {"data": [{"id": "served", "root": "model-root"}]}
            if label == "models"
            else response(ids=[1, 2])
        )

    monkeypatch.setattr(repro, "capture", fake)
    assert repro.run(args(tmp_path, text_control=True, diagnostic_token_ids=True)) == 0
    assert [label for label, _ in calls] == ["models", "text-control", "image"]
    assert all(payload["max_tokens"] == 8 for _, payload in calls[1:])


def test_empty_invalid_diagnostic_is_reported_as_symptom(repro, monkeypatch, tmp_path):
    def fake(url, payload, timeout, output, label):
        return (
            {"data": [{"id": "served", "root": "model-root"}]}
            if label == "models"
            else response("", [248320] * 8)
        )

    monkeypatch.setattr(repro, "capture", fake)
    assert repro.run(args(tmp_path, diagnostic_token_ids=True)) == 1
    summary = json.loads((tmp_path / "raw/summary.json").read_text())
    assert summary["symptom_detected"] is True
    assert summary["image"]["invalid_token_ids"] == [248320] * 8


def test_capture_preserves_exact_http_error_body_and_disables_redirects(
    repro, monkeypatch, tmp_path
):
    body = b'{ "error": "original raw body" }\n'

    def fail(*args, **kwargs):
        raise HTTPError(
            "http://localhost:8000/v1/models", 500, "failed", {}, io.BytesIO(body)
        )

    def opener(*handlers):
        assert any(isinstance(handler, repro.NoRedirect) for handler in handlers)
        assert any(
            isinstance(handler, repro.ProxyHandler) and handler.proxies == {}
            for handler in handlers
        )
        return SimpleNamespace(open=fail)

    monkeypatch.setattr(repro, "build_opener", opener)
    with pytest.raises(ValueError, match="HTTP 500"):
        repro.capture("http://localhost:8000/v1/models", None, 300, tmp_path, "models")
    assert (tmp_path / "models.response.body").read_bytes() == body
    assert json.loads((tmp_path / "models.http.json").read_text())["http_status"] == 500
    assert (
        repro.NoRedirect().redirect_request(None, None, None, None, None, None) is None
    )


def test_text_control_uses_its_own_expected_answer(repro):
    result = repro.completion_summary(response("42"), 248320, expected_terms=("42",))
    assert result["expected_terms"] == ["42"]
    assert result["expected_words_in_order"] is True
    assert (
        repro.completion_summary(response("42"), 248320)["expected_words_in_order"]
        is False
    )

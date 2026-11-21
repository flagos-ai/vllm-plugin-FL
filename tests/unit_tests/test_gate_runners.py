# Copyright (c) 2026 BAAI. All rights reserved.

"""Runner-level tests for the gate execution split.

Covers the three-way execution mode split, server-argument assembly, the
shared gather executor, artifact/contract validation, and (where the
module is importable without torch/vllm) the serving runner wiring with
mocked transports.  Assertion cores are plain functions so they also run
without the full pytest/conftest environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.e2e_tests.adaptation_gate.gate_contract import (
    apply_local_media_flag,
    evaluate_gate_entry,
    execute_gate_entries,
    format_gate_scenario_log,
    format_gate_total_log,
    gate_results_root,
    run_concurrent_scenario,
    validate_scenario_documents,
    write_gate_document,
)
from tests.e2e_tests.chat_cases import (
    assemble_gate_extra_args,
    gate_entry,
    resolve_execution_mode,
    validate_chat_case_tasks,
    validate_gate_case,
)
from tests.unit_tests.test_chat_cases import REPO_ROOT, REQUEST_DATA_MARKERS

RUNNER_SOURCES = (
    REPO_ROOT / "tests/e2e_tests/serving/test_serving_smoke.py",
    REPO_ROOT / "tests/e2e_tests/concurrent/test_concurrent_smoke.py",
    REPO_ROOT / "tests/e2e_tests/chat_cases.py",
    REPO_ROOT / "tests/e2e_tests/adaptation_gate/gate_contract.py",
)

TEMPLATE_MODEL = "qwen3_6"
TEMPLATE_CASE = "27b_tp2_eager"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingAPI:
    """Async fake for client.chat.completions.create."""

    def __init__(self, text="reply now", fail=False):
        self.calls = []
        self.inflight = 0
        self.max_inflight = 0
        self._text = text
        self._fail = fail

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0)
        self.inflight -= 1
        if self._fail:
            raise RuntimeError("injected transport failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))]
        )


def _fake_gate_client(api):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=api.create))
    )


def _generic_entries(count, scenario="text_concurrent_8", start=1):
    return [
        {
            "name": f"{scenario}-{index:02d}",
            "scenario": scenario,
            "messages": [{"role": "user", "content": f"generic probe {index}"}],
            "expected": ["reply"],
            "min_length": 1,
            "max_tokens": 32 + index,
        }
        for index in range(start, start + count)
    ]


def _fake_openai_httpx_modules():
    """Install recording stand-ins for openai.AsyncOpenAI and httpx."""
    saved = (
        sys.modules.get("openai"),
        sys.modules.get("httpx"),
    )

    class _FakeAsyncHttpClient:
        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            _FakeAsyncHttpClient.instances.append(self)

        async def close(self):
            self.closed = True

    class _FakeAsyncOpenAI:
        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.api = _RecordingAPI()
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.api.create)
            )
            _FakeAsyncOpenAI.instances.append(self)

        async def close(self):
            self.closed = True

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.AsyncClient = _FakeAsyncHttpClient
    sys.modules["openai"] = fake_openai
    sys.modules["httpx"] = fake_httpx
    return saved, _FakeAsyncOpenAI, _FakeAsyncHttpClient


def _restore_openai_httpx(saved):
    for name, module in zip(("openai", "httpx"), saved):
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _passing_text(expected, min_length):
    """Build a response that passes all ten checks (unique filler tokens)."""
    text = " ".join(expected)
    if len(text) < min_length:
        filler = " ".join(f"w{index:03d}" for index in range(1, 80))
        text = (text + " " + filler)[: max(min_length, 1)].rstrip()
        while len(text) < min_length:
            text += " tail"
    return text


# ---------------------------------------------------------------------------
# Assertion cores
# ---------------------------------------------------------------------------


def _assert_execution_mode_split():
    legacy_cases = [{"name": "old", "messages": [{"role": "user", "content": "x"}]}]
    gate_serving = [
        dict(legacy_cases[0], name="g", task="serving", scenario="text_single")
    ]
    gate_conc = [
        dict(legacy_cases[0], name="c", task="concurrent", scenario="text_concurrent_8")
    ]

    # Explicit task cases win over a present verification block.
    assert resolve_execution_mode(gate_serving, True, "serving") == "gate", (
        "explicit task: serving must take priority over verification"
    )
    assert resolve_execution_mode(gate_conc, True, "concurrent") == "gate"
    # Verification only applies when no explicit cases exist for the task.
    assert resolve_execution_mode(legacy_cases, True, "concurrent") == "verification"
    assert resolve_execution_mode(gate_serving, True, "concurrent") == "verification"
    # Neither → legacy behaviour on both runners.
    assert resolve_execution_mode(legacy_cases, False, "serving") == "legacy"
    assert resolve_execution_mode(legacy_cases, False, "concurrent") == "legacy"

    validate_chat_case_tasks(gate_serving + gate_conc + legacy_cases)
    try:
        validate_chat_case_tasks([{"name": "bad", "task": "typo"}])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown task must be rejected")


def _assert_server_args_assembly():
    base = ["--max-model-len", "8192", "--enforce-eager"]
    image_case = {
        "name": "img",
        "task": "serving",
        "scenario": "image_single",
        "image_path": "tools/adaptation-gate-cases/images/image_05.png",
        "prompt": "generic",
        "expected": ["x"],
        "min_length": 1,
        "max_tokens": 8,
    }
    text_case = dict(image_case, name="txt", scenario="text_single")
    text_case.pop("image_path")
    text_case["messages"] = [{"role": "user", "content": "x"}]

    with_images = assemble_gate_extra_args(base, [image_case], REPO_ROOT)
    assert with_images[: len(base)] == base, "existing extra_args must be preserved"
    assert with_images.count("--allowed-local-media-path") == 1
    media_dir = with_images[with_images.index("--allowed-local-media-path") + 1]
    assert media_dir.endswith("tools/adaptation-gate-cases/images")

    no_images = assemble_gate_extra_args(base, [text_case], REPO_ROOT)
    assert no_images == base, "text-only gate cases must not add the media flag"

    try:
        apply_local_media_flag(["--allowed-local-media-path", "/already"], None)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate media flag must fail loudly")


def _assert_gate_case_validation():
    good = {
        "name": "g",
        "task": "concurrent",
        "scenario": "text_concurrent_8",
        "messages": [{"role": "user", "content": "x"}],
        "expected": ["x"],
        "min_length": 1,
        "max_tokens": 8,
    }
    validate_gate_case(good, "concurrent")
    for bad, task in (
        (dict(good, task="serving"), "concurrent"),
        (dict(good, scenario="text_single"), "concurrent"),
        (dict(good, expected=[]), "concurrent"),
        (dict(good, min_length=True), "concurrent"),
        (dict(good, max_tokens=0), "concurrent"),
        (dict(good, image_path="x", messages=None), "concurrent"),
        (dict(good, scenario="unknown_scenario"), "concurrent"),
    ):
        try:
            validate_gate_case(bad, task)
        except ValueError:
            continue
        raise AssertionError(f"validate_gate_case accepted {bad}")


def _assert_executor_single_gather_and_case_params():
    api = _RecordingAPI(text="reply now with enough length to pass")
    client = _fake_gate_client(api)
    entries = _generic_entries(8)
    outputs = asyncio.run(execute_gate_entries(client, "model-x", entries))
    assert len(outputs) == 8
    assert api.max_inflight == 8, "eight requests must run inside one gather"
    assert len(api.calls) == 8
    for call, entry in zip(api.calls, entries):
        assert call["max_tokens"] == entry["max_tokens"], (
            "case-level max_tokens must be used, not serve.max_tokens"
        )
        assert call["model"] == "model-x"
        assert call["temperature"] == 0.0
        assert call["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert call["messages"] == entry["messages"]
    assert all(output["passed"] for output in outputs)
    for output in outputs:
        assert len(output["checks"]) == 10
        assert all(isinstance(value, bool) for value in output["checks"].values())

    failing_api = _RecordingAPI(fail=True)
    outputs = asyncio.run(
        execute_gate_entries(_fake_gate_client(failing_api), "m", _generic_entries(2))
    )
    assert all(not output["passed"] for output in outputs)
    for output in outputs:
        assert "RuntimeError" in output["error"]
        assert len(output["checks"]) == 10
        assert not any(output["checks"].values())


def _assert_run_concurrent_scenario_mock():
    saved, fake_openai_cls, fake_httpx_cls = _fake_openai_httpx_modules()
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            document_dir = Path(tmp) / "concurrent"
            outputs = run_concurrent_scenario(
                _generic_entries(8),
                served_model_name="served-x",
                base_url="http://127.0.0.1:9/v1",
                api_key=None,
                timeout="300",
                document_dir=document_dir,
                scenario="text_concurrent_8",
                model_path="/models/x",
                port="9",
            )
            assert len(outputs) == 8
            client = fake_openai_cls.instances[-1]
            http_client = fake_httpx_cls.instances[-1]
            assert client.kwargs["base_url"] == "http://127.0.0.1:9/v1"
            assert client.kwargs["api_key"] == "EMPTY"
            assert http_client.kwargs["trust_env"] is False, (
                "the gate HTTP client must set trust_env=False explicitly"
            )
            assert client.closed and http_client.closed, "clients must be closed"
            assert client.api.max_inflight == 8

            document = json.loads((document_dir / "text_concurrent_8.json").read_text())
            assert document["summary"]["request_count"] == 8
            assert len(document["output"]) == 8
            assert document["schema_version"] == 1
            validate_scenario_documents(document_dir, (("text_concurrent_8", 8),))
    finally:
        _restore_openai_httpx(saved)


def _assert_document_contract_counts():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        serving_dir = Path(tmp) / "serving"
        for scenario, expected, min_length in (
            ("text_single", ["reply"], 1),
            ("image_single", ["hello"], 1),
        ):
            entry = {
                "name": f"serving-{scenario}",
                "scenario": scenario,
                "messages": [{"role": "user", "content": "x"}],
                "expected": expected,
                "min_length": min_length,
                "max_tokens": 8,
            }
            output = evaluate_gate_entry(entry, _passing_text(expected, min_length))
            write_gate_document(
                serving_dir,
                scenario,
                model_path="/m",
                served_model_name="s",
                port="1",
                entries=[entry],
                outputs=[output],
            )
        summary = validate_scenario_documents(
            serving_dir, (("text_single", 1), ("image_single", 1))
        )
        assert summary["logical_requests"] == 2 and summary["checks"] == 20
        assert summary["passed_requests"] == 2 and summary["passed_checks"] == 20
        total_line = format_gate_total_log("serving", summary)
        assert "requests=2/2" in total_line and "checks=20/20" in total_line
        scenario_line = format_gate_scenario_log(
            "serving",
            "text_single",
            [
                {
                    "passed": True,
                    "checks": {
                        k: True
                        for k in (
                            "non_empty",
                            "minimum_length",
                            "expected_semantics",
                            "expected_order",
                            "no_bang_triplet",
                            "no_mojibake",
                            "no_control_characters",
                            "no_long_character_run",
                            "no_repeated_word_run",
                            "no_repeated_phrase",
                        )
                    },
                }
            ],
        )
        assert (
            scenario_line
            == "[Gate] serving scenario=text_single requests=1/1 checks=10/10"
        )

        # Failure variants must fail the validator loudly.
        (serving_dir / "image_single.json").unlink()
        try:
            validate_scenario_documents(
                serving_dir, (("text_single", 1), ("image_single", 1))
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("missing scenario document must fail validation")

        saved, fake_openai_cls, _ = _fake_openai_httpx_modules()
        try:
            with tempfile.TemporaryDirectory() as tmp2:
                concurrent_dir = Path(tmp2) / "concurrent"
                groups = {
                    "text_concurrent_8": _generic_entries(8, "text_concurrent_8"),
                    "image_concurrent_8": _generic_entries(8, "image_concurrent_8"),
                    "mixed_concurrent_8": _generic_entries(8, "mixed_concurrent_8"),
                }
                for scenario, entries in groups.items():
                    run_concurrent_scenario(
                        entries,
                        served_model_name="s",
                        base_url="http://127.0.0.1:9/v1",
                        api_key="EMPTY",
                        timeout="300",
                        document_dir=concurrent_dir,
                        scenario=scenario,
                        model_path="/m",
                        port="9",
                    )
                total = 0
                checks_total = 0
                summary = validate_scenario_documents(
                    concurrent_dir, tuple((scenario, 8) for scenario in groups)
                )
                total = summary["logical_requests"]
                checks_total = summary["checks"]
                assert (total, checks_total) == (24, 240)
                assert summary["passed_requests"] == 24
                assert summary["passed_checks"] == 240
                line = format_gate_total_log(
                    "concurrent",
                    {
                        "passed": True,
                        "passed_requests": 24,
                        "logical_requests": 24,
                        "passed_checks": 240,
                        "checks": 240,
                    },
                )
                assert "requests=24/24" in line and "checks=240/240" in line

                # A single dropped request breaks the contract.
                document = json.loads(
                    (concurrent_dir / "text_concurrent_8.json").read_text()
                )
                document["output"].pop()
                document["input"].pop()
                (concurrent_dir / "broken.json").write_text(json.dumps(document))
                try:
                    validate_scenario_documents(
                        concurrent_dir, (("text_concurrent_8", 8),)
                    )
                except AssertionError:
                    pass
                else:
                    raise AssertionError("unbalanced documents must fail validation")
                (concurrent_dir / "broken.json").unlink()

                # A missing check key or a non-bool check fails too.
                for mutate in (
                    lambda out: out["checks"].pop("non_empty"),
                    lambda out: out["checks"].__setitem__("non_empty", "yes"),
                ):
                    document = json.loads(
                        (concurrent_dir / "text_concurrent_8.json").read_text()
                    )
                    mutate(document["output"][0])
                    (concurrent_dir / "broken.json").write_text(json.dumps(document))
                    try:
                        validate_scenario_documents(
                            concurrent_dir, (("text_concurrent_8", 8),)
                        )
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("malformed checks must fail validation")
                    (concurrent_dir / "broken.json").unlink()
        finally:
            _restore_openai_httpx(saved)

    # Directory isolation: serving and concurrent roots never collide.
    serving_root = gate_results_root(
        Path(tempfile.gettempdir()),
        platform="p",
        device="d",
        model="m",
        case="c-serving",
    )
    concurrent_root = gate_results_root(
        Path(tempfile.gettempdir()),
        platform="p",
        device="d",
        model="m",
        case="c-concurrent",
    )
    assert serving_root != concurrent_root
    assert serving_root.name == "c-serving" and concurrent_root.name == "c-concurrent"


def _install_openai_stub():
    """Let the legacy verification pack import without the openai package."""
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("openai")

        class _StubAsyncOpenAI:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("HTTP client must not be constructed in tests")

        stub.AsyncOpenAI = _StubAsyncOpenAI
        sys.modules.setdefault("openai", stub)


def _install_torch_stub():
    """Satisfy the import-time ``import torch`` in the module chain.

    device_utils only touches torch inside functions; the serving-runner
    code exercised here never calls into it.  A MagicMock absorbs the
    attribute lookups that happen while module-level annotations evaluate.
    """
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        from unittest.mock import MagicMock

        sys.modules.setdefault("torch", MagicMock())


def _import_serving_smoke(case=TEMPLATE_CASE):
    """Import the serving runner without torch/vllm (model path shimmed)."""
    _install_openai_stub()
    _install_torch_stub()
    environment = {
        "FL_TEST_MODEL": TEMPLATE_MODEL,
        "FL_TEST_CASE": case,
        "FL_TEST_PLATFORM": "",
        "FL_TEST_DEVICE": "",
    }
    saved_env = {key: os.environ.get(key) for key in environment}
    saved_exists = os.path.exists
    module_name = "tests.e2e_tests.serving.test_serving_smoke"
    sys.modules.pop(module_name, None)
    os.environ.update(environment)
    os.path.exists = lambda path: True  # shim the /data model path at import time
    try:
        module = importlib.import_module(module_name)
    finally:
        os.path.exists = saved_exists
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def _assert_serving_runner_gate_mode():
    module = _import_serving_smoke()
    try:
        assert len(module._GATE_CASES) == 2

        # Server arguments: tp and existing args preserved, media flag added once.
        class _FakeVllmServer:
            last_kwargs = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                _FakeVllmServer.last_kwargs = kwargs
                self.port = "8080"
                self.base_url = "http://127.0.0.1:8080/v1"
                self._process = None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def log_on_failure(self, message):
                return nullcontext()

            def preserve_failure_logs(self, message):
                return message

        original_server = module.VllmServer
        module.VllmServer = _FakeVllmServer
        try:
            # Bypass pytest's direct-call guard: run the raw fixture generator.
            fixture_generator = module.server.__wrapped__()
            fake_server = next(fixture_generator)
            kwargs = _FakeVllmServer.last_kwargs
            assert kwargs["tp_size"] == 2, "tensor_parallel_size must be passed through"
            assert kwargs["max_retries"] == module._CFG.serve.startup_retries
            extra_args = kwargs["extra_args"]
            assert "--max-model-len" in extra_args and "8192" in extra_args
            assert extra_args.count("--allowed-local-media-path") == 1
            media_dir = extra_args[extra_args.index("--allowed-local-media-path") + 1]
            from tests.e2e_tests.chat_cases import local_media_directory

            expected_media = str(
                local_media_directory(module._GATE_CASES, module._REPO_ROOT)
            )
            assert media_dir == expected_media
            with contextlib.suppress(StopIteration):
                fixture_generator.close()
        finally:
            module.VllmServer = original_server

        # Gate case execution: normal transport, JSON artifact, real counters.
        text_single = module._GATE_CASES[0]
        image_single = module._GATE_CASES[1]
        assert text_single["scenario"] == "text_single"

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            document_dir = Path(tmp) / "serving"
            document_dir.mkdir()
            calls = {"post": 0, "stream": 0}

            def fake_post(base_url, headers, payload):
                calls["post"] += 1
                content = str(payload["messages"][0]["content"])
                if "Introduce large language models" in content:
                    text = _passing_text(
                        text_single["expected"], text_single["min_length"]
                    )
                else:
                    text = _passing_text(
                        image_single["expected"], image_single["min_length"]
                    )
                return text, "stop", {"total_tokens": 10}

            def fake_stream(base_url, api_key, create_kwargs):
                calls["stream"] += 1
                messages = create_kwargs["messages"]
                content = str(messages[0]["content"])
                if "Introduce large language models" in content:
                    text = _passing_text(
                        text_single["expected"], text_single["min_length"]
                    )
                else:
                    text = _passing_text(
                        image_single["expected"], image_single["min_length"]
                    )
                return text, "stop", {"total_tokens": 10}

            original_post = module._post_chat_completions
            original_stream = module._stream_chat_completions
            module._post_chat_completions = fake_post
            module._stream_chat_completions = fake_stream
            fake_server = SimpleNamespace(port="8080")
            try:
                assert module._CFG.serve.stream is False
                output = module._execute_serving_gate_case(
                    text_single, fake_server, "http://b", {}, document_dir
                )
                assert output["passed"] is True
                assert calls == {"post": 1, "stream": 0}
                assert (document_dir / "text_single.json").exists()
                assert len(output["checks"]) == 10

                # Streaming must reuse the same pipeline when configured.
                module._CFG.serve.stream = True
                output = module._execute_serving_gate_case(
                    image_single, fake_server, "http://b", {}, document_dir
                )
                assert output["passed"] is True
                assert calls["stream"] == 1 and calls["post"] == 1
            finally:
                module._post_chat_completions = original_post
                module._stream_chat_completions = original_stream
                module._CFG.serve.stream = False

            # Multimodal messages reach the streaming transport as file:// URIs.
            def fake_stream_capture(base_url, api_key, create_kwargs):
                module._captured_stream_kwargs = create_kwargs
                return (
                    _passing_text(image_single["expected"], image_single["min_length"]),
                    "stop",
                    None,
                )

            module._stream_chat_completions = fake_stream_capture
            module._CFG.serve.stream = True
            try:
                module._execute_serving_gate_case(
                    image_single, fake_server, "http://b", {}, document_dir
                )
            finally:
                module._stream_chat_completions = original_stream
                module._CFG.serve.stream = False
            messages = module._captured_stream_kwargs["messages"]
            image_url = messages[0]["content"][1]["image_url"]["url"]
            assert image_url.startswith("file://"), image_url
            assert image_url.endswith("image_05.png")

            # Both scenario documents now validate: 2 requests / 20 checks.
            summary = validate_scenario_documents(
                document_dir, (("text_single", 1), ("image_single", 1))
            )
            assert (summary["passed_requests"], summary["passed_checks"]) == (2, 20)

        # The legacy endpoint test skips itself in gate mode.
        try:
            module.test_endpoint("chat", None, None, None)
        except pytest.skip.Exception:
            pass
        else:
            raise AssertionError("test_endpoint must skip in gate mode")
    finally:
        sys.modules.pop("tests.e2e_tests.serving.test_serving_smoke", None)


def _assert_legacy_path_removed():
    """The verification compatibility path is gone; gate is the only path."""
    import subprocess

    assert not (REPO_ROOT / "tests/e2e_tests/verification_cases.py").exists(), (
        "verification_cases.py must be deleted in the cleanup phase"
    )
    probe = subprocess.run(
        [sys.executable, "-c", "import tests.e2e_tests.verification_cases"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0 and "ModuleNotFoundError" in probe.stderr, (
        "tests.e2e_tests.verification_cases must no longer be importable"
    )

    module = _import_serving_smoke(case="35b_a3b_tp2_eager")
    try:
        assert len(module._GATE_CASES) == 2
        for gone in ("_CASE_UTILS", "_VERIFICATION", "_OLD_VERIFICATION"):
            assert not hasattr(module, gone), (
                f"serving runner must not keep the legacy symbol {gone}"
            )
        source = (
            REPO_ROOT / "tests/e2e_tests/serving/test_serving_smoke.py"
        ).read_text()
        for forbidden in (
            "load_case_pack",
            "case_pack_environment",
            "_CFG.verification",
            "verification_cases",
        ):
            assert forbidden not in source, (
                f"serving runner still references the legacy path: {forbidden}"
            )
    finally:
        sys.modules.pop("tests.e2e_tests.serving.test_serving_smoke", None)


def _assert_no_request_data_in_runners():
    # Pure production modules: no reference to the tools pack at all.
    for path in RUNNER_SOURCES:
        if path.name not in ("chat_cases.py", "gate_contract.py"):
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in ("case_utils", "adaptation-gate-cases", "tools/"):
            assert forbidden not in text, (
                f"{path.name} must not reference tools ({forbidden!r} found)"
            )
    # Runners: no gate request data, no tools path, and gate clients only
    # via the shared factory.  The legacy offline prompt pool in the
    # concurrent runner predates the gate work and stays byte-identical,
    # so the one colliding marker is exempted there (all other markers
    # still apply, and the pure-production scan above keeps the full list).
    offline_pool_collision = {"capital of France"}
    for path in RUNNER_SOURCES:
        if path.name not in ("test_serving_smoke.py", "test_concurrent_smoke.py"):
            continue
        text = path.read_text(encoding="utf-8")
        markers = set(REQUEST_DATA_MARKERS) - offline_pool_collision
        for marker in sorted(markers):
            assert marker not in text, (
                f"{path.name} must not embed gate request data (found {marker!r})"
            )
        assert "tools/adaptation-gate-cases" not in text
        assert "AsyncOpenAI(" not in text, (
            f"{path.name} must build gate clients only via gate_contract"
        )


def _assert_gate_entries_from_template():
    from tests.e2e_tests.chat_cases import group_concurrent_cases
    from tests.utils.model_config import ModelConfig

    for case_name in sorted(
        path.stem for path in (REPO_ROOT / "tests/models/qwen3_6").glob("*.yaml")
    ):
        config = ModelConfig.load(TEMPLATE_MODEL, case_name)
        cases = config.serve.chat_cases
        assert len(cases) == 26, case_name
        assert len([c for c in cases if c.get("task") == "serving"]) == 2, case_name
        groups = group_concurrent_cases(cases)
        assert [len(group) for group in groups.values()] == [8, 8, 8], case_name
        entries = [gate_entry(case, REPO_ROOT) for case in cases]
        assert all(entry["messages"] for entry in entries), case_name
        assert all(entry["max_tokens"] in (128, 512) for entry in entries), case_name


def _platform_qwen36_references():
    """Collect every (platform, device, case) the matrix schedules for qwen3_6."""
    refs = set()
    for platform_path in sorted((REPO_ROOT / "tests/platforms").glob("*.yaml")):
        data = yaml.safe_load(platform_path.read_text())
        for device_name, device in (data or {}).items():
            if not isinstance(device, dict) or "tests" not in device:
                continue
            e2e = device.get("tests", {}).get("e2e") or {}
            for stage in ("pr", "nightly", "weekly"):
                for task, models in (e2e.get(stage) or {}).items():
                    if not isinstance(models, dict):
                        continue
                    for case in models.get("qwen3_6") or []:
                        refs.add((platform_path.stem, device_name, case))
    return sorted(refs)


def _assert_platform_merge_preserves_gate_cases():
    """Guard: platform overrides must never swallow the 26-case inventory.

    For every matrix reference, ModelConfig.load(platform, device) must
    still expose the full explicit inventory, and platform-specific serve
    settings must survive the merge.
    """
    from tests.e2e_tests.chat_cases import (
        group_concurrent_cases,
        local_media_directory,
        validate_gate_chat_cases,
    )
    from tests.utils.model_config import ModelConfig

    assert _platform_qwen36_references(), "matrix references must be discoverable"
    for platform, device, case in _platform_qwen36_references():
        merged = ModelConfig.load("qwen3_6", case, platform=platform, device=device)
        cases = merged.serve.chat_cases
        summary = validate_gate_chat_cases(cases)
        assert summary["total"] == 26, f"{platform}/{device}/{case}"
        assert len([c for c in cases if c.get("task") == "serving"]) == 2, (
            f"{platform}/{device}/{case}"
        )
        assert len([c for c in cases if c.get("task") == "concurrent"]) == 24, (
            f"{platform}/{device}/{case}"
        )
        assert list(group_concurrent_cases(cases)) == [
            "text_concurrent_8",
            "image_concurrent_8",
            "mixed_concurrent_8",
        ], f"{platform}/{device}/{case}"
        # Image paths must still resolve after the merge.
        assert (
            local_media_directory([c for c in cases if c.get("image_path")], REPO_ROOT)
            is not None
        ), f"{platform}/{device}/{case}"

        # Engine identity must not drift from the case file.
        base = ModelConfig.load("qwen3_6", case)
        assert merged.engine.get("tensor_parallel_size") == base.engine.get(
            "tensor_parallel_size"
        ), f"{platform}/{device}/{case}: tensor_parallel_size drifted"
        assert merged.engine.get("enforce_eager") == base.engine.get("enforce_eager"), (
            f"{platform}/{device}/{case}: enforce_eager drifted"
        )

        # Platform-specific serve settings survive (kunlunxin example).
        if platform == "kunlunxin":
            assert merged.serve.served_model_name == "qwen3"
            assert merged.serve.startup_retries == 420
            assert "--reasoning-parser" in merged.serve.extra_args
            assert merged.serve.extra_body == {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        # The legacy verification compatibility path stays removed after merge.
        assert not hasattr(merged, "verification"), f"{platform}/{device}/{case}"


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


def test_execution_mode_split():
    _assert_execution_mode_split()


def test_server_args_assembly():
    _assert_server_args_assembly()


def test_gate_case_validation():
    _assert_gate_case_validation()


def test_executor_single_gather_and_case_params():
    _assert_executor_single_gather_and_case_params()


def test_run_concurrent_scenario_mock():
    _assert_run_concurrent_scenario_mock()


def test_document_contract_counts():
    _assert_document_contract_counts()


def test_serving_runner_gate_mode():
    _assert_serving_runner_gate_mode()


def test_legacy_path_removed():
    _assert_legacy_path_removed()


def test_platform_merge_preserves_gate_cases():
    _assert_platform_merge_preserves_gate_cases()


def test_no_request_data_in_runners():
    _assert_no_request_data_in_runners()


def test_gate_entries_from_template():
    _assert_gate_entries_from_template()

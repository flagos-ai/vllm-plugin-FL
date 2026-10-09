# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU regressions for retaining server diagnostics after serving failures."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from tests.e2e_tests.serving.server_helper import VllmServer
from tests.utils.model_config import ModelConfig


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server = VllmServer(model="/models/Example-27B", tp_size=8, api_key="test-secret")
    process = Mock()

    def launch(cmd, **kwargs):
        kwargs["stdout"].write(
            b"SERVER_STARTED\nEngineCore failed: injected crash\ntest-secret\n"
        )
        kwargs["stdout"].flush()
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(server, "_wait_ready", lambda: None)
    server.start()
    log_path = Path(server._log_file.name)
    yield server, log_path, process
    server.stop()


@pytest.mark.parametrize(
    "error",
    [
        AssertionError("bad response"),
        requests.ConnectionError("engine died"),
        pytest.fail.Exception("test failed"),
    ],
)
def test_request_failure_survives_normal_yield_fixture_teardown(
    running_server, capsys, error
):
    server, log_path, process = running_server
    request = Mock(side_effect=error)
    with (
        pytest.raises(type(error), match=str(error)),
        server.log_on_failure("Serving chat failed"),
    ):
        request()
    request.assert_called_once()

    # A pytest yield fixture calls __exit__ with no exception, even on failure.
    server.__exit__(None, None, None)
    output = capsys.readouterr().out
    assert "EngineCore failed: injected crash" not in output
    assert "Full log:" in output
    assert "test-secret" not in output
    assert log_path.parent == Path("test-results/logs").resolve()
    assert "Example-27B_tp8_" in log_path.name
    assert log_path.exists()
    assert "EngineCore failed: injected crash" in log_path.read_text()
    assert "test-secret" not in log_path.read_text()
    assert "[REDACTED]" in log_path.read_text()
    process.terminate.assert_called_once()
    assert server._log_file is None


def test_success_removes_log_without_failure_diagnostics(running_server, capsys):
    server, log_path, _ = running_server
    with server.log_on_failure("Unexpected failure"):
        pass
    server.__exit__(None, None, None)
    output = capsys.readouterr().out
    assert "[Failure]" not in output
    assert "Log preserved:" not in output
    assert not log_path.exists()


def test_skipped_request_does_not_preserve_log(running_server, capsys):
    server, log_path, _ = running_server
    with (
        pytest.raises(pytest.skip.Exception),
        server.log_on_failure("Skipped request"),
    ):
        pytest.skip("no applicable endpoint")
    server.__exit__(None, None, None)
    assert not log_path.exists()
    assert "[Failure]" not in capsys.readouterr().out


def test_context_failure_preserves_log(running_server, capsys):
    server, log_path, _ = running_server
    server.__exit__(RuntimeError, RuntimeError("engine failed"), None)
    assert log_path.exists()
    assert "EngineCore failed: injected crash" not in capsys.readouterr().out


def test_tail_is_bounded_but_complete_log_is_retained(running_server, capsys):
    server, log_path, _ = running_server
    server.emit_log_tail = True
    capsys.readouterr()
    server._log_file.write(b"x" * 20000 + b"\nFINAL_ENGINE_ERROR\n")
    server.preserve_failure_logs("First failure")
    server.preserve_failure_logs("Second failure")
    output = capsys.readouterr().out
    assert output.count("[Failure]") == 1
    assert "FINAL_ENGINE_ERROR" in output
    assert "SERVER_STARTED" not in output
    assert len(output) < 16500
    server.stop()
    assert "SERVER_STARTED" in log_path.read_text()
    assert "FINAL_ENGINE_ERROR" in log_path.read_text()


def test_startup_process_exit_preserves_log(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    process = Mock(returncode=42)
    process.poll.return_value = 42

    def launch(cmd, **kwargs):
        kwargs["stdout"].write(b"ENGINE_STARTUP_ERROR\n")
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    server = VllmServer(model="/models/Example-27B", max_retries=1)
    with pytest.raises(pytest.fail.Exception, match="code=42"):
        server.start()
    logs = list(Path("test-results/logs").glob("*.log"))
    assert len(logs) == 1
    assert "ENGINE_STARTUP_ERROR" in logs[0].read_text()
    assert "ENGINE_STARTUP_ERROR" in capsys.readouterr().out
    assert server._process is None
    assert server._log_file is None
    process.terminate.assert_called_once()


def test_launch_failure_closes_and_preserves_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess, "Popen", Mock(side_effect=FileNotFoundError("vllm"))
    )
    server = VllmServer(model="/models/Example-27B")
    with pytest.raises(FileNotFoundError, match="vllm"):
        server.start()
    assert len(list(Path("test-results/logs").glob("*.log"))) == 1
    assert server._process is None
    assert server._log_file is None


@pytest.fixture
def smoke_module(tmp_path, monkeypatch):
    """Import the real smoke test module without requiring model weights."""
    cfg = SimpleNamespace(
        model=str(tmp_path),
        serve=SimpleNamespace(
            endpoints=["chat"], request_model=lambda _: "Example-27B"
        ),
    )
    monkeypatch.setenv("FL_TEST_MODEL", "example")
    monkeypatch.setenv("FL_TEST_CASE", "case")
    monkeypatch.setattr(ModelConfig, "load", lambda *args, **kwargs: cfg)
    path = Path(__file__).parents[1] / "e2e_tests/serving/test_serving_smoke.py"
    spec = importlib.util.spec_from_file_location("_serving_smoke_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_streaming_runner_failure_captures_engine_log(
    smoke_module, running_server, monkeypatch, capsys
):
    server, log_path, _ = running_server
    runner = Mock(side_effect=RuntimeError("streaming connection closed"))
    monkeypatch.setitem(smoke_module._ENDPOINT_RUNNERS, "chat", runner)
    with pytest.raises(RuntimeError, match="streaming connection closed"):
        smoke_module.test_endpoint("chat", server, server.base_url, {})
    server.__exit__(None, None, None)
    assert log_path.exists()
    assert "EngineCore failed: injected crash" not in capsys.readouterr().out


def test_smoke_model_list_failure_captures_engine_log(
    smoke_module, running_server, monkeypatch, capsys
):
    server, log_path, _ = running_server
    monkeypatch.setattr(
        smoke_module.requests,
        "get",
        Mock(return_value=SimpleNamespace(status_code=503)),
    )
    with pytest.raises(AssertionError):
        smoke_module.test_model_list(server, server.base_url, {})
    server.__exit__(None, None, None)
    assert log_path.exists()
    assert "EngineCore failed: injected crash" not in capsys.readouterr().out

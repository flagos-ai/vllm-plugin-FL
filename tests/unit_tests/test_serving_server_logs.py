# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU regressions for retaining server diagnostics after serving failures."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import requests

from tests.e2e_tests.serving.server_helper import VllmServer
from tests.utils.model_config import ModelConfig, ServeConfig


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


@pytest.fixture(params=[False, True], ids=["standard", "adaptation-gate"])
def smoke_module(tmp_path, monkeypatch, request):
    """Import with real config defaults so the fixture tracks the serving contract."""
    cfg = ModelConfig(
        model=str(tmp_path),
        serve=ServeConfig(
            adaptation_gate=request.param,
            endpoints=["chat"],
            served_model_name="Example-27B",
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


def test_smoke_server_uses_config_defaults(smoke_module, monkeypatch):
    """Exercise the server fixture as well as import-time endpoint collection."""
    factory = MagicMock()
    monkeypatch.setattr(smoke_module, "VllmServer", factory)
    fixture = smoke_module.server.__wrapped__()
    assert next(fixture) is factory.return_value.__enter__.return_value
    args = factory.call_args.kwargs
    assert args["max_retries"] == 60
    assert args["served_model_name"] == "Example-27B"
    media_args = [
        "--allowed-local-media-path",
        str(smoke_module.GATE_ROOT / "images"),
    ]
    assert args["extra_args"] == (
        media_args if smoke_module._CFG.serve.adaptation_gate else []
    )
    with pytest.raises(StopIteration):
        next(fixture)
    factory.return_value.__exit__.assert_called_once_with(None, None, None)


def test_smoke_gate_skip_matches_config(smoke_module):
    marks = smoke_module.test_adaptation_gate.pytestmark
    skip = next(mark for mark in marks if mark.name == "skipif")
    assert skip.args == (not smoke_module._CFG.serve.adaptation_gate,)


def test_smoke_gate_failure_preserves_diagnostics(
    smoke_module, running_server, monkeypatch, tmp_path
):
    """Gate exceptions must retain diagnostics and use the shared server config."""
    from tests.utils import adaptation_gate

    server, log_path, _ = running_server
    monkeypatch.chdir(tmp_path)
    run_case = Mock(side_effect=RuntimeError("gate request failed"))
    monkeypatch.setattr(
        adaptation_gate, "load_gate", lambda: SimpleNamespace(run_case=run_case)
    )
    with pytest.raises(RuntimeError, match="gate request failed"):
        smoke_module.test_adaptation_gate("text_single", server, monkeypatch)
    run_case.assert_called_once_with("text_single")
    assert smoke_module.os.environ["SERVED_MODEL_NAME"] == "Example-27B"
    assert smoke_module.os.environ["MODEL_PATH"] == smoke_module._CFG.model
    server.stop()
    assert log_path.exists()
    assert "EngineCore failed: injected crash" in log_path.read_text()

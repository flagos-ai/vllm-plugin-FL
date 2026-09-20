# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU-only regressions for serving-test log retention."""

from pathlib import Path

import pytest

from tests.e2e_tests.serving import server_helper


@pytest.fixture
def server(monkeypatch):
    class FinishedAPI:
        pid = 123
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(server_helper, "_get_free_port", lambda: 12345)
    monkeypatch.setattr(
        server_helper.subprocess, "Popen", lambda *a, **kw: FinishedAPI()
    )
    monkeypatch.setattr(server_helper.VllmServer, "_wait_ready", lambda self: None)
    monkeypatch.setattr(
        server_helper.VllmServer, "_record_processes", lambda self: None
    )
    monkeypatch.setattr(server_helper.VllmServer, "_live_processes", lambda self: [])
    instance = server_helper.VllmServer(model="unused")
    yield instance
    if instance._process is not None:
        instance.stop(check_shutdown=False)


@pytest.mark.parametrize("relative", [False, True])
def test_configured_directory_retains_successful_log(
    server, monkeypatch, tmp_path, capsys, relative
):
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "artifacts" / "server-logs"
    monkeypatch.setenv(
        "VLLM_TEST_LOG_DIR", "artifacts/server-logs" if relative else str(log_dir)
    )
    assert not log_dir.exists()

    server.start()
    log_path = Path(server._log_file.name)
    server._log_file.write(b"successful server output\n")
    assert log_path.parent.resolve() == log_dir.resolve()
    assert log_path.name.startswith("vllm_unused_")
    assert log_path.suffix == ".log"
    assert str(log_path) in capsys.readouterr().out

    server.stop()

    assert log_path.read_bytes() == b"successful server output\n"
    assert f"Log preserved: {log_path}" in capsys.readouterr().out


@pytest.mark.parametrize("empty_value", [False, True])
def test_default_directory_deletes_successful_log(server, monkeypatch, empty_value):
    if empty_value:
        monkeypatch.setenv("VLLM_TEST_LOG_DIR", "")
    else:
        monkeypatch.delenv("VLLM_TEST_LOG_DIR", raising=False)

    server.start()
    log_path = Path(server._log_file.name)
    assert log_path.is_file()

    server.stop()

    assert not log_path.exists()


def test_configured_directory_retains_failure_before_cleanup(
    server, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("VLLM_TEST_LOG_DIR", str(tmp_path / "logs"))
    server.start()
    log_path = Path(server._log_file.name)
    server._log_file.write(b"failed server output\n")
    server._process.returncode = 1
    cleanup_calls = []

    def cleanup():
        # Recovery cannot erase the original lifecycle assertion.
        assert "[Teardown] FAILED:" in capsys.readouterr().out
        cleanup_calls.append(True)

    monkeypatch.setattr(server, "_cleanup_owned_processes", cleanup)
    with pytest.raises(pytest.fail.Exception, match="did not exit cleanly"):
        server.stop()

    assert cleanup_calls == [True]
    assert log_path.read_bytes() == b"failed server output\n"
    assert f"Log preserved: {log_path}" in capsys.readouterr().out

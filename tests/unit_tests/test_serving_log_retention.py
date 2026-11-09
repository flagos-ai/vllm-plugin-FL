# SPDX-License-Identifier: Apache-2.0
"""Preserve complete failed-service evidence across real pytest subprocesses."""

import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests.run import TestCase as Case, TestRunner as Runner


@pytest.fixture
def serving_runner(monkeypatch, tmp_path):
    # The child uses the real serving fixture/helper, with a Python log-writing
    # process in place of the model server. No accelerator or model is loaded.
    probe = tmp_path / "test_service_probe.py"
    probe.write_text(
        textwrap.dedent(
            r"""
            import importlib
            import sys
            from pathlib import Path
            from types import SimpleNamespace

            import pytest
            import requests

            from tests.e2e_tests.serving import server_helper
            from tests.utils.model_config import ModelConfig, ServeConfig

            @pytest.mark.parametrize("mode", ["startup", "endpoint", "success"])
            def test_service(mode, monkeypatch, tmp_path_factory):
                config = ModelConfig(
                    model=str(tmp_path_factory.getbasetemp()),
                    serve=ServeConfig(endpoints=["chat"], startup_retries=300),
                )
                monkeypatch.setenv("FL_TEST_MODEL", "log-probe")
                monkeypatch.setenv("FL_TEST_CASE", mode)
                monkeypatch.setattr(ModelConfig, "load", lambda *a, **kw: config)
                monkeypatch.setattr(
                    server_helper, "get_visible_device_env_var", lambda: "CPU_PROBE"
                )
                monkeypatch.setattr(server_helper, "_get_free_port", lambda: 31337)
                smoke = importlib.import_module(
                    "tests.e2e_tests.serving.test_serving_smoke"
                )
                real_popen = server_helper.subprocess.Popen
                processes = []
                handles = []

                def start_python(cmd, **kwargs):
                    script = (
                        "import sys,time; "
                        "print('PRIMARY_WORKER_ERROR', flush=True); "
                        "print('x' * 40000, flush=True); "
                        "print('WORKER_STDERR', file=sys.stderr, flush=True); "
                        + ("sys.exit(1)" if mode == "startup" else "time.sleep(60)")
                    )
                    handles.append(kwargs["stdout"])
                    process = real_popen([sys.executable, "-u", "-c", script], **kwargs)
                    processes.append(process)
                    return process

                def health(*args, **kwargs):
                    if mode == "startup" or not handles or "WORKER_STDERR" not in (
                        Path(handles[0].name).read_text()
                    ):
                        raise requests.ConnectionError("probe not ready")
                    return SimpleNamespace(status_code=200)

                monkeypatch.setattr(server_helper.subprocess, "Popen", start_python)
                monkeypatch.setattr(server_helper.requests, "get", health)
                # Keep the same real readiness/failure path, with short CPU polls.
                original_ready = server_helper.VllmServer._wait_ready
                def ready(server):
                    server.poll_interval = 0.01
                    return original_ready(server)
                monkeypatch.setattr(server_helper.VllmServer, "_wait_ready", ready)
                fixture = smoke.server.__wrapped__(tmp_path_factory)
                try:
                    next(fixture)
                    if mode == "endpoint":
                        pytest.fail("endpoint assertion sentinel")
                finally:
                    fixture.close()
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=5)
            """
        )
    )
    runner = Runner(
        "nvidia", device="a100", scope="e2e", output_dir=tmp_path / "reports"
    )
    monkeypatch.setattr(
        "tests.run.ModelConfig.load", lambda *a, **kw: SimpleNamespace(engine={})
    )
    monkeypatch.setattr("tests.run.wait_for_memory", lambda *a, **kw: (True, ""))
    monkeypatch.setattr(
        runner,
        "_build_pytest_cmd",
        lambda tc: [
            sys.executable,
            "-m",
            "pytest",
            str(probe),
            "-q",
            "-s",
            "--tb=short",
            "-k",
            tc.case,
            *tc.extra_args,
        ],
    )
    return runner


def _case(mode, extra_args=None):
    return Case(
        f"serving/probe/{mode}",
        "unused",
        task="serving",
        model="probe",
        case=mode,
        extra_args=extra_args or [],
    )


@pytest.mark.parametrize("mode", ["startup", "endpoint"])
def test_failed_case_retains_original_error_before_console_tail(serving_runner, mode):
    result = serving_runner._run_single(_case(mode), _quiet=True)
    assert not result.passed
    reason = (
        "process exited unexpectedly"
        if mode == "startup"
        else "endpoint assertion sentinel"
    )
    assert reason in result.stdout, result.stdout
    logs = list(serving_runner.output_dir.rglob("vllm_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert len(text) > 40000
    assert text.startswith("PRIMARY_WORKER_ERROR\n")
    assert text.endswith("WORKER_STDERR\n")
    assert "PRIMARY_WORKER_ERROR" not in result.stdout


def test_parallel_case_cleanup_preserves_failed_sibling(serving_runner, tmp_path):
    sentinel = tmp_path / "unrelated.log"
    sentinel.write_text("untouched")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda mode: serving_runner._run_single(_case(mode), _quiet=True),
                ("success", "endpoint"),
            )
        )
    assert results[0].passed, results[0].stdout
    assert not results[1].passed and "endpoint assertion sentinel" in results[1].stdout
    root = serving_runner.output_dir / "test-results-nvidia-logs"
    assert len(list(root.iterdir())) == 1
    logs = list(root.rglob("vllm_*.log"))
    assert len(logs) == 1 and "PRIMARY_WORKER_ERROR" in logs[0].read_text()
    assert sentinel.read_text() == "untouched"


@pytest.mark.parametrize("equals", [False, True])
def test_explicit_pytest_basetemp_remains_caller_owned(
    serving_runner, tmp_path, equals
):
    owned = tmp_path / "caller-owned"
    args = [f"--basetemp={owned}"] if equals else ["--basetemp", str(owned)]
    result = serving_runner._run_single(_case("success", args), _quiet=True)
    assert result.passed, result.stdout
    assert list(owned.rglob("vllm_*.log"))
    assert not serving_runner.output_dir.exists()


def test_nonexecuted_serving_cases_create_no_log_directories(
    serving_runner, monkeypatch
):
    serving_runner.dry_run = True
    assert serving_runner._run_single(_case("success")).passed
    serving_runner.dry_run = False
    monkeypatch.setattr("tests.run.wait_for_memory", lambda *a, **kw: (False, "busy"))
    result = serving_runner._run_single(_case("success"))
    assert not result.passed and "device memory" in result.message
    assert not serving_runner.output_dir.exists()

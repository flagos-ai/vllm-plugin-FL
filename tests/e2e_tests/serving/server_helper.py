# Copyright (c) 2025 BAAI. All rights reserved.

"""
Shared vLLM server lifecycle helper for serving E2E tests.

Provides ``VllmServer`` — a context manager that starts a vLLM serve
process, waits for readiness, and tears it down on exit.

Set ``VLLM_TEST_LOG_DIR`` to retain every server log in an artifact directory,
including logs from successful tests. Otherwise, successful logs are temporary.

Usage in test fixtures::

    @pytest.fixture(scope="module")
    def vllm_server():
        with VllmServer(model=MODEL_PATH, tp_size=8) as srv:
            yield srv


    def test_completion(vllm_server):
        resp = requests.post(f"{vllm_server.base_url}/completions", ...)
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import psutil
import pytest
import requests

# Bypass HTTP proxies for local server connections.
_NO_PROXY = {"http": None, "https": None}


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class VllmServer:
    """Manages a vLLM serving process lifecycle."""

    model: str
    tp_size: int = 1
    extra_args: list[str] = field(default_factory=list)
    host: str = "127.0.0.1"
    api_key: str = ""
    served_model_name: str = ""
    max_retries: int = 60
    poll_interval: int = 10
    shutdown_timeout: float = 45.0

    # Set after start
    port: int = 0
    base_url: str = ""
    _process: subprocess.Popen | None = None
    _log_file: tempfile._TemporaryFileWrapper | None = None
    _owned_processes: dict[int, psutil.Process] = field(default_factory=dict)

    def start(self) -> None:
        self.port = _get_free_port()
        self.base_url = f"http://{self.host}:{self.port}/v1"

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--tensor-parallel-size",
            str(self.tp_size),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        if self.served_model_name:
            cmd.extend(["--served-model-name", self.served_model_name])
        cmd.extend(self.extra_args)

        model_short = os.path.basename(self.model)
        print(f"\n[Setup] Starting vLLM ({model_short}, TP={self.tp_size})")
        print(f"[Setup] Command: {' '.join(cmd)}")

        log_dir = os.environ.get("VLLM_TEST_LOG_DIR") or None
        self._keep_log = log_dir is not None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix=f"vllm_{model_short}_",
            suffix=".log",
            delete=False,
            dir=log_dir,
        )
        print(f"[Setup] Server log: {self._log_file.name}")
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            # Match terminal Ctrl+C without signalling the pytest runner.
            start_new_session=True,
        )
        self._record_processes()
        self._wait_ready()

    def _record_processes(self) -> None:
        """Remember process identities, including children already re-parented.

        Every child inherits this server's isolated session. Recording session
        members also finds workers whose parent exited before the next poll.
        psutil caches creation time so a reused PID is never treated as ours.
        """
        if self._process is None:
            return
        # Include descendants that create their own session as well.
        for parent in list(self._owned_processes.values()):
            try:
                if parent.is_running():
                    for child in parent.children(recursive=True):
                        child.create_time()
                        self._owned_processes.setdefault(child.pid, child)
            except psutil.NoSuchProcess:
                continue
        for process in psutil.process_iter():
            try:
                if os.getsid(process.pid) != self._process.pid:
                    continue
                process.create_time()
                self._owned_processes.setdefault(process.pid, process)
            except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                continue

    def _live_processes(self) -> list[psutil.Process]:
        live = []
        for process in self._owned_processes.values():
            try:
                # Zombies no longer own device contexts. Direct children are
                # reaped by Popen.wait(); worker reaping has separate unit tests.
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process)
            except psutil.NoSuchProcess:
                continue
        return live

    def _cleanup_owned_processes(self) -> None:
        """Failure cleanup only; never search or kill by process name."""
        self._record_processes()
        remaining = self._live_processes()
        for process in remaining:
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()
        api_pid = self._process.pid if self._process is not None else None
        if self._process is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=5)
        # Popen owns waitpid for the API child. Letting psutil reap it first
        # would make Popen report zero after ChildProcessError, hiding SIGKILL.
        psutil.wait_procs(
            [process for process in remaining if process.pid != api_pid], timeout=5
        )

    def stop(self, *, check_shutdown: bool = True) -> None:
        if self._process is None:
            return
        print("\n[Teardown] Shutting down vLLM service...")
        failure = ""
        try:
            self._record_processes()
            if self._live_processes():
                # Signal this server's entire foreground group, as Ctrl+C does.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._process.pid, signal.SIGINT)
            deadline = time.monotonic() + self.shutdown_timeout
            while True:
                self._process.poll()  # reap the direct API process
                self._record_processes()
                remaining = self._live_processes()
                if not remaining:
                    self._process.wait(timeout=1)
                    break
                if time.monotonic() >= deadline:
                    failure = (
                        "vLLM shutdown left live processes before test cleanup: "
                        f"{[(p.pid, p.name()) for p in remaining]}"
                    )
                    break
                time.sleep(0.1)
            if not failure and self._process.returncode != 0:
                failure = (
                    "vLLM did not exit cleanly after process-group SIGINT: "
                    f"returncode={self._process.returncode}"
                )
            print(
                f"[Teardown] SIGINT result: returncode={self._process.returncode}, "
                f"tracked_pids={sorted(self._owned_processes)}, "
                f"live_pids={[p.pid for p in remaining]}"
            )
        except Exception as exc:
            failure = f"Could not validate vLLM shutdown: {exc!r}"
        finally:
            if failure:
                # Record failure before any recovery: killing leaked workers
                # must never turn a failed lifecycle check into a passing test.
                print(f"[Teardown] FAILED: {failure}")
                self._keep_log = True
                self._cleanup_owned_processes()
            if self._log_file:
                self._log_file.close()
                if getattr(self, "_keep_log", False):
                    print(f"[Teardown] Log preserved: {self._log_file.name}")
                    if failure:
                        failure += f"\nFull log: {self._log_file.name}"
                else:
                    os.unlink(self._log_file.name)
            self._process = None
            self._owned_processes.clear()
        if failure and check_shutdown:
            pytest.fail(failure, pytrace=False)

    def __enter__(self) -> VllmServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Use /health for readiness check — it returns 200 once the engine
        # core IPC channel is established.  /v1/models may return 503 for
        # a long time in vLLM V1's multi-process architecture even after
        # the HTTP server is accepting connections.
        health_url = f"http://{self.host}:{self.port}/health"
        print(f"[Setup] Waiting for service to be ready (polling {health_url})...")
        for i in range(self.max_retries):
            self._record_processes()
            if self._process.poll() is not None:
                self._fail_with_logs(
                    "vLLM process exited unexpectedly "
                    f"(code={self._process.returncode})"
                )

            try:
                resp = requests.get(
                    health_url, headers=headers, timeout=10, proxies=_NO_PROXY
                )
                if resp.status_code == 200:
                    print(f"[Setup] vLLM service ready (port={self.port})")
                    return
                detail = ""
                with contextlib.suppress(Exception):
                    detail = f" body={resp.text[:200]}"
                print(
                    f"[Setup] Waiting ({i + 1}/{self.max_retries})"
                    f" status={resp.status_code}{detail}"
                )
            except requests.exceptions.RequestException as exc:
                print(
                    f"[Setup] Waiting ({i + 1}/{self.max_retries}) {type(exc).__name__}"
                )

            time.sleep(self.poll_interval)

        self._fail_with_logs("vLLM service startup timed out")

    def _fail_with_logs(self, message: str) -> None:
        logs = ""
        log_path = ""
        if self._log_file:
            self._log_file.flush()
            log_path = self._log_file.name
            with open(log_path) as f:
                logs = f.read()
        # Keep log file for post-mortem inspection
        self._keep_log = True
        self.stop(check_shutdown=False)
        tail = logs[-16000:] if len(logs) > 16000 else logs
        extra = f"\nFull log: {log_path}" if log_path else ""
        pytest.fail(
            f"{message}.{extra}\nLogs ({len(logs)} chars, showing tail):\n{tail}"
        )

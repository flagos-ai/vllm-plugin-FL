# Copyright (c) 2025 BAAI. All rights reserved.

"""
Shared vLLM server lifecycle helper for serving E2E tests.

Provides ``VllmServer`` — a context manager that starts a vLLM serve
process, waits for readiness, and tears it down on exit.

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
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    emit_log_tail: bool = False

    # Set after start
    port: int = 0
    base_url: str = ""
    _process: subprocess.Popen | None = None
    _log_file: tempfile._TemporaryFileWrapper | None = None
    _keep_log: bool = False
    _failure_logged: bool = False

    def start(self) -> None:
        self._keep_log = False
        self._failure_logged = False
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

        log_dir = Path("test-results/logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model_short)
        self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix=f"vllm_{log_model}_tp{self.tp_size}_{self.port}_",
            suffix=".log",
            dir=log_dir,
            delete=False,
        )
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
            )
            self._wait_ready()
        except BaseException:
            self.preserve_failure_logs("vLLM service startup failed")
            self.stop()
            raise

    def stop(self) -> None:
        try:
            if self._process is not None:
                print("\n[Teardown] Shutting down vLLM service...")
                # Send SIGTERM to the main process; vLLM handles child cleanup.
                self._process.terminate()
                self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        except Exception:
            self._process.kill()
        finally:
            if self._log_file:
                log_path = Path(self._log_file.name)
                self._log_file.close()
                self._log_file = None
                if self._keep_log:
                    # The child has stopped: redact the artifact without racing
                    # its stdout writes or changing its active file offset.
                    try:
                        logs = log_path.read_text(encoding="utf-8", errors="replace")
                        log_path.write_text(self._redact(logs), encoding="utf-8")
                    except OSError as exc:
                        print(f"[Teardown] Could not redact log: {type(exc).__name__}")
                    print(f"[Teardown] Log preserved: {log_path}")
                else:
                    log_path.unlink()
            self._process = None

    def __enter__(self) -> VllmServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is not None and not isinstance(exc_value, pytest.skip.Exception):
            self.preserve_failure_logs("vLLM serving context failed")
        self.stop()

    @contextlib.contextmanager
    def log_on_failure(self, message: str):
        """Capture request/assertion failures before a yield fixture tears down.

        Pytest yield fixtures resume normally even when their test failed, so
        ``__exit__`` alone cannot detect serving smoke test failures.
        """
        try:
            yield
        except pytest.skip.Exception:
            raise
        except BaseException:
            self.preserve_failure_logs(message)
            raise

    def _redact(self, text: str) -> str:
        keys = [self.api_key] if self.api_key else []
        for i, arg in enumerate(self.extra_args):
            if arg == "--api-key" and i + 1 < len(self.extra_args):
                keys.append(self.extra_args[i + 1])
            elif arg.startswith("--api-key="):
                keys.append(arg.split("=", 1)[1])
        for key in keys:
            if key:
                text = text.replace(key, "[REDACTED]")
        return text

    def preserve_failure_logs(
        self, message: str, *, include_tail: bool | None = None
    ) -> str:
        """Retain logs locally; only emit their contents when explicitly enabled.

        Post-start server output may contain sensitive data. Tests only receive
        its local path by default, so files can be reviewed before sharing.
        Startup readiness failures retain their existing diagnostic tail.
        """
        self._keep_log = True
        logs = ""
        log_path = ""
        emit_tail = self.emit_log_tail if include_tail is None else include_tail
        if self._log_file:
            log_path = self._log_file.name
            if emit_tail:
                try:
                    self._log_file.flush()
                    with open(log_path, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        f.seek(max(0, f.tell() - 16000))
                        logs = f.read().decode("utf-8", errors="replace")
                except OSError as exc:
                    logs = f"Unable to read server log: {type(exc).__name__}"
        details = f"{message}.\nFull log: {log_path or 'unavailable'}"
        if emit_tail:
            details += f"\nServer log tail:\n{logs}"
        details = self._redact(details)
        if not self._failure_logged:
            print(f"\n[Failure] {details}")
            self._failure_logged = True
        return details

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
        details = self.preserve_failure_logs(message, include_tail=True)
        self.stop()
        pytest.fail(details, pytrace=False)

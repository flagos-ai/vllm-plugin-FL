# Copyright (c) 2026 BAAI. All rights reserved.

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def test_gate_readiness_deadline_handles_a_stalled_http_server(tmp_path):
    if os.name != "posix" or not shutil.which("bash") or not shutil.which("curl"):
        import pytest

        pytest.skip("The Gate readiness script requires POSIX bash and curl")

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(10)
    port = listener.getsockname()[1]
    accepted = threading.Event()
    release = threading.Event()

    def stall_response():
        try:
            connection, _ = listener.accept()
            with connection:
                accepted.set()
                release.wait(15)
        except OSError:
            pass

    thread = threading.Thread(target=stall_response, daemon=True)
    thread.start()
    gate_dir = Path(__file__).resolve().parents[2] / "tools/adaptation-gate-cases"
    script = gate_dir / "run_test.sh"
    env = dict(
        os.environ,
        MODEL_PATH="",
        SERVED_MODEL_NAME="readiness-test",
        PORT=str(port),
        BASE_URL=f"http://127.0.0.1:{port}/v1",
        SERVICE_TIMEOUT="1",
        SERVER_PID_FILE=str(tmp_path / "unused.pid"),
    )
    process = subprocess.Popen(
        ["bash", str(script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        release.set()
        listener.close()
        thread.join(timeout=2)

    assert accepted.is_set(), "The probe must reach the stalled HTTP server"
    assert process.returncode == 1, stdout + stderr
    assert "Service was not ready within 1s" in stdout + stderr


def test_gate_readiness_detects_an_unreaped_service_exit(tmp_path):
    if not Path("/proc/self/stat").is_file() or not shutil.which("ps"):
        import pytest

        pytest.skip("Zombie detection regression requires Linux procfs and ps")

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    server = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        for _ in range(100):
            state = Path(f"/proc/{server.pid}/stat").read_text().split(") ", 1)[1]
            if state.startswith("Z "):
                break
            time.sleep(0.05)
        assert state.startswith("Z "), "Service must exit without being reaped"
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(server.pid))
        script = (
            Path(__file__).resolve().parents[2]
            / "tools/adaptation-gate-cases/run_test.sh"
        )
        result = subprocess.run(
            ["bash", str(script)],
            env=dict(
                os.environ,
                SERVED_MODEL_NAME="readiness-test",
                PORT=str(port),
                BASE_URL=f"http://127.0.0.1:{port}/v1",
                SERVICE_TIMEOUT="30",
                SERVER_PID_FILE=str(pid_file),
            ),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert f"Service process {server.pid} exited" in result.stdout + result.stderr
    finally:
        server.wait(timeout=5)

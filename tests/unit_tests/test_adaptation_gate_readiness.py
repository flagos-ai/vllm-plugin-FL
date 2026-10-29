# Copyright (c) 2026 BAAI. All rights reserved.

import os
import shutil
import signal
import socket
import subprocess
import threading
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

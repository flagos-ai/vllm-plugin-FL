# Copyright (c) 2025 BAAI. All rights reserved.

"""
vLLM serving tests for MiniCPM model.
Tests HTTP API with MiniCPM multimodal model.
"""

import json
import os
import signal
import socket
import subprocess
import tempfile
import time

import pytest
import requests

MODEL_PATH = "/data/models/MiniCPM"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}"
)
API_KEY = "token-abc123"
HOST = "127.0.0.1"


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module", autouse=True)
def vllm_server():
    port = _get_free_port()
    base_url = f"http://{HOST}:{port}/v1"

    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--dtype",
        "auto",
        "--max-model-len",
        "2048",
        "--tensor-parallel-size",
        "8",
        "--api-key",
        API_KEY,
        "--gpu-memory-utilization",
        "0.9",
        "--trust-remote-code",
        "--max-num-batched-tokens",
        "2048",
        "--load-format",
        "dummy",
        "--host",
        HOST,
        "--port",
        str(port),
    ]

    print(f"\n[Setup] Starting vLLM service: {' '.join(cmd)}")
    log_file = tempfile.NamedTemporaryFile(prefix="vllm_minicpm_", suffix=".log", delete=False)
    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid
    )

    max_retries = 60
    ready = False
    probe_headers = {"Authorization": f"Bearer {API_KEY}"}
    print("[Setup] Waiting for service to be ready...")
    for i in range(max_retries):
        if process.poll() is not None:
            log_file.flush()
            log_path = log_file.name
            with open(log_path) as f:
                logs = f.read()
            pytest.fail(
                f"vLLM process exited unexpectedly (code={process.returncode}).\n"
                f"Full log: {log_path}\n"
                f"Logs (last 8000 chars):\n{logs[-8000:]}"
            )
        try:
            response = requests.get(
                f"{base_url}/models", headers=probe_headers, timeout=5
            )
            if response.status_code == 200:
                ready = True
                print(
                    f"\n[Response Content]: "
                    f"{json.dumps(response.json(), indent=2, ensure_ascii=False)}"
                )
                print(f"\n[Setup] vLLM service is ready! (port={port})")
                break
        except requests.exceptions.ConnectionError:
            pass

        print(f"[Setup] Waiting for service to start ({i + 1}/{max_retries})...")
        time.sleep(5)

    if not ready:
        log_file.flush()
        log_path = log_file.name
        with open(log_path) as f:
            logs = f.read()
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except Exception:
            process.kill()
        pytest.fail(
            f"vLLM service startup timed out.\n"
            f"Full log: {log_path}\n"
            f"Logs (last 8000 chars):\n{logs[-8000:]}"
        )

    yield {"base_url": base_url, "process": process}

    # --- Teardown ---
    print("\n[Teardown] Shutting down vLLM service...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:
        process.kill()
    finally:
        log_file.close()
        os.unlink(log_file.name)


@pytest.fixture
def base_url(vllm_server):
    return vllm_server["base_url"]


@pytest.fixture
def headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def test_chat_completion(base_url, headers):
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": "Running test."}],
        "max_tokens": 100,
    }
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    assert response.status_code == 200
    print(f"Response: {response.json()['choices'][0]['message']['content']}")

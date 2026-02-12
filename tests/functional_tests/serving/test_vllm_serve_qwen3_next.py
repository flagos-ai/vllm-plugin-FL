# Copyright (c) 2025 BAAI. All rights reserved.

"""
vLLM serving tests for Qwen3-Next model.
Tests HTTP API with tensor parallelism (TP=8).
"""

import os
import signal
import socket
import subprocess
import tempfile
import time

import pytest
import requests

MODEL_PATH = "/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}"
)
HOST = "127.0.0.1"
TP_SIZE = 8


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
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "512",
        "--gpu-memory-utilization",
        "0.85",
        "--host",
        HOST,
        "--port",
        str(port),
    ]

    print(f"\n[Setup] Starting vLLM service (TP={TP_SIZE}): {' '.join(cmd)}")
    log_file = tempfile.NamedTemporaryFile(prefix="vllm_qwen3_next_", suffix=".log", delete=False)
    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid
    )

    max_retries = 60
    ready = False
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
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                ready = True
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


def test_api_completion(base_url):
    payload = {
        "model": MODEL_PATH,
        "prompt": "please introduce yourself",
        "max_tokens": 20,
        "temperature": 0,
    }

    response = requests.post(f"{base_url}/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data

    generated_text = data["choices"][0]["text"]
    print(f"\nGenerated text: {generated_text}")


def test_model_list(base_url):
    response = requests.get(f"{base_url}/models")
    assert response.status_code == 200
    models = response.json()["data"]
    assert any(m["id"] == MODEL_PATH for m in models)

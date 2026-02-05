import pytest
import subprocess
import requests
import time
import os
import signal

MODEL_PATH = "/models/Qwen3-Next-80B-A3B-Instruct"
TP_SIZE = 4
PORT = 8000
BASE_URL = f"http://localhost:{PORT}/v1"


@pytest.fixture(scope="module", autouse=True)
def vllm_server():
    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "2048",
        "--gpu-memory-utilization",
        "0.85",
        "--port",
        str(PORT),
    ]

    print(f"\n[Setup] Starting the vLLM model (TP={TP_SIZE})...")

    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=os.setsid
    )

    max_retries = 60
    ready = False
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/models", timeout=2)
            if response.status_code == 200:
                ready = True
                print(f"\n[Setup] vLLM service is ready (take time {i * 5}s)")
                break
        except Exception:
            pass
        time.sleep(20)
        print(f"Waiting ... ({i + 1}/{max_retries})")

    if not ready:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        pytest.fail("vLLM service startup timeout")

    yield process

    print("\n[Teardown] The vLLM service is being shut down ..")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.wait(timeout=30)


def test_api_completion():
    payload = {
        "model": MODEL_PATH,
        "prompt": "please introduce yourself",
        "max_tokens": 20,
        "temperature": 0,
    }

    response = requests.post(f"{BASE_URL}/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data

    generated_text = data["choices"][0]["text"]
    print(f"\nGenerated text: {generated_text}")


def test_model_list():
    response = requests.get(f"{BASE_URL}/models")
    assert response.status_code == 200
    models = response.json()["data"]
    assert any(m["id"] == MODEL_PATH for m in models)

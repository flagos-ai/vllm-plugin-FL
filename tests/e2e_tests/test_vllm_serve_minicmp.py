import pytest
import requests
import subprocess
import time
import json

MODEL_PATH = "/models/MiniCMP"
API_KEY = "token-abc123"
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}/v1/chat/completions"


@pytest.fixture(scope="session", autouse=True)
def vllm_server():
    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--dtype",
        "auto",
        "--max-model-len",
        "2048",
        "--api-key",
        API_KEY,
        "--gpu_memory_utilization",
        "0.9",
        "--trust-remote-code",
        "--max-num-batched-tokens",
        "2048",
        "--load-format",
        "dummy",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    print(f"\n[Setup] The vllm service is being started: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True
    )

    max_retries = 30
    ready = False
    probe_headers = {"Authorization": f"Bearer {API_KEY}"}
    print("[Setup] Waiting for service to be ready...")
    for i in range(max_retries):
        if process.poll() is not None:
            pytest.fail(
                "The vLLM process failed to start. Please manually run the command to check for errors."
            )
        try:
            response = requests.get(
                f"http://{HOST}:{PORT}/v1/models", headers=probe_headers, timeout=100
            )
            print(
                f"\n[Response Content]: {json.dumps(response.json(), indent=2, ensure_ascii=False)}"
            )
            if response.status_code == 200:
                ready = True
                print("\n[Setup] vLLM service is ready!")
                break
            else:
                print(f"Error Detail: {response.text}")
        except requests.exceptions.ConnectionError:
            pass

        print(f"[Setup] Waiting for service to start ({i + 1}/{max_retries})...")
        time.sleep(20)

    if not ready:
        process.terminate()
        pytest.fail(
            "The startup of the vLLM service has timed out. Please check the GPU resources or logs."
        )

    yield

    # --- Teardown ---
    print("\n[Teardown] The vLLM service is being shut down ..")
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture
def headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def test_chat_completion(headers):
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": "Running test."}],
        "max_tokens": 100,
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    assert response.status_code == 200
    print(f"Response: {response.json()['choices'][0]['message']['content']}")

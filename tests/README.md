# Tests

This directory contains unit tests and functional tests for the vllm-plugin-FL project.

## Directory Structure

```
tests/
├── unit_tests/                  # Fast, isolated tests (no GPU/model required)
│   ├── conftest.py              # Shared fixtures (mock tensors, devices, etc.)
│   ├── dispatch/                # Op dispatch system tests
│   │   ├── test_call_op.py
│   │   ├── test_discovery.py
│   │   ├── test_manager.py
│   │   ├── test_policy.py
│   │   ├── test_registry.py
│   │   └── test_types.py
│   ├── distributed/             # Distributed communication tests
│   │   ├── test_communicator.py
│   │   └── test_flagcx.py
│   ├── compilation/             # Graph compilation tests
│   │   └── test_graph.py
│   ├── ops/                     # Operator unit tests
│   │   ├── test_activation.py
│   │   ├── test_layernorm.py
│   │   ├── test_numerical.py
│   │   └── test_rotary_embedding.py
│   ├── worker/                  # Worker tests
│   │   ├── test_model_runner.py
│   │   └── test_worker.py
│   └── flaggems/                # FlagGems integration tests
│       ├── test_gems_whitelist.py
│       └── test_flaggems_get_ops.py
│
└── functional_tests/            # End-to-end tests (require GPU and models)
    ├── conftest.py              # Shared fixtures (device, markers)
    ├── inference/               # Offline inference tests
    │   ├── vllm_runner.py       # VllmRunner test utility
    │   ├── test_offline_qwen3_06b.py
    │   ├── test_offline_qwen3_next.py
    │   └── test_offline_minicpm.py
    ├── serving/                 # HTTP API serving tests
    │   ├── test_vllm_serve_qwen3_next.py
    │   └── test_vllm_serve_minicpm.py
    ├── ops/                     # Operator correctness tests
    │   └── test_ops_correctness.py
    ├── compilation/             # CUDA graph capture tests
    │   └── test_graph_capture.py
    └── distributed/             # Multi-GPU collective ops tests
        └── test_collective_ops.py
```

## Prerequisites

```bash
# Install the project in development mode
pip install -e .

# For MiniCPM audio tests
pip install vllm[audio]
```

## Running Tests

### Run all unit tests (no GPU required)

```bash
pytest tests/unit_tests/
```

### Run all functional tests (requires GPU and models)

```bash
pytest tests/functional_tests/
```

### Run specific test categories

```bash
# Inference tests only
pytest tests/functional_tests/inference/

# Serving tests only
pytest tests/functional_tests/serving/

# A single test file
pytest tests/functional_tests/inference/test_offline_qwen3_next.py

# A single test function
pytest tests/functional_tests/inference/test_offline_qwen3_next.py::test_basic_generation
```

### Filter by markers

```bash
# Only GPU tests
pytest -m gpu

# Skip slow tests
pytest -m "not slow"
```

### Useful options

```bash
# Verbose output with print statements visible
pytest -v -s tests/functional_tests/inference/

# Stop on first failure
pytest -x tests/unit_tests/

# Run tests matching a keyword
pytest -k "minicpm" tests/

# Show coverage report
pytest --cov=vllm_fl tests/unit_tests/
```

## Model Path Requirements

Functional tests require model weights at specific paths. Tests will be **automatically skipped** if the model path does not exist.

| Test File | Model Path | GPU Requirement |
|---|---|---|
| `test_offline_qwen3_06b.py` | `/data/models/Qwen/Qwen3-0.6B` | 1 GPU |
| `test_offline_qwen3_next.py` | `/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct` | 8 GPUs (TP=8) |
| `test_offline_minicpm.py` | `/data/models/MiniCPM` | 2 GPUs (TP=2) |
| `test_vllm_serve_qwen3_next.py` | `/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct` | 8 GPUs (TP=8) |
| `test_vllm_serve_minicpm.py` | `/data/models/MiniCPM` | 8 GPUs (TP=8) |

## Writing New Tests

### Adding a unit test

Unit tests should be fast, isolated, and not require GPU or model weights. Place them under `tests/unit_tests/` in the appropriate subdirectory.

```python
# tests/unit_tests/ops/test_my_op.py

import pytest
import torch


class TestMyOp:
    def test_basic(self, device):
        """The `device` fixture is provided by conftest.py."""
        x = torch.randn(4, 8, device=device)
        # ... test logic ...
        assert result.shape == expected_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_specific(self):
        """Test that requires GPU."""
        # ...
```

### Adding an inference test

Inference tests validate offline model generation. Place them under `tests/functional_tests/inference/`.

```python
# tests/functional_tests/inference/test_offline_my_model.py

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_PATH = "/data/models/MyModel"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}"
)


@pytest.fixture(scope="module")
def llm_instance():
    """Create LLM once per module to avoid repeated initialization."""
    return LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
    )


@pytest.fixture
def default_params():
    return SamplingParams(max_tokens=10, temperature=0.0)


def test_basic_generation(llm_instance, default_params):
    outputs = llm_instance.generate(["Hello, world"], default_params)
    assert len(outputs) > 0
    generated_text = outputs[0].outputs[0].text
    assert len(generated_text) > 0
```

Key conventions:
- Use `pytestmark` with `os.path.exists()` to skip when model is unavailable
- Use `scope="module"` fixtures for expensive objects like `LLM` instances
- Use `SamplingParams(temperature=0.0)` for deterministic output in assertions

### Adding a serving test

Serving tests start a vLLM HTTP server and validate API endpoints. Place them under `tests/functional_tests/serving/`.

```python
# tests/functional_tests/serving/test_vllm_serve_my_model.py

import os
import signal
import socket
import subprocess
import tempfile
import time

import pytest
import requests

MODEL_PATH = "/data/models/MyModel"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}"
)
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
        "vllm", "serve", MODEL_PATH,
        "--host", HOST,
        "--port", str(port),
        "--gpu-memory-utilization", "0.85",
    ]

    log_file = tempfile.NamedTemporaryFile(
        prefix="vllm_my_model_", suffix=".log", delete=False
    )
    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Wait for server to be ready
    max_retries = 60
    ready = False
    for i in range(max_retries):
        if process.poll() is not None:
            log_file.flush()
            with open(log_file.name) as f:
                logs = f.read()
            pytest.fail(
                f"vLLM process exited unexpectedly (code={process.returncode}).\n"
                f"Full log: {log_file.name}\n"
                f"Logs (last 8000 chars):\n{logs[-8000:]}"
            )
        try:
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)

    if not ready:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        pytest.fail("vLLM service startup timed out.")

    yield {"base_url": base_url, "process": process}

    # Teardown
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


def test_completions(base_url):
    payload = {
        "model": MODEL_PATH,
        "prompt": "Hello",
        "max_tokens": 10,
    }
    response = requests.post(f"{base_url}/completions", json=payload)
    assert response.status_code == 200
    assert "choices" in response.json()
```

Key conventions:
- Use `_get_free_port()` to avoid port conflicts
- Use `preexec_fn=os.setsid` + `os.killpg()` for reliable process cleanup
- Capture logs to a temp file; print on failure for debugging
- Use `scope="module"` so the server starts once per test file

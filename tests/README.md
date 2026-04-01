# Tests

Platform-agnostic test suite for the vllm-plugin-FL project. All tests are designed to run on both NVIDIA CUDA and Huawei Ascend NPU without modification.

## Architecture Overview

```
                          ┌──────────────┐
                          │  tests/run.py │  ← Unified entry point
                          └──────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
        ┌─────▼─────┐   ┌───────▼───────┐   ┌──────▼──────┐
        │ Unit Tests │   │  Functional   │   │  E2E Tests  │
        │ (no GPU)   │   │  (GPU, no     │   │ (GPU +      │
        │            │   │   models)     │   │  models)    │
        └────────────┘   └───────────────┘   └──────┬──────┘
                                                     │
                                              ┌──────┴──────┐
                                              │             │
                                        ┌─────▼────┐ ┌─────▼─────┐
                                        │Inference │ │ Serving   │
                                        │(LLM API) │ │(HTTP API) │
                                        └──────────┘ └───────────┘
```

**Key design principles:**

- **YAML-driven E2E tests**: Model configs in `tests/models/` drive both inference and serving smoke tests — no per-model test files needed.
- **Platform-aware orchestration**: Platform YAML configs (`tests/platforms/`) control which tests run on which hardware.
- **Process isolation**: Each E2E test case runs in its own subprocess via `run.py`, with device cleanup between cases.
- **Backend-agnostic fixtures**: The `device` fixture returns the correct device (`cuda:0` or `npu:0`) based on detected hardware.

## Directory Structure

```
tests/
├── run.py                          # Unified test runner (wraps pytest)
├── conftest.py                     # Root fixtures: device, tolerance, markers
├── __init__.py
│
├── models/                         # Model YAML configs (drive E2E tests)
│   ├── qwen3/
│   │   ├── 4b_tp2.yaml
│   │   ├── 06b_tp2.yaml
│   │   └── next_tp8.yaml
│   ├── qwen3_5/
│   │   └── 35b_tp4.yaml
│   └── minicpm/
│       └── o45_tp4.yaml
│
├── platforms/                      # Platform-specific test configs
│   ├── cuda.yaml                   # NVIDIA GPU: device types, tolerance, test matrix
│   ├── ascend.yaml                 # Huawei NPU: device types, tolerance, test matrix
│   └── template.yaml               # Template for adding new platforms
│
├── e2e_tests/                      # End-to-end tests (require GPU + model files)
│   ├── inference/
│   │   └── test_inference_smoke.py # Unified inference test (YAML-driven)
│   └── serving/
│       ├── test_serving_smoke.py   # Unified serving test (YAML-driven)
│       └── server_helper.py        # VllmServer lifecycle manager
│
├── functional_tests/               # Component-level GPU tests (no model files)
│   ├── conftest.py
│   ├── ops/
│   │   └── test_ops_correctness.py # Operator correctness vs reference impls
│   ├── compilation/
│   │   └── test_graph_capture.py   # Graph capture & replay (CUDA + NPU)
│   └── distributed/
│       └── test_collective_ops.py  # Collective ops (all_reduce, etc.)
│
├── unit_tests/                     # Fast isolated tests (no GPU required)
│   ├── conftest.py
│   ├── dispatch/                   # Op dispatch system
│   │   ├── test_call_op.py
│   │   ├── test_discovery.py
│   │   ├── test_manager.py
│   │   ├── test_policy.py
│   │   ├── test_registry.py
│   │   ├── test_types.py
│   │   ├── test_io_dumper.py
│   │   └── test_io_inspector.py
│   ├── distributed/                # Distributed communication
│   │   ├── test_communicator.py
│   │   └── test_flagcx.py
│   ├── compilation/                # Graph compilation
│   │   └── test_graph.py
│   ├── ops/                        # Operator unit tests
│   │   ├── test_activation.py
│   │   ├── test_layernorm.py
│   │   ├── test_numerical.py
│   │   └── test_rotary_embedding.py
│   ├── worker/                     # Worker & model runner
│   │   ├── test_worker.py
│   │   ├── test_model_runner.py
│   │   └── test_model_imports.py
│   └── flaggems/                   # FlagGems integration
│       ├── test_gems_whitelist.py
│       └── test_flaggems_get_ops.py
│
└── utils/                          # Shared test utilities
    ├── device_utils.py             # Backend detection & device abstraction
    ├── model_config.py             # YAML model config loader
    ├── platform_config.py          # Platform YAML config loader
    ├── cleanup.py                  # Device cleanup between E2E cases
    ├── gold.py                     # Gold-value comparison
    └── report.py                   # Test result reporting (JUnit XML + JSON)
```

## Running Tests

### Via `run.py` (recommended for CI and full runs)

`run.py` is the unified entry point that handles platform config, tolerance injection, test discovery, and structured reporting.

```bash
# Run all tests for a platform/device
python tests/run.py --platform cuda --device a100

# Run only unit tests
python tests/run.py --platform cuda --device a100 --scope unit

# Run only functional tests (ops, compilation, distributed)
python tests/run.py --platform cuda --device a100 --scope functional

# Run only E2E tests (inference + serving)
python tests/run.py --platform cuda --device a100 --scope e2e

# Run a specific E2E test case
python tests/run.py --platform cuda --device a100 \
    --scope e2e --task inference --model qwen3 --case 06b_tp2

# Dry-run — show what would be executed
python tests/run.py --platform cuda --device a100 --dry-run

# Ascend NPU
python tests/run.py --platform ascend --device 910b --scope unit
```

### Via `pytest` directly (for development)

```bash
# Unit tests (no GPU required)
pytest tests/unit_tests/ -v

# Functional tests (requires GPU)
pytest tests/functional_tests/ -v -s

# Single E2E inference test (requires env vars)
FL_TEST_MODEL=qwen3 FL_TEST_CASE=06b_tp2 \
    pytest tests/e2e_tests/inference/test_inference_smoke.py -v -s

# Single E2E serving test
FL_TEST_MODEL=qwen3 FL_TEST_CASE=next_tp8 \
    pytest tests/e2e_tests/serving/test_serving_smoke.py -v -s
```

### Filter by markers

```bash
pytest -m gpu               # Only GPU tests
pytest -m "not slow"        # Skip slow tests
pytest -m e2e               # Only E2E tests
pytest -m multi_gpu         # Only multi-GPU tests
```

## Model YAML Config Format

E2E tests are driven by YAML configs under `tests/models/<model>/<case>.yaml`. Each config defines the LLM engine parameters and test behavior.

### Text model example

```yaml
# tests/models/qwen3/06b_tp2.yaml
llm:
  model: "/data/models/Qwen/Qwen3-0.6B"
  tensor_parallel_size: 2
  max_model_len: 8192
  gpu_memory_utilization: 0.7
  enforce_eager: true
  trust_remote_code: true

generate:
  prompts:
    - "Hello, my name is"
    - text: "The capital of France is"
      expected: "Paris"                    # Optional: substring validation
  sampling:
    max_tokens: 10
    temperature: 0.0
  parametrize:                             # Optional: Cartesian product of engine overrides
    enforce_eager: [true, false]

serve:                                     # Optional: serving test config
  endpoints: [completion]
  completion_prompt: "please introduce yourself"
  max_tokens: 20
```

### Multimodal model example

```yaml
# tests/models/minicpm/o45_tp4.yaml
llm:
  model: "/data/models/MiniCPM"
  tensor_parallel_size: 4
  max_model_len: 2048
  enforce_eager: true
  trust_remote_code: true

generate:
  modality: audio                          # text (default) | audio | image | video
  prompts:
    - question: "What is 1+1?"
      asset_count: 0
    - question: "What is recited in the audio?"
      asset_count: 1
  assets:                                  # vllm built-in asset names
    - mary_had_lamb
    - winning_call
  sampling:
    max_tokens: 128
    temperature: 0.2
```

### Config fields reference

| Field | Type | Required | Description |
|---|---|---|---|
| `llm.model` | str | Yes | Model path (auto-skips if not found) |
| `llm.*` | dict | Yes | Arguments passed to `LLM()` constructor |
| `generate.modality` | str | No | `text` (default), `audio`, `image`, `video` |
| `generate.prompts` | list | Yes | Strings or dicts with `text`/`expected` (text) or `question`/`asset_count` (multimodal) |
| `generate.assets` | list | No | vllm built-in asset names (required for multimodal) |
| `generate.sampling` | dict | Yes | Arguments passed to `SamplingParams()` |
| `generate.parametrize` | dict | No | Key=engine param, value=list of values (Cartesian product) |
| `serve.endpoints` | list | No | Endpoints to test: `completion`, `chat` |
| `serve.completion_prompt` | str | No | Prompt for `/v1/completions` |
| `serve.chat_messages` | list | No | Messages for `/v1/chat/completions` |
| `serve.max_tokens` | int | No | Max tokens for serving requests (default: 50) |
| `serve.api_key` | str | No | API key for authenticated endpoints |
| `serve.extra_engine` | dict | No | Engine param overrides for serving only |

## Platform Config Format

Platform configs (`tests/platforms/<platform>.yaml`) define which tests to run per device type, tolerance settings, and environment defaults.

```yaml
# tests/platforms/cuda.yaml
platform: cuda
vendor: nvidia

device_types:
  a100:
    compute_capability: "8.0"
    memory_gb: 80
    tags: [ampere, fp64, bf16]

tolerance:
  inference:
    default: {exact: true}

unsupported_features: []        # Strings to match against model names for skipping

env_defaults:
  CUDA_DEVICE_MAX_CONNECTIONS: "1"

a100:
  name: "a100"
  tests:
    e2e:
      inference:
        qwen3: ["4b_tp2", "06b_tp2", "next_tp8"]
        qwen3_5: ["35b_tp4"]
        minicpm: ["o45_tp4"]
      serving:
        qwen3: ["next_tp8"]
        minicpm: ["o45_tp4"]
    functional:
      include: "*"
      exclude: []
    unit:
      include: "*"
      exclude: []
```

## Writing New Tests

### Adding a new E2E model test

1. Create a YAML config: `tests/models/<model>/<case>.yaml`
2. Register it in the platform config: `tests/platforms/cuda.yaml` (and/or `ascend.yaml`)
3. Done — no Python code needed. The unified smoke tests handle the rest.

### Adding a unit test

Unit tests should be fast, isolated, and not require GPU or model weights. Use the `device` fixture for any tensor operations.

```python
# tests/unit_tests/ops/test_my_op.py
import pytest
import torch

class TestMyOp:
    def test_basic(self, device):
        """The `device` fixture returns cuda:0 or npu:0 automatically."""
        x = torch.randn(4, 8, device=device)
        # ... test logic ...
        assert result.shape == expected_shape
```

### Adding a functional test

Functional tests validate component correctness on real hardware. They require GPU but not model files.

```python
# tests/functional_tests/ops/test_my_kernel.py
import pytest
import torch

pytestmark = pytest.mark.gpu

def test_my_kernel(device):
    x = torch.randn(16, 256, device=device)
    result = my_kernel(x)
    reference = reference_impl(x)
    assert torch.allclose(result, reference, rtol=1e-3, atol=1e-3)
```

### Adding a new platform

1. Copy `tests/platforms/template.yaml` to `tests/platforms/<platform>.yaml`
2. Fill in device types, tolerance, env defaults, and test matrix
3. Add a cleanup function in `tests/utils/cleanup.py` if needed
4. Add backend detection in `tests/utils/device_utils.py`

## Test Scopes

| Scope | Directory | GPU | Models | Runs via |
|---|---|---|---|---|
| `unit` | `tests/unit_tests/` | No | No | `pytest` directly |
| `functional` | `tests/functional_tests/` | Yes | No | `pytest` directly |
| `e2e` | `tests/e2e_tests/` | Yes | Yes | Subprocess per case (via `run.py`) |

## Available Markers

| Marker | Description |
|---|---|
| `@pytest.mark.gpu` | Requires a single GPU/accelerator |
| `@pytest.mark.multi_gpu` | Requires multiple GPUs/accelerators |
| `@pytest.mark.slow` | Long-running test |
| `@pytest.mark.e2e` | End-to-end test |
| `@pytest.mark.functional` | Functional test |
| `@pytest.mark.flaggems` | Requires FlagGems library |
| `@pytest.mark.flagcx` | Requires FlagCX library |

## Shared Fixtures (from `tests/conftest.py`)

| Fixture | Scope | Description |
|---|---|---|
| `device` | session | `torch.device` for current backend (`cuda:0` / `npu:0`) |
| `backend` | session | Backend string (`"nvidia"` / `"ascend"` / `"cpu"`) |
| `has_accelerator` | session | Whether any GPU/NPU is available |
| `device_count` | session | Number of available accelerators |
| `cpu_device` | function | Always `torch.device("cpu")` |
| `mock_tensor` | function | `torch.randn(2, 4, 8)` on test device |
| `platform_config` | session | Loaded `PlatformConfig` (when `--platform` is set) |
| `tolerance` | session | Tolerance lookup helper from platform config |
| `model_config` | session | `ModelConfig.load` callable |
| `save_gold_mode` | session | Whether `--save-gold` was passed |

# Enflame S60 Deployment Guide

This guide starts an OpenAI-compatible vLLM service on Enflame S60 GCUs using
the dependency stack validated by this repository. Use the pinned image and
install the plugin from the revision you intend to deploy; the image does not
contain each candidate's plugin changes.

## Validated software stack

| Component | Validated version or configuration |
|---|---|
| Prebuilt image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-enflame-ci` |
| Image manifest | `sha256:f21e53101ffcf00fa5b70b39913bd401be93c0d1834c0ae8d07684bf0690bc88` |
| Accelerator / host driver | Enflame S60 / `1.10.6` |
| vLLM | `0.28.0+empty` (`VLLM_TARGET_DEVICE=empty`) |
| Python | `3.12.8` |
| PyTorch | `2.11.0+cpu` with `torch-gcu==2.11.0+3.8.20260713` |
| FlagTree | `0.7.0+enflame3.6`, the Enflame-specific release |
| Triton GCU | `3.6.0+1.0.20260722`; FlagTree provides the `triton` Python namespace |
| FlagGems | `5.4.0` |
| OpenAI SDK / pandas | `3.28.0` / `3.0.6` |

The `+cpu` PyTorch version is the vendor package's version string; execution
uses `torch_gcu` and the `gcu` device. Do not replace it with a generic PyTorch
wheel. Likewise, a generic FlagTree build or standalone upstream Triton wheel
is not interchangeable with this Enflame compiler stack.

Keep the plugin and its
[Enflame dispatch policy](../../vllm_fl/dispatch/config/enflame.yaml) on the same
revision. The policy selects the GCU attention backend and retains its tested
FlagGems exclusions. Build instructions and the rationale for the temporary
vendor adapters are in the [image recipe](../../docker/enflame/README.md).

## 1. Check the host and choose devices

Run Docker commands on the S60 host. It needs the compatible Enflame driver,
Docker, and tmux. Check accelerator memory and process ownership before choosing
two available devices for tensor parallelism:

```bash
efsmi
docker version
tmux -V
```

The checked-in CI configuration uses devices `2,3`; those IDs are an example,
not a guarantee that they are free. Do not stop another user's service to make
room. The examples below run one model and execution mode at a time on two GCUs.

Start a unique tmux session before cloning, pulling the image, or running setup
and validation. Run the following host steps inside that session so an SSH
disconnect does not interrupt them:

```bash
tmux new-session -s "enflame-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

## 2. Prepare the checkout and models

Use a private checkout at the branch, tag, or commit you intend to deploy.
Keep model weights on the host and mount them read-only. The native test YAMLs
expect the following two directories under `/data/models/Qwen` in the container:

```bash
git clone --branch 0.4.0-dev https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL

export REPO_DIR="$PWD"
export MODEL_ROOT=/data/models/Qwen
export TOPS_VISIBLE_DEVICES=2,3  # Replace with the available device pair.

test -f "$MODEL_ROOT/Qwen3.6-27B/config.json"
test -f "$MODEL_ROOT/Qwen3.6-35B-A3B/config.json"
git rev-parse HEAD
```

`MODEL_ROOT` must point to the actual host directory containing complete model
weights, tokenizer files, and configuration. The commands do not download
missing models. Record the selected plugin commit with the validation results.

## 3. Pull the image and create a private container

The digest pins the dependency image even if its tag changes. Authenticate with
`docker login harbor.baai.ac.cn` only if required by your registry access.

```bash
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-enflame-ci@sha256:f21e53101ffcf00fa5b70b39913bd401be93c0d1834c0ae8d07684bf0690bc88
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'

export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export CONTAINER="vllm-fl-enflame-$RUN_ID"
export LOG_DIR="$PWD/enflame-logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# Choose one of the four combinations in the table below.
export MODEL_NAME=Qwen3.6-27B
export MODE=eager
export PORT=8000

docker run --detach \
  --name "$CONTAINER" \
  --hostname vllm-plugin-fl \
  --privileged \
  --ipc=host \
  --ulimit nofile=65535:65535 \
  -p "$PORT:8000" \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v "$MODEL_ROOT:/data/models/Qwen:ro" \
  -v "$LOG_DIR:/logs" \
  -e VLLM_PLUGINS=fl \
  -e TOPS_VISIBLE_DEVICES="$TOPS_VISIBLE_DEVICES" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=99999 \
  -e TORCH_GCU_ENABLE_INT64_AND_UINT64=1 \
  -e ENABLE_I64_CHECK=0 \
  -e TORCHGCU_INDUCTOR_ENABLE=1 \
  -e MODEL_NAME="$MODEL_NAME" \
  -e MODE="$MODE" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc 'exec sleep infinity'
```

The privileged container exposes the vendor devices, matching the existing S60
CI configuration. Keep Docker's private PID namespace: do not add `--pid=host`,
because the test runner performs process cleanup. The model mount uses the
container paths from the native YAMLs regardless of the host's storage location.

| Model | `MODEL_NAME` | `MODE` |
|---|---|---|
| Qwen3.6-27B eager | `Qwen3.6-27B` | `eager` |
| Qwen3.6-27B graph | `Qwen3.6-27B` | `graph` |
| Qwen3.6-35B-A3B eager | `Qwen3.6-35B-A3B` | `eager` |
| Qwen3.6-35B-A3B graph | `Qwen3.6-35B-A3B` | `graph` |

## 4. Verify and install the selected plugin

Continue in the tmux session created above and enter the private container:

```bash
docker exec -it "$CONTAINER" bash
```

Inside the container:

```bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
bash .github/scripts/enflame/check.sh 2>&1 | tee /logs/check.log
bash .github/scripts/enflame/setup.sh 2>&1 | tee /logs/setup.log
```

`check.sh` requires at least two visible GCUs and creates a small GCU tensor.
`setup.sh` verifies package versions and runs both checkout guards:

- [`verify_compiler.py`](../../docker/enflame/verify_compiler.py) checks the
  imported Triton origin and FlagTree's recorded release-file hashes.
- [`verify_index_fill.py`](../../docker/enflame/verify_index_fill.py) checks the
  supported FlagGems release bytes used by the GCU `index_fill` adapter.

It then installs this checkout with
`python -m pip install --no-build-isolation --no-deps -e .` and checks imports.
Do not resolve dependencies again with a generic `pip install -U`: a dependency
can overwrite FlagTree's Triton files while leaving its package metadata intact.
The guards detect that corruption; metadata versions alone do not prove that
the intended compiler is active.

## 5. Start and query the service

Continue inside the container. These examples use the native TP2 model settings
and the S60 overrides: `max_model_len=8192`, memory utilization `0.94`, and the
GCU collective path. Graph mode omits `--enforce-eager`.

```bash
case "$MODEL_NAME" in
  Qwen3.6-27B|Qwen3.6-35B-A3B) ;;
  *) echo "Unsupported example MODEL_NAME: $MODEL_NAME" >&2; exit 2 ;;
esac
mode_args=()
case "$MODE" in
  eager) mode_args+=(--enforce-eager) ;;
  graph) ;;
  *) echo "MODE must be eager or graph" >&2; exit 2 ;;
esac

vllm serve "/data/models/Qwen/$MODEL_NAME" \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.94 \
  "${mode_args[@]}" 2>&1 | tee "/logs/$MODEL_NAME-$MODE-server.log"
```

Cold-cache compilation and graph capture can take substantial time. The native
S60 e2e jobs allow 120 minutes. Follow the service log and wait for startup to
finish before sending requests. In a second host terminal, use the host port
selected above (8000 in this example):

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

Check the returned model name and response, and retain server errors if any.
An HTTP response is a deployment smoke check; it does not establish full model
or adaptation Gate coverage. Stop this example server with Ctrl-C in its tmux
session before using the same devices for another model, mode, or test.

## 6. Run the repository validation

With the example server stopped, run these commands inside the private
container and retain the generated reports and logs:

```bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
python tests/run.py --platform enflame --device s60 --scope unit \
  --output-dir /logs/unit 2>&1 | tee /logs/unit-console.log
python tests/run.py --platform enflame --device s60 --scope functional \
  --output-dir /logs/functional 2>&1 | tee /logs/functional-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task inference \
  --model qwen3_6 --case 27b_tp2_eager --output-dir /logs/27b-eager \
  2>&1 | tee /logs/27b-eager-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task inference \
  --model qwen3_6 --case 27b_tp2_graph --output-dir /logs/27b-graph \
  2>&1 | tee /logs/27b-graph-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task serving \
  --model qwen3_6 --case 35b_a3b_tp2_eager --output-dir /logs/35b-eager \
  2>&1 | tee /logs/35b-eager-console.log
python tests/run.py --platform enflame --device s60 --scope e2e --task serving \
  --model qwen3_6 --case 35b_a3b_tp2_graph --output-dir /logs/35b-graph \
  2>&1 | tee /logs/35b-graph-console.log
```

Each invocation writes to a separate output directory so its JSON/XML reports
are preserved instead of overwritten by the next run.

These are the cases selected in
[`tests/platforms/enflame.yaml`](../../tests/platforms/enflame.yaml), with the
original [Qwen3.6 model YAMLs](../../tests/models/qwen3_6). The 27B inference
cases check text completion; the 35B serving cases check model listing and
text/image chat. They do not replace the longer concurrent scenarios in the
[adaptation Gate](../../tools/adaptation-gate-cases/README.md).

Run each Gate model in both eager and graph modes, preserving the original
requests and checks. Record the actual plugin commit, model, mode, port, and
server arguments alongside the raw scenario JSONs. Acceptance requires every
`summary.passed` and every request's checks to pass, rather than only a pytest
exit code. The S60 capacity validation for 35B graph used TP2, max length 32768,
max sequences 256, memory utilization 0.94, and FP32 Mamba state; the 8192-token
CI smoke settings above do not reproduce that capacity test. Retain FP32 Mamba
state and the original quality checks when repeating it.

The pinned dependency image has hardware validation history, but it does not
establish acceptance of an arbitrary plugin checkout. Re-run the relevant tests
and original Gate for the exact revision you deploy.

## Limitations and troubleshooting

- **Device unavailable:** check `efsmi`, selected device ownership, the host
  driver, container device access, and `TOPS_VISIBLE_DEVICES`. Package import
  success alone does not establish GCU availability.
- **Compiler guard fails:** retain its output and recreate the container from
  the pinned image. Do not bypass the guard or install a generic Triton wheel.
- **Out of memory:** first check that the selected pair is available and that
  the S60 `0.94` budget is applied. Changing context length, concurrency,
  precision, or tensor parallelism requires separate validation of that new
  configuration; it is not a passing result for the original Gate.
- **Runner or speculative-decoding error:** Enflame uses `ModelRunnerFL` for
  eager and graph execution. Explicit `VLLM_USE_V2_MODEL_RUNNER=1` and V2-only
  features are unsupported. Hybrid models with speculative decoding and
  `mamba_cache_mode=align` are rejected because the fused Mamba state-copy path
  does not yet translate its indirect pointers to GCU addresses. Other
  speculative/cache combinations still need their own hardware validation.
- **Other dependency versions or optional C++ operators:** the compatibility
  adapters target the fixed stack above. The Python FlagGems wheel does not
  establish coverage of the optional native C++ extension; see the
  [image recipe](../../docker/enflame/README.md) before changing that stack.

After saving results and confirming that this private container has no active
tests or service, stop it from the host with `docker stop "$CONTAINER"`. Keep
its logs and result files; do not stop a shared container or another task.

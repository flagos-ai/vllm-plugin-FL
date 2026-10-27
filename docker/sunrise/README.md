# Sunrise deployment on vLLM 0.28.0

This guide covers the Sunrise-S2 / Persistence-M backend (`torch.ptpu`) on
`0.4.0-dev`. Use the vendor PyTorch / `torch_ptpu`, Triton, Tang runtime and
PCCL packages as a matched stack. Installing an ordinary CUDA PyTorch or CUDA
vLLM wheel into this environment does not provide Sunrise support.

## Prepare the environment

The earlier vLLM 0.24.0 baseline used Ubuntu 22.04, Python 3.10,
PyTorch `2.11.0+cpu`, FlagGems `5.3.4`, vendor Triton `3.6.0`,
FlagTree `0.6.0+sunrise3.6`, and FlagCX `0.13.0`. Its image was:

```text
harbor.baai.ac.cn/plugin/sunrise001-gems5.3.4-treenone-triton3.6.0-cx0.13.0-plugin0.3.0-vllm0.24.0-cp310-pt211-x64-v0.24.0:202609281010
```

That image is a **0.24.0 baseline**, not a prebuilt 0.28.0 environment.
Use a separate container/environment with a Sunrise-compatible vLLM 0.28.0
wheel or source build. Keep the vendor CPU PyTorch and `torch_ptpu` packages;
installing vLLM with dependencies may replace that runtime. No Sunrise 0.28.0
image or wheel URL is prescribed here.

For a vendor-built wheel, provide its actual local path:

```bash
python -m pip install --no-deps "${SUNRISE_VLLM_WHEEL:?set the path to the Sunrise-compatible vLLM 0.28.0 wheel}"
```

Alternatively, build the `empty` target from vLLM `v0.28.0` inside the matched
vendor build environment, with its build dependencies already installed:

```bash
git clone --branch v0.28.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
tmux new-session -d -s sunrise-vllm-build \
  'VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation --no-deps . > /tmp/sunrise-vllm-build.log 2>&1'
# Check /tmp/sunrise-vllm-build.log and the installed version before proceeding.
```

Mount the host's `/usr/local/tangrt`, `/usr/local/pccl`, and model directory
into the container. Use the vendor's container/device access configuration.
Install this branch of the plugin without resolving dependencies:

```bash
cd /path/to/vllm-plugin-FL
python -m pip install --no-build-isolation --no-deps -e .
```

The baseline image defaults to FlagTree. Select the **vendor Triton** compiler
before starting any inference process; the baseline report observed device
failures with the default compiler:

```bash
compiler triton
export GEMS_VENDOR=sunrise
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
export VLLM_USE_V2_MODEL_RUNNER=0
export TA_CLOSE_GRAPH_MEMORY=1
export PCCL_PROTO=Simple
export TANG_VISIBLE_DEVICES=0,1
export FLAGCX_PATH=/workspace/FlagCX
unset VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS
```

Set `TANG_VISIBLE_DEVICES` to two available cards, and `FLAGCX_PATH` to the
actual matched FlagCX checkout. The zero-output GDN benchmark workaround from
the earlier report is unsuitable for correctness validation and must remain
disabled. Do not enable it to make graph execution appear to pass.

Verify the selected runtime before loading a model:

```bash
python - <<'PY'
import torch
import torch_ptpu  # registers torch.ptpu
import triton
import vllm
import vllm_fl
from vllm.platforms import current_platform

print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("vllm:", vllm.__version__)
print("plugin:", vllm_fl.__file__)
print("device:", current_platform.device_type)
assert vllm.__version__.split("+")[0] == "0.28.0"
assert current_platform.device_type == "ptpu"
assert torch.ptpu.is_available()
assert torch.ptpu.device_count() >= 2
PY
```

## KV-cache sizing on older PTPU runtimes

Some older PTPU runtimes report the **current whole-device memory usage** from
both `memory_allocated()` and `max_memory_allocated()`. These values are not a
process allocation counter or a recorded allocation peak. Synthesizing the
`memory_stats()` keys needed by the worker does not make the underlying values
accurate process statistics, so automatic KV-cache sizing is limited on these
runtimes.

For production acceptance, set a conservative per-worker/per-device
`--kv-cache-memory-bytes` explicitly, after accounting for model weights,
activations and runtime workspaces on the selected cards. The examples below
start with 4 GiB (`4294967296` bytes) per worker; adjust this budget and the
context/concurrency settings for the actual model and available memory. An
explicit KV-cache budget overrides automatic KV-cache sizing based on
`--gpu-memory-utilization`; it does not constrain all other allocations.

## Serve and validate both models

Use `Qwen3.6-27B` and `Qwen3.6-35B-A3B`, with the original model configs,
tokenizer and checkpoint shards. Run every model once in eager mode and once
in `FULL_DECODE_ONLY` graph mode. Execute the following from the repository
root inside the prepared container:

```bash
export MODEL_PATH=/data/models/Qwen/Qwen3.6-27B
export PORT=18080
export KV_CACHE_MEMORY_BYTES=4294967296
export GATE_DIR="$PWD/tools/adaptation-gate-cases"
export RUN_DIR=/tmp/sunrise-028-qwen36-27b
mkdir -p "$RUN_DIR"

tmux new-session -d -s sunrise-028-eager \
  "vllm serve '$MODEL_PATH' --host 127.0.0.1 --port '$PORT' \
    --tensor-parallel-size 2 --disable-custom-all-reduce \
    --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 --skip-mm-profiling \
    --kv-cache-memory-bytes '$KV_CACHE_MEMORY_BYTES' \
    --allowed-local-media-path '$GATE_DIR/images' --enforce-eager \
    > '$RUN_DIR/eager.log' 2>&1"

tmux new-session -d -s sunrise-028-eager-gate \
  "cd '$GATE_DIR' && MODEL_PATH='$MODEL_PATH' PORT='$PORT' \
    RESULTS_DIR='$RUN_DIR/eager-results' ./run_test.sh \
    > '$RUN_DIR/eager-gate.log' 2>&1; echo \$? > '$RUN_DIR/eager-gate.status'"
```

Inspect the server log, gate log, JSON results and gate exit status. After the
gate finishes, stop the eager service (`tmux send-keys -t sunrise-028-eager C-c`)
and wait for its worker processes to exit before reusing the cards. Confirm
that no unrelated workload uses the selected devices.

Start graph mode on a different port so results remain separate:

```bash
export PORT=18081

tmux new-session -d -s sunrise-028-graph \
  "vllm serve '$MODEL_PATH' --host 127.0.0.1 --port '$PORT' \
    --tensor-parallel-size 2 --disable-custom-all-reduce \
    --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 --skip-mm-profiling \
    --kv-cache-memory-bytes '$KV_CACHE_MEMORY_BYTES' \
    --allowed-local-media-path '$GATE_DIR/images' \
    --compilation-config '{\"mode\": 0, \"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' \
    > '$RUN_DIR/graph.log' 2>&1"

tmux new-session -d -s sunrise-028-graph-gate \
  "cd '$GATE_DIR' && MODEL_PATH='$MODEL_PATH' PORT='$PORT' \
    RESULTS_DIR='$RUN_DIR/graph-results' ./run_test.sh \
    > '$RUN_DIR/graph-gate.log' 2>&1; echo \$? > '$RUN_DIR/graph-gate.status'"
```

Repeat both modes with `MODEL_PATH=/data/models/Qwen/Qwen3.6-35B-A3B` and a
separate `RUN_DIR`. Stop each service and remove its completed tmux sessions
before reusing the session names. The current Gate prompts fit within this 8192-token context setting; eight
sequences cover its full eight-request concurrency. Keep all shared cases and
quality checks unchanged when adjusting engine capacity for the actual model.

The [adaptation gate](../../tools/adaptation-gate-cases/README.md) checks text,
images and concurrent mixed requests with semantic/output-quality assertions.
Keep the logs and per-request JSON reports for all four runs. A successful
server start or HTTP 200 response alone does not establish correctness; graph
validation also requires the log to show successful graph capture/replay.
These commands are an acceptance procedure, not a claim that this hardware
matrix has already passed on vLLM 0.28.0.

## CI and the standalone shared Gate

The dedicated workflow is `.github/workflows/_sunrise_test.yml`. It runs the
Sunrise unit contracts with zero skips, followed by the real
`tools/adaptation-gate-cases/run_test.sh` for both models in both modes. Every
run must produce all five scenario reports and 26 passing requests; all four
runs therefore contain 20 scenarios and 104 requests. Graph acceptance requires
successful capture plus actual non-`NONE` replay markers recorded **after** the
owned service becomes ready, during the Gate requests. The receipt also records
the actual logged KV-cache token capacity, which must cover the requested
context. Initialization/warmup replays do not satisfy the request-time check.

The published FlagGems backend registry identifies the real runner label
`sunrise`; see [the backend registry](https://github.com/flagos-ai/FlagGems/blob/master/.github/backends.json).
`SUNRISE_CI_RUNNER_LABELS` can override it with a JSON array of administrator
confirmed labels. A runner service named `Sunrise8` does not imply a label of
that name. The Sunrise entry in `.github/configs/platforms.yml` remains disabled
until a usable 0.28 stack, reserved healthy devices and the four Gate runs have
been accepted; it must not block unrelated PRs on unprovisioned resources.
The dedicated workflow also has a manual dispatch entry.

Configure the actual resources through repository/organization variables:

| Variable | Meaning |
|---|---|
| `SUNRISE_CI_IMAGE` | Actual prepared vendor 0.28 image; the versioned default is the older 0.24 vendor base |
| `SUNRISE_CI_VLLM_WHEEL` / `SUNRISE_CI_VLLM_SOURCE` | One actual absolute mounted path for a 0.28 empty wheel/source build when using the older base |
| `SUNRISE_CI_RUNNER_LABELS` | JSON label array; default `["sunrise"]` |
| `SUNRISE_CI_CONTAINER_VOLUMES` | JSON mounts for runtime libraries, models and any wheel/source override |
| `SUNRISE_CI_VISIBLE_DEVICES` | Two distinct reserved card indices; default `6,7` |
| `SUNRISE_CI_MODEL_27B` / `SUNRISE_CI_MODEL_35B` | Actual absolute model directories inside the container |
| `SUNRISE_CI_FLAGCX_PATH` | Actual matched FlagCX checkout inside the container |
| `SUNRISE_CI_MAX_MODEL_LEN` / `SUNRISE_CI_MAX_NUM_SEQS` | Engine capacity, default `8192` / `8`; shared cases are unchanged |
| `SUNRISE_CI_KV_CACHE_MEMORY_BYTES` | Explicit per-worker budget, default `4294967296` |

Setup builds a separate environment, selects vendor Triton through
`/etc/profile.d/zz-compiler.sh`, persists the resulting `PYTHONPATH`, installs
the plugin without dependency resolution and uses the shared installer for
FlagGems-vllm `v0.2.0` commit `b5c2c2249319277ccda4d41aa0f65164bd9b2a19`.
The subsequent check requires actual vLLM 0.28.0, Sunrise/PTPU registration,
two visible devices and the pinned provider identity. The prepared build/image
environment must already supply compatible Python/runtime dependencies; setup
does not resolve a replacement PyTorch or Triton stack.

The same Gate runner works independently of GitHub Actions in the configured
Sunrise environment:

```bash
python .github/scripts/sunrise/run_gate.py \
  --model /data/models/Qwen/Qwen3.6-27B --mode eager \
  --max-model-len 8192 --max-num-seqs 8 --kv-cache-memory-bytes 4294967296 \
  --output-dir /tmp/sunrise-028-27b-eager
```

Run this long command in a tmux session for remote validation. Repeat with
`--mode graph`, then with `Qwen3.6-35B-A3B`, using a fresh output directory each
time. The runner owns its server session, waits for that exact model endpoint,
runs the unchanged shared Gate and cleans up its own processes before returning.
Keep `summary.json`, `server.log`, `gate.log` and all request JSON files. A failed
cleanup stops the serial CI matrix before another service can use the pair.

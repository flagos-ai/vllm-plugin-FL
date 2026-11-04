# MUSA deployment

vLLM-Plugin-FL supports vLLM **0.28.0** on Moore Threads MUSA with the V1 model
runner, native MCCL communication and piecewise MUSA graph capture.

## Runtime requirements

Use Linux with the MUSA driver and container runtime installed. The supported
runtime baseline is:

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch / TorchMUSA | 2.9.0 |
| MUSA SDK | 4.3.5 |
| vLLM | 0.28.0 `empty` build |
| FlagGems | `5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix2` |
| FlagTree | `0.7.0+mthreads3.6.git69148faf.profilefix1` |

The default [CI configuration](../../.github/configs/musa.yml) pins this image:

```text
harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-musa-ci@sha256:1e3cdbd8b24d65828cf52166fce2f6b16183cbe13a3e95f543a34781f4507998
```

The image provides the vendor runtime, an ordinary empty vLLM wheel and the
MUSA dependency wheels. Preserve its Torch, TorchMUSA and SDK packages when
installing the plugin. Generic PyPI Torch, CUDA wheels or an unpatched FlagTree
build are not substitutes for this MUSA stack.

## Required dependencies and issue handling

**Use the pinned image above for this guide. Its corrected FlagGems and FlagTree
wheels are required parts of this tested runtime; no additional dependency source
edits, cherry-picks or issue reproducer commands are needed.** The issue links
below explain what is already included, rather than extra installation steps.

| Issue | Treatment in the pinned image | Action when following this guide |
|---|---|---|
| [FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973) | Device-selection corrections for pinning and shared attention. | Keep the supplied FlagGems wheel; no manual patch. |
| [FlagGems #7005](https://github.com/flagos-ai/FlagGems/issues/7005) | Propagate recognized fatal MUSA illegal-memory errors instead of treating them as recoverable tuning failures. | No extra setting or patch. If a native error occurs, stop that process; do not keep using its device context. |
| [FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344) | The optional-profiler OFF/OFF header guard is included in the prebuilt FlagTree wheel. | Relevant when building FlagTree from source, not a step needed to use this image. |
| [FlagTree #1346](https://github.com/flagos-ai/FlagTree/issues/1346) | The supplied FlagGems MUSA multidimensional multiplication path uses the tested BLOCK512 workaround. | Keep the supplied wheels. This is a workaround, not a confirmed compiler repair. |

Installing the current plugin does not apply these patches to another environment.
An unpatched upstream install or another dependency version needs separate
validation; an issue being closed does not make it interchangeable with these
wheels. This guide does not require rebuilding TorchMUSA, FlagTree or FlagGems.

## Prepare the host and model

The commands below run a single-card Qwen3-0.6B server. Host prerequisites are a
working MUSA driver/container runtime, Docker, Git, curl, registry pull access and
network access to GitHub. Downloading the example model also requires access to
Hugging Face. All paths supplied to Docker must be absolute.

Run on the **host**, choosing an idle card assigned to your job and a local model
directory:

```bash
set -euo pipefail
IMAGE=harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-musa-ci@sha256:1e3cdbd8b24d65828cf52166fce2f6b16183cbe13a3e95f543a34781f4507998
MODEL_DIR="$HOME/models/Qwen3-0.6B"
SOURCE_DIR="$PWD/vllm-plugin-FL"
PHYSICAL_DEVICE=0  # Replace with the host card allocated to your job.

mthreads-gmi
docker pull "$IMAGE"
git clone --depth 1 https://github.com/flagos-ai/vllm-plugin-FL.git "$SOURCE_DIR"
# While this adaptation is under review, select its public PR head instead of main.
git -C "$SOURCE_DIR" fetch --depth 1 origin pull/575/head
git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
mkdir -p "$MODEL_DIR"
```

If the model is not already present, download it once with a CPU-only container.
This writes model assets, without changing the runtime image:

```bash
docker run --rm --runtime=runc \
  --env MTHREADS_VISIBLE_DEVICES= --env MUSA_VISIBLE_DEVICES= \
  --mount "type=bind,src=$MODEL_DIR,dst=/model" \
  --entrypoint /opt/musa/venv/bin/python "$IMAGE" -I -B -c \
  'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca", local_dir="/model")'
```

An existing model directory must contain its complete config, tokenizer and
weights; pointing to a cache parent or an empty directory is insufficient. The
following container mounts the model read-only and publishes the API only on the
host's loopback interface:

```bash
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors"
docker run --rm -it \
  --runtime=mthreads --ipc=private --shm-size=64g \
  --env GEMS_VENDOR=mthreads --env VLLM_PLUGINS=fl \
  --env "MTHREADS_VISIBLE_DEVICES=$PHYSICAL_DEVICE" \
  --env MUSA_VISIBLE_DEVICES=0 --env MODEL_DIR=/model \
  --mount "type=bind,src=$SOURCE_DIR,dst=/workspace/vllm-plugin-FL" \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  --workdir /workspace/vllm-plugin-FL \
  --publish 127.0.0.1:8000:8000 \
  --entrypoint bash "$IMAGE"
```

Set the physical allocation when creating the container; changing it in a running
container does not change which host devices are passed through. The environment
variables serve different layers:

| Variable | Purpose |
|---|---|
| `GEMS_VENDOR=mthreads` | Select the Mthreads backend in FlagGems. |
| `VLLM_PLUGINS=fl` | Enable the FL platform/model plugin in vLLM. |
| `MTHREADS_VISIBLE_DEVICES` | Give the Mthreads container runtime the allocated **host physical** card IDs. Never replace a job's allocation with `all`. |
| `MUSA_VISIBLE_DEVICES` | Limit the MUSA SDK/TorchMUSA logical device IDs inside the container. The single exposed card in this example is logical `0`, even when its host ID is different. |

The CI runner has a separate fixed four-card allocation: host cards `4,5,6,7`
become logical cards `0,1,2,3`. These numbers are runner-specific, not a requirement
to use the plugin. For a four-card deployment, supply your assigned four host IDs,
set the logical mask to `0,1,2,3`, and use `--tensor-parallel-size 4` with a model
that supports that parallelism. `USE_FLAGGEMS=1` is unnecessary here because the
plugin enables FlagGems by default. `FLAGCX_PATH` is not required for native MCCL.

## Install the current plugin

Run **inside the container**, from `/workspace/vllm-plugin-FL`. Use the image's
interpreter for both installation and serving. Install an ordinary wheel so fresh
serving workers use this checkout rather than the image's older plugin:

```bash
set -euo pipefail
PYTHON=/opt/musa/venv/bin/python
unset SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL
WHEEL_DIR=$(mktemp -d)
env -u VLLM_VENDOR "$PYTHON" -I -B -m pip wheel \
  --no-deps --no-build-isolation --wheel-dir "$WHEEL_DIR" .
"$PYTHON" -I -B -m pip install --no-deps --force-reinstall \
  "$WHEEL_DIR"/vllm_plugin_fl-*.whl
"$PYTHON" -I -B .github/scripts/install_flaggems_vllm.py
```

The last command is **required**: it installs the shared FlagGems-vllm provider
from the repository's pinned upstream commit
`f775b37f3066e2935742d26c7592a5dd55499159`, unless that exact source is already
installed. It checks that the existing runtime package versions remain unchanged.
This provider is separate from the corrected FlagGems wheel already in the image.
Do not install generic PyPI Torch/CUDA packages or replace the image's dependencies.

Check the runtime before loading the model:

```bash
"$PYTHON" -I -B - <<'PY'
import sys
from pathlib import Path
from importlib.metadata import version
import torch
import torch_musa
import vllm_fl
from vllm.platforms import current_platform

expected = {
    "vllm": "0.28.0+empty",
    "flag-gems": "5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix2",
    "flagtree": "0.7.0+mthreads3.6.git69148faf.profilefix1",
}
for package, wanted in expected.items():
    assert version(package) == wanted, (package, version(package))
assert torch.musa.is_available(), "Check the host driver and allocated devices"
assert torch.musa.device_count() == 1, "This example expects one exposed card"
assert current_platform.device_type == "musa", current_platform
assert Path(vllm_fl.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print("MUSA runtime ready:", vllm_fl.__file__)
PY
```

The [CI setup script](../../.github/scripts/musa/setup.sh) additionally installs
test-reporting plugins. Running that script is not required for this deployment.

## Start the server and send a request

In the same **container shell**, run:

```bash
"$PYTHON" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name musa-smoke \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 2048 \
  --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete` in the server log. The first start can
take longer while kernels compile and graphs capture. In a second **host shell**,
check readiness and request a completion:

```bash
curl --fail --show-error http://127.0.0.1:8000/health
curl --fail --show-error http://127.0.0.1:8000/v1/chat/completions \
  --header 'Content-Type: application/json' \
  --data '{"model":"musa-smoke","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

Success means `/health` returns HTTP 200 and the completion JSON contains a
non-empty `choices[0].message.content`. Stop the server with Ctrl-C, then `exit`
the container shell; `--rm` removes this container. If port 8000 is occupied,
change the host side of `--publish` and the two curl URLs together.

MUSA selects the V1 runner and MCCL communication. The default compiler adapter
is FX/eager: Dynamo tracing, piecewise splitting and native graph replay remain
enabled. This setting is different from `--enforce-eager`, which disables graph
execution. Explicit Inductor selection remains experimental for this stack.

Compiled inference with gradients disabled uses MUSA-specific RMSNorm, SiLU and
rotary operator boundaries. Their fake metadata and mutation schemas keep
Python device helpers outside the traced graph. Eager, gradient-enabled and
other-platform paths retain their own operator dispatch.

## Graph behavior and limitations

MUSA graph handling lives in
[`dispatch/backends/vendor/musa/graph.py`](../../vllm_fl/dispatch/backends/vendor/musa/graph.py).
Capture and replay use `torch.musa.MUSAGraph` directly. Captured inputs stay owned
by the graph; output views are reconstructed from storage and tensor metadata.
Stable captured buffers replay. Calls with replacement input allocations execute
the original callable so mutations and output aliases remain visible to the
caller.

All configured capture sizes and their order are retained. A process-wide budget
admits at most 1900 live native graphs, including reservations for pending
captures. Additional entries execute the original callable until their wrapper
entries are cleared. The budget is a conservative resource policy; it does not
measure the SDK's exact capacity or guarantee native capture of every size.

A weak-reference set tracks graph lifetime. Clearing a wrapper releases its
entries; graphs held by external references continue to count against the budget
until released. Capture, replay and eager execution failures propagate and poison
the budget, stopping subsequent graph work. This path requires no extra C++ shim,
Ninja build or executable rotation. A different TorchMUSA version needs separate
validation.

The live-graph budget and eager fallback define resource behavior; they do not
establish a GPU latency or throughput guarantee. Representative hardware
bucket-switching latency and broad ABI compatibility remain unmeasured.

## CI coverage

MUSA uses the repository's
[shared platform workflow](../../.github/workflows/_platform_test.yml) for a
combined unit/functional job, E2E inference and serving, and benchmark smoke. The
[platform test configuration](../../tests/platforms/musa.yaml) currently lists
six inference and five serving configurations for PR validation, retaining the
0.6B smoke cases alongside all four 27B/35B modes. Nightly also includes
concurrent cases; MUSA E2E cases run serially to avoid overlapping
MCCL group initialization.

Use the repository's `ci.yml` platform selection. Runner labels and device
environment settings come from the
[CI configuration](../../.github/configs/musa.yml); model cases and test settings
come from the platform test configuration. Keep the original model checks,
capture sizes, memory budgets and timeouts when validating changes. Consult the
actual CI run for its source, platform results and conditional skips.

Installation checks or CPU tests alone do not establish device-kernel,
multimodal-model, native ABI or performance acceptance.

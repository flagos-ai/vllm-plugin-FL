# MUSA deployment

vLLM-Plugin-FL supports vLLM **0.32.0** on Moore Threads MUSA with the V1 model
runner, native MCCL communication and piecewise MUSA graph capture.

## Runtime requirements

Use Linux with the MUSA driver and container runtime installed. The supported
runtime baseline is:

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch / TorchMUSA | 2.9.0 (`torch.version.musa` = `40306`) / 2.9.0 |
| MUSA SDK | 4.3.6 |
| vLLM | 0.32.0 `empty` build |
| FlagGems | `5.4.0` |
| FlagTree | `0.7.0+mthreads3.6` (also provides Triton 3.6.0) |

The [CI configuration](../../.github/configs/musa.yml) pins the adapted image:

```text
harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-musa-ci
```

The image is published to the `plugin` project. It contains the vendor runtime,
the vLLM 0.32.0 empty wheel and the test tooling; the plugin itself and
`flaggems-vllm` are installed from the checkout at job time by
[`.github/scripts/musa/setup.sh`](../../.github/scripts/musa/setup.sh). The
0.32.0 adaptation was additionally validated on the FlagOS 2.2 runtime image
`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa4.3.6:2.2.0`
with the plugin installed editable (see below).

The image provides the vendor runtime and an ordinary empty vLLM wheel. Preserve
its Torch, TorchMUSA and SDK packages when installing the plugin. Generic PyPI
Torch, CUDA wheels or a standalone Triton are not substitutes for this MUSA stack.

## Required dependencies and issue handling

**Use the image above that ships the pinned tuple; the corrected FlagGems and
FlagTree wheels are required parts of this tested runtime.** The issue links
below were recorded while adapting the 0.28.0 stack and are kept for lineage;
they were not re-verified individually on 0.32.0.

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
# Validated runtime. The CI image adds the same vLLM 0.32.0 empty wheel:
#   harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-musa-ci
# Installing the plugin editable from the checkout below works on either image.
IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa4.3.6:2.2.0
MODEL_DIR="$HOME/models/Qwen3-0.6B"
SOURCE_DIR="$PWD/vllm-plugin-FL"
PHYSICAL_DEVICE=0  # Replace with the host card allocated to your job.

mthreads-gmi
docker pull "$IMAGE"
# This adaptation targets the 0.5.0-dev branch (vLLM 0.32.0).
git clone --depth 1 --branch 0.5.0-dev https://github.com/flagos-ai/vllm-plugin-FL.git "$SOURCE_DIR"
mkdir -p "$MODEL_DIR"
```

If the model is not already present, download it once with a CPU-only container.
This writes model assets, without changing the runtime image:

```bash
docker run --rm --runtime=runc \
  --env MTHREADS_VISIBLE_DEVICES= --env MUSA_VISIBLE_DEVICES= \
  --mount "type=bind,src=$MODEL_DIR,dst=/model" \
  --entrypoint /flagos/bin/python "$IMAGE" -I -B -c \
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
| `VLLM_USE_V2_MODEL_RUNNER=0` | Required: the plugin implements the V1 `ModelRunnerFL`; the upstream V2 runner uses CUDA/UVA APIs MUSA does not provide. The plugin sets this default itself. |
| `USE_FLAGTUNE` | The plugin sets `USE_FLAGTUNE=0` on MUSA so FlagGems does not fetch the FlagTune cost-model bundle; set `VLLM_FL_MUSA_FLAGTUNE=1` to opt back in. |

The CI runner has a separate fixed four-card allocation: host cards `4,5,6,7`
become logical cards `0,1,2,3`. These numbers are runner-specific, not a requirement
to use the plugin. For a four-card deployment, supply your assigned four host IDs,
set the logical mask to `0,1,2,3`, and use `--tensor-parallel-size 4` with a model
that supports that parallelism. `USE_FLAGGEMS=1` is unnecessary here because the
plugin enables FlagGems by default. `FLAGCX_PATH` is not required for native MCCL.

## Install the current plugin

Run **inside the container**, from the plugin checkout. Use the image's
interpreter for both installation and serving. Install the checkout editable so
fresh serving workers use this source tree:

```bash
set -euo pipefail
PYTHON=/flagos/bin/python
unset SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL
env -u VLLM_VENDOR "$PYTHON" -m pip install \
  --no-deps --no-build-isolation -e .
"$PYTHON" .github/scripts/install_flaggems_vllm.py
```

The last command is **required**: it installs the shared FlagGems-vllm provider
from the repository's pinned upstream commit
`f775b37f3066e2935742d26c7592a5dd55499159`, unless that exact source is already
installed. It checks that the existing runtime package versions remain unchanged.
This provider is separate from the corrected FlagGems wheel already in the image.
Do not install generic PyPI Torch/CUDA packages or replace the image's dependencies.

Check the runtime before loading the model:

```bash
"$PYTHON" - <<'PY'
from importlib.metadata import version
import torch
import torch_musa
import triton
import vllm_fl
from vllm.platforms import current_platform

expected = {
    "vllm": "0.32.0",
    "flag-gems": "5.4.0",
    "flagtree": "0.7.0+mthreads3.6",
}
for package, wanted in expected.items():
    assert version(package).startswith(wanted), (package, version(package))
assert torch.__version__.startswith("2.9.0"), torch.__version__
assert triton.__version__.startswith("3.6."), triton.__version__
assert torch.musa.is_available(), "Check the host driver and allocated devices"
assert torch.musa.device_count() == 1, "This example expects one exposed card"
assert current_platform.device_type == "musa", current_platform
print("MUSA runtime ready:", vllm_fl.__file__)
PY
```

The [CI setup script](../../.github/scripts/musa/setup.sh) additionally installs
test-reporting plugins. Running that script is not required for this deployment.

## Start the server and send a request

In the same **container shell**, run:

```bash
"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name musa-smoke \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 2048 \
  --host 0.0.0.0 --port 8000
```

For hybrid (GDN/Mamba) models keep `--mamba-cache-mode none` (the vLLM 0.32
default); `align` is not supported by the plugin runner on MUSA.

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

MUSA graph capture runs in PIECEWISE mode. torch_musa does not support
`musaStreamCaptureModeThreadLocal`, so FULL cudagraphs are rejected; requesting
`FULL` falls back to `PIECEWISE`. Capture and replay use `torch.musa.MUSAGraph`
directly. Captured inputs stay owned by the graph; output views are reconstructed
from storage and tensor metadata. Stable captured buffers replay. Calls with
replacement input allocations execute the original callable so mutations and
output aliases remain visible to the caller.

Two MUSA policies shape `CompilationConfig` before vLLM compiles:

- MCCL collectives (`_c10d_functional::*`) are appended to `splitting_ops`
  because `ProcessGroupMCCL` submits and polls them on its own stream, which is
  illegal while a graph is recording.
- The capture ladder is thinned to at most 10 sizes, ordered and unique, always
  keeping the largest batch size. The torch_musa graph pool is not shared across
  captures, so every captured size costs a full activation footprint (~2.4 GiB
  for a 27B model at TP4 with a 512-token batch); vLLM's default 51-size ladder
  needs more memory than the card has.

Capture, replay and eager execution failures propagate. A different TorchMUSA
version needs separate validation. The policies above define resource behavior;
they do not establish a latency or throughput guarantee.

## Known limitations

- FlagGems FlagTune is disabled by default (`USE_FLAGTUNE=0`): its cost-model
  bundle is fetched from a CDN that offline hosts cannot reach. FlagGems kernels
  and local Triton autotuning stay enabled; `VLLM_FL_MUSA_FLAGTUNE=1` opts back
  into the remote model.
- FlagCX is not available for MUSA; distributed runs use native MCCL (`mccl`).
  The FlagGems + FlagTree + FlagCX combined acceptance gate is therefore blocked
  for this platform and reported as such.
- FULL cudagraph mode is unsupported; graph runs stay PIECEWISE.
- MoE `topk_softmax` is routed through the plugin dispatch (FlagGems/reference).
  The 0.32-only `is_padding` mask and `e_score_correction_bias` arguments are
  accepted but not applied; MoE bias correction is not exercised on MUSA yet.

## CI coverage

The `v0.32.0-musa-ci` image referenced by the CI configuration is built with
`./docker/build.sh --platform musa --target ci` and pushed to the `plugin`
project. The 0.32.0 adaptation was verified on the FlagOS 2.2 runtime image with
the plugin installed editable, using
`tools/adaptation-gate-cases/run_serve_graph.sh` / `run_serve_eager.sh` followed
by `run_test.sh` for Qwen3.6-27B and Qwen3.6-35B-A3B in both eager and graph
modes.

MUSA uses the repository's
[shared platform workflow](../../.github/workflows/_platform_test.yml) for a
combined unit/functional job, E2E inference and serving, and benchmark smoke. The
[platform test configuration](../../tests/platforms/musa.yaml) currently lists
four inference and four serving cases for PR validation (27B and 35B-A3B in eager
and graph modes, TP4). Nightly adds the two 0.6B smoke cases, the same four
27B/35B modes and two concurrent cases; MUSA E2E cases run serially to avoid
overlapping MCCL group initialization.

Use the repository's `ci.yml` platform selection. Runner labels and device
environment settings come from the
[CI configuration](../../.github/configs/musa.yml); model cases and test settings
come from the platform test configuration. Keep the original model checks,
capture sizes, memory budgets and timeouts when validating changes. Consult the
actual CI run for its source, platform results and conditional skips.

Installation checks or CPU tests alone do not establish device-kernel,
multimodal-model, native ABI or performance acceptance.

## Adaptation record

The 0.32.0 adaptation was accepted on 2026-11-05:

| Field | Value |
|---|---|
| Date | 2026-11-05 |
| Platform | MUSA MTT S5000 ×4 (80 GiB per card) |
| Host / container | `10.121.38.14`, container `flagos22` |
| Branch | `vllm-plugin-FL` `0.5.0-dev` (vLLM 0.32.0 `empty` build) |
| Models | Qwen3.6-27B, Qwen3.6-35B-A3B (TP4) |
| Runtime | `flagos-runtime-mthreads-musa4.3.6:2.2.0`, plugin installed editable |
| CI image | `plugin/vllm-plugin-fl:v0.32.0-musa-ci@sha256:62acf6a6590ce0c7be91130ec8f8e5def01ab5572400d74ddb6907fd11ead0fa` |
| Dependency tuple | torch 2.9.0 (`torch.version.musa` = 40306), FlagGems 5.4.0, FlagTree 0.7.0+mthreads3.6, Triton 3.6.0 |
| Gate | 4/4 groups `GATE_TEST_RC=0` (27B and 35B-A3B × eager and graph) |
| Unit tests | raw full suite 2004 passed, 225 skipped, 1 failed (`test_gcu_kv_zeroer`, outside MUSA); `FL_TEST_PLATFORM=musa` (CI path) 1884 passed, 231 skipped, 0 failed |
| Evidence | Validation host `/data/mt032/final_{27b,35b}_{graph,eager}/` (`*.server.log`, `*.test.log`, gate results) |
| Status | Accepted for MUSA; FlagCX combined gate not applicable (native MCCL) |

Key changes in this adaptation:

- vLLM 0.32 API drift: the runner passes the Mamba state-copy mapping returned by
  `get_mamba_state_copy_funcs`, and MoE `topk_softmax` is redirected to the plugin
  dispatch through the `vllm._custom_ops` leaf module.
- `DeviceConfig` keeps a device with no index, so every TP rank binds to its own
  card; the compatibility fallback handles both `auto` and `musa` device strings.
- Graphs: MCCL collectives are split out of the captured graph, and the capture
  ladder is thinned to at most 10 sizes because the torch_musa pool is not shared.
- FlagTune is disabled by default (`USE_FLAGTUNE=0`) so serving does not require
  the remote cost-model bundle.
- CI setup asserts the pinned tuple (vLLM 0.32, FlagGems 5.4, FlagTree 0.7,
  Triton 3.6, torch 2.9).

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
harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-musa-ci@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06
```

The image provides the vendor runtime, an ordinary empty vLLM wheel and the
MUSA dependency wheels. Preserve its Torch, TorchMUSA and SDK packages when
installing the plugin. Generic PyPI Torch, CUDA wheels or an unpatched FlagTree
build are not substitutes for this MUSA stack.

## Install the current plugin

Start the image with the MUSA runtime and the devices assigned to your job.
Mount model data read-only. Set `MTHREADS_VISIBLE_DEVICES` to the allocated
physical cards and `MUSA_VISIBLE_DEVICES` to their logical order.

Inside the container, build and install the current checkout as an ordinary
wheel using the image's interpreter. Use a new output directory and leave
setuptools-scm version overrides unset:

```bash
export GEMS_VENDOR=mthreads
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
PYTHON=/opt/musa/venv/bin/python
WHEEL_DIR=$(mktemp -d)
env -u VLLM_VENDOR "$PYTHON" -I -B -m pip wheel \
  --no-deps --no-build-isolation --wheel-dir "$WHEEL_DIR" .
"$PYTHON" -I -B -m pip install --no-deps "$WHEEL_DIR"/vllm_plugin_fl-*.whl
"$PYTHON" -I -B .github/scripts/install_flaggems_vllm.py
```

The last command installs the shared FlagGems-vllm provider at the release
pinned by the repository. It checks that the existing runtime package versions
remain unchanged. Run it with the same interpreter that runs vLLM.

The [MUSA setup script](../../.github/scripts/musa/setup.sh) uses the image's
existing interpreter. It installs the current checkout with `VLLM_VENDOR` unset,
adds the reporting and timeout plugins required by the shared test runner, and
installs the shared FlagGems-vllm provider. Preserve the image's vendor Torch,
TorchMUSA and SDK packages.

## Run inference

Use a local model directory and set tensor parallelism to the number of cards
allocated to this process. For example, a four-card server uses:

```bash
"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --tensor-parallel-size 4
```

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
[shared platform workflow](../../.github/workflows/_platform_test.yml) for unit
tests, functional tests, E2E inference and serving, and benchmark smoke. The
[platform test configuration](../../tests/platforms/musa.yaml) currently lists
four inference and three serving configurations.

Use the repository's `ci.yml` platform selection. Runner labels and device
environment settings come from the
[CI configuration](../../.github/configs/musa.yml); model cases and test settings
come from the platform test configuration. Keep the original model checks,
capture sizes, memory budgets and timeouts when validating changes. Consult the
actual CI run for its source, platform results and conditional skips.

## Dependency limitations

The pinned MUSA dependency profiles include corrections beyond generic upstream
releases. Track their individual scope in [FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973),
[FlagGems #7005](https://github.com/flagos-ai/FlagGems/issues/7005),
[FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344) and
[FlagTree #1346](https://github.com/flagos-ai/FlagTree/issues/1346).
Installation checks or CPU tests alone do not establish device-kernel,
multimodal-model, native ABI or performance acceptance.

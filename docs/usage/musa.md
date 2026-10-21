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
harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06
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

CI uses a fresh private environment and reinstalls the image's pinned ordinary
Gems, Tree and PyYAML wheels there. A nested environment must not assume that
`--system-site-packages` also inherits packages from the image's parent virtual
environment. The [MUSA setup script](../../.github/scripts/musa/setup.sh) performs
that installation without replacing the vendor Torch or SDK.

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

MUSA graph handling is isolated from the shared graph implementation. Captured
inputs stay owned by the graph; output views are reconstructed from storage and
tensor metadata. Stable captured buffers replay. Calls with replacement input
allocations execute the original callable so mutations and output aliases remain
visible to the caller.

All configured capture sizes are retained. Initialization and memory profiling
may rotate native executables to keep at most 1900 resident per process. Runtime
replay only uses resident executables; a cold piece executes the original
callable without synchronizing, destroying or instantiating an executable.

The initialization helper requires Linux, a C++ compiler, Ninja and the installed
PyTorch/TorchMUSA 2.9 and MUSA SDK headers. TorchMUSA 2.9 has no public API to
release only an executable while retaining its captured graph and RNG state;
its public [`reset()`](https://github.com/MooreThreads/torch_musa/blob/1dc78726ff84bb2da61697055be0d037a8a29170/torch_musa/csrc/aten/musa/MUSAGraph.cpp) releases the template too. The small version-bound native
shim supplies that missing operation and propagates loading or native failures.
A different TorchMUSA version or header layout needs separate validation.

The executable cap and runtime fallback define resource behavior; they do not
establish a GPU latency or throughput guarantee. Representative hardware
bucket-switching latency and broad ABI compatibility remain unmeasured.

## CI coverage

MUSA runs unit tests, functional tests, the four inference and three serving
configurations, and benchmark smoke serially in one allocated container. The
workflow checks the physical device mapping and idle state before Torch starts,
then cleans up only processes owned by the current job. These checks do not
replace the runner scheduler's resource allocation.

Use the repository's `ci.yml` platform selection and the existing allocation
inputs for runner labels, device ordinals, PCI IDs and model locations. The
`model_27b` and `model_35b_a3b` allocation keys override the corresponding E2E
model paths. Keep the original model checks, capture sizes, memory budgets and
timeouts when comparing changes. Consult the actual CI run for its source,
platform results and conditional skips.

## Dependency limitations

The pinned MUSA dependency profiles include corrections beyond generic upstream
releases. Track their individual scope in [FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973),
[FlagGems #7005](https://github.com/flagos-ai/FlagGems/issues/7005),
[FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344) and
[FlagTree #1346](https://github.com/flagos-ai/FlagTree/issues/1346).
Installation checks or CPU tests alone do not establish device-kernel,
multimodal-model, native ABI or performance acceptance.

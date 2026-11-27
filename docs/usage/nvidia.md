# NVIDIA: vLLM 0.32 CUDA deployment

NVIDIA uses the full official CUDA distribution of vLLM 0.32.0, including its
native extensions, on `0.5.0-dev`. The plugin keeps FlagGems enabled and adds FL
dispatch and diagnostics around the upstream CUDA platform and V1/V2 runners.
The `VLLM_TARGET_DEVICE=empty` installation is for other hardware backends.

## Software stack

| Component | Current source configuration |
|---|---|
| NVIDIA CI image | `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.32.0-cuda-dev` |
| Build base | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.0.3:2.3.0` |
| vLLM | Full CUDA `0.32.0`; preserve the installed native extensions |
| Image-baked FlagGems-vllm | `0.3.0.dev20261119` in the current Dockerfile |
| CI FlagGems-vllm override | Source `b5c2c2249319277ccda4d41aa0f65164bd9b2a19` (`v0.2.0`; package metadata reports `0.1.0`) |
| FlagGems and FlagTree | Inherited from the selected FlagOS runtime; record installed versions and active compiler paths |
| Runtime defaults | FlagGems enabled, FlagTune disabled; CI explicitly selects Runner V2 |

The current recipe follows [#785](https://github.com/flagos-ai/vllm-plugin-FL/pull/785)
and [docker/cuda/Dockerfile](../../docker/cuda/Dockerfile). The
[NVIDIA CI setup](../../.github/scripts/nvidia/setup.sh) replaces the image-baked
provider with the qualified `b5c2c224` source after graph-profiling faults were
observed with the development package. Record the image and the final interpreter
separately. The earlier development-commit stack in
[#615](https://github.com/flagos-ai/vllm-plugin-FL/pull/615) is historical validation.

Use the image reference selected in [the NVIDIA CI configuration](../../.github/configs/nvidia.yml).
Record its resolved digest, plugin commit and installed dependency versions.
The KV-cache API update in [#737](https://github.com/flagos-ai/vllm-plugin-FL/pull/737)
adds no new FlagGems, FlagGems-vllm, FlagTree or FlagCX revision requirement.
Separate upstream test and CI fixes are not runtime prerequisites for this image.
Do not install a second Triton distribution over FlagTree or replace the
official vLLM CUDA package with an EMPTY wheel.

Keep the plugin checkout and its
[NVIDIA dispatch policy](../../vllm_fl/dispatch/config/nvidia.yaml) together.
Compatible explicit fused operators use FlagGems-vllm; general ATen registration
remains with FlagGems. See the [provider guide](../flaggems-vllm.md).

## Prepare the container

Use Linux, Docker with NVIDIA Container Toolkit, and a driver compatible with
CUDA 13.0.3. Check the selected GPUs with `nvidia-smi`. Run long remote setup and
serving commands inside `tmux`.

Prepare a checkout containing the adaptation and complete local checkpoints.
Before the adaptation is merged, use its PR revision instead of the target
branch.

```bash
git clone --branch 0.5.0-dev https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL
git rev-parse HEAD
export REPO_DIR="$PWD"
export MODELS_DIR=/data/models
export RUN_DIR=/data/vllm-fl-runs
mkdir -p "$RUN_DIR"
export IMAGE=harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.32.0-cuda-dev

docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
docker run --rm -it --name vllm-fl-nvidia \
  --gpus all --ipc=host --ulimit nofile=65535:65535 \
  -p 127.0.0.1:8000:8000 \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v "$MODELS_DIR:/models:ro" \
  -v "$RUN_DIR:/results" \
  -w /workspace/vllm-plugin-FL \
  --entrypoint /bin/bash "$IMAGE"
```

If registry authentication is required, run `docker login harbor.baai.ac.cn`
before pulling. Install the mounted plugin checkout inside the container:

```bash
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1 GEMS_VENDOR=nvidia USE_FLAGTUNE=0
export PYTHONPATH=/opt/triton${PYTHONPATH:+:$PYTHONPATH}

python .github/scripts/install_flaggems_vllm.py --source-archive
python -c 'import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)'
env -u VLLM_VENDOR python -m pip install \
  --break-system-packages --no-build-isolation --no-deps -e .
python -c 'import vllm_fl; from vllm import _C; print(vllm_fl.__file__, _C.__file__)'
```

The plugin uses the official vLLM CUDA extensions and installs its Python
integration without building `vllm_fl._C`. Record the native vLLM package,
final provider revision, active Triton and plugin import paths. Avoid
running from an upstream vLLM source checkout or shadowing its installed package
through `PYTHONPATH`.

Leave attention, cache and quantization backend selection at upstream defaults
unless configuring a supported deployment explicitly. Native CUDA kernels, UVA
and graph fusions remain available. The examples use NCCL; leave `FLAGCX_PATH`
unset unless intentionally configuring and validating FlagCX.

## Basic service and runner selection

Run inside the prepared container with a local `Qwen3-4B` checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/Qwen3-4B \
  --served-model-name fl-model
```

The example uses upstream defaults for the runner, dtype, memory, batching and
compilation. The service listens on port 8000. Adjust resource limits or tensor
parallelism only when your model and hardware require them.

To select upstream V1 explicitly, stop that service and run:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm serve /models/Qwen3-4B --served-model-name fl-model
```

Set `VLLM_USE_V2_MODEL_RUNNER=1` to select V2 explicitly. Both runners are
inherited from upstream through `NvidiaWorkerFL`; confirm the selected runner
in the worker logs. `--enforce-eager` is optional for debugging and is independent
of runner selection.

From another host terminal, check readiness and send a request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fl-model","messages":[{"role":"user","content":"What is 2 + 2? Give only the number."}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

Check that the response answers the question correctly.

## Model coverage

These are the requested model validation targets. Results must come from the
current full-CUDA image and plugin revision; earlier EMPTY runs do not validate
this configuration. Run inference on the CI runner's allocated devices with
complete checkpoints and an appropriate memory/parallelism configuration.

| Model | Validation scope |
|---|---|
| Qwen3-4B | Basic inference, upstream V1/V2 selection, eager and graph execution |
| Qwen3.8-Flash-Next | Text generation and repeated greedy requests |
| DeepSeek-V4.1-Flash | Official checkpoint quantization, text generation and repeated greedy requests |
| MiniCPM-V-4.7-1B | Text, image, video and paired-image correctness |
| Hy4-preview | Official checkpoint loading and model inference on sufficient allocated hardware |

For example, start MiniCPM's multimodal service with:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/MiniCPM-V-4.7-1B \
  --served-model-name fl-model
```

Use an `image_url` or `video_url` item in the chat message for media input.
For paired images, send two `image_url` items in the same message. Evaluate the
answer against the supplied media.

## Build and CI

Build the current CUDA development image from the repository root when needed:

```bash
docker build --target dev -f docker/cuda/Dockerfile -t vllm-fl-local:v0.32.0-cuda-dev .
```

[Dockerfile](../../docker/cuda/Dockerfile) retains the full native vLLM CUDA
package and inherits the FlagOS runtime. CI
[setup.sh](../../.github/scripts/nvidia/setup.sh) installs the provider source
override and this checkout with its test dependencies. The older `Dockerfile.ci`
recipe is separate from the image currently selected by `nvidia.yml`.

NVIDIA is the enabled backend for this adaptation. Existing CI runners execute
unit and model tests using the selected image; their labels, model mounts and
test entry points are defined in the existing workflow. The usage examples
above do not add automatic CI jobs. See the [test guide](../../tests/README.md).

## Troubleshooting

- **Missing vLLM CUDA extension:** restore the selected full-CUDA runtime and
  verify the native extension import. An EMPTY wheel is not supported on NVIDIA.
- **Wrong dependency/provider:** restore the pinned image or rebuild its recipe;
  keep FlagGems enabled and the matching NVIDIA dispatch policy installed.
- **Incorrect model output:** retain prompts, generated token IDs, worker logs,
  image digest and plugin/dependency revisions. For media, also retain the input
  and preprocessing configuration.
- **Memory failure:** distinguish weight loading, temporary buffers and KV-cache
  allocation. Select an appropriate topology and request limits; reducing GPU
  memory utilization does not shrink model weights.
- **Slow first request:** inspect compiler and worker logs. Compilation cache
  state affects startup time; GPU utilization alone does not identify a hang.

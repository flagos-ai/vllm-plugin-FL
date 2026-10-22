# NVIDIA: vLLM 0.32 CUDA deployment

NVIDIA uses the full official CUDA distribution of vLLM 0.32.0, including its
native extensions, on `0.5.0-dev`. The plugin keeps FlagGems enabled and adds FL
dispatch and diagnostics around the upstream CUDA platform and V1/V2 runners.
The `VLLM_TARGET_DEVICE=empty` installation is for other hardware backends.

## Software stack

| Component | Pinned configuration |
|---|---|
| NVIDIA CI image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-cuda-ci` |
| Official base | `vllm/vllm-openai:v0.32.0-cu129` |
| vLLM | Full official CUDA `0.32.0`; preserve the installed native extensions |
| FlagGems | Development commit `6fe7709e7717e49cf12bb00346719a0885f505fa` |
| FlagGems-vllm | Commit `b5c2c2249319277ccda4d41aa0f65164bd9b2a19` (`v0.2.0`; package metadata reports `0.1.0`) |
| FlagTree | Development commit `317a0249ec25c6756e0aee3a87e8e3c92f154335`; supplies Triton `3.6.0` |
| Runtime defaults | FlagGems enabled, FlagTune disabled, upstream V2 runner |

Use the image reference selected in [the CUDA CI configuration](../../.github/configs/cuda.yml).
Record the resolved image digest and plugin commit for each deployment. The
image records source and wheel provenance in `/opt/fl-ci/provenance.json`.
These development revisions are fixed rather than floating branch installs.
Do not install a second Triton distribution over FlagTree or replace the
official vLLM CUDA package with an EMPTY wheel.

Keep the plugin checkout and its
[NVIDIA dispatch policy](../../vllm_fl/dispatch/config/nvidia.yaml) together.
Compatible explicit fused operators use FlagGems-vllm; general ATen registration
remains with FlagGems. See the [provider guide](../flaggems-vllm.md).

## Prepare the container

Use Linux, Docker with NVIDIA Container Toolkit, and a driver compatible with
CUDA 12.9. Check the selected GPUs with `nvidia-smi`. Run long remote setup and
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
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-cuda-ci

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
export VLLM_USE_V2_MODEL_RUNNER=1
export FL_STACK_PROVENANCE=/opt/fl-ci/provenance.json

python .github/scripts/install_flaggems_vllm.py
python -I -B .github/scripts/cuda/stack.py --runtime-only
env -u VLLM_VENDOR python -m pip install \
  --break-system-packages --no-build-isolation --no-deps -e .
python -I -B .github/scripts/cuda/stack.py
```

The plugin uses the official vLLM CUDA extensions and installs its Python
integration without building `vllm_fl._C`. The stack checks verify the native
vLLM package, pinned FL dependencies, active Triton provider and plugin
installation. Avoid
running from an upstream vLLM source checkout or shadowing its installed package
through `PYTHONPATH`.

Leave attention, cache and quantization backend selection at upstream defaults
unless configuring a supported deployment explicitly. Native CUDA kernels, UVA
and graph fusions remain available. The examples use NCCL; leave `FLAGCX_PATH`
unset unless intentionally configuring and validating FlagCX.

## Basic service and runner selection

Run inside the prepared container with a local `Qwen3-4B` checkpoint. This
example selects upstream V2 with eager execution:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm serve /models/Qwen3-4B \
  --served-model-name fl-model --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.25 --seed 0 --enforce-eager
```

To use upstream V1 with CUDA graphs, stop that service and run:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm serve /models/Qwen3-4B \
  --served-model-name fl-model --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.25 --seed 0 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'
```

Both runners are inherited from upstream through `NvidiaWorkerFL`. Eager and
graph execution are independent of runner selection: omit `--enforce-eager`
for graph execution with either runner. Confirm the selected runner in the
worker logs.

From another host terminal, wait for readiness and send a deterministic request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fl-model","messages":[{"role":"user","content":"What is 2 + 2? Give only the number."}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false,"thinking":false},"return_token_ids":true}'
```

Check the answer, then repeat the request and compare the generated token IDs.
Health and successful model construction alone do not validate generation.

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
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm serve /models/MiniCPM-V-4.7-1B \
  --served-model-name fl-model --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.30 --seed 0 --enforce-eager \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":4}}'
```

Use an `image_url` or `video_url` item in the chat message for media input.
For paired images, restart with `--limit-mm-per-prompt '{"image":2,"video":0}'`
and `--max-num-batched-tokens 4096`, then send two `image_url` items in one
message. Evaluate the answer against the supplied media; a nonempty response
alone is not a correctness check.

## Build and CI

Build the native CUDA CI image from the repository root when needed:

```bash
bash docker/build.sh --platform cuda --target ci \
  --image-name vllm-fl-local --image-tag v0.32.0-cuda-ci
```

[Dockerfile.ci](../../docker/cuda/Dockerfile.ci) retains the official vLLM CUDA
installation and installs the fixed FlagGems, FlagTree and FlagGems-vllm sources.
The build records provenance and checks the runtime. Install the plugin checkout
and run the full stack check after starting the resulting image.

CI [setup.sh](../../.github/scripts/cuda/setup.sh) verifies the provider and
runtime pins, installs the plugin with `--no-deps`, then checks the stack and audio
dependencies. Audio checks require access to the official vLLM audio fixtures
or a populated `VLLM_ASSETS_CACHE`.

NVIDIA is the enabled backend for this adaptation. Existing CI runners execute
unit and model tests using the selected image; their labels, model mounts and
test entry points are defined in the existing workflow. The usage examples
above do not add automatic CI jobs. See the [test guide](../../tests/README.md).

## Troubleshooting

- **Missing vLLM CUDA extension:** restore the full official CUDA image and run
  the stack check. An EMPTY wheel is not a supported NVIDIA installation.
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

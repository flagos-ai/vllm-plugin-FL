# NVIDIA: vLLM 0.32 EMPTY deployment

Use the pinned image below with the NVIDIA implementation on `0.5.0-dev`.
It contains an ordinary `vllm==0.32.0+empty` wheel: upstream Python model and
runner code is present, but upstream CUDA extensions are absent. The plugin
supplies its own extension and the required FlagGems/Triton operator adapters.

## Software stack

| Component | Pinned configuration |
|---|---|
| Image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-cuda-empty-ci` |
| Manifest digest | `sha256:28a000b21c00ba11b9ddd8e01f147cb3d18f24594b071295fcd4921cd30d58c0` |
| Official image base | `vllm/vllm-openai:v0.32.0-cu129` |
| vLLM | Distribution `0.32.0+empty`; module `0.32.0` |
| Python / PyTorch / CUDA runtime | `3.12.3` / `2.13.0+cu129` / `12.9` |
| FlagGems | `5.4.0+g6fe7709e7717`, source `6fe7709e7717e49cf12bb00346719a0885f505fa` |
| FlagTree | `0.7.0+git317a0249`, source `317a0249ec25c6756e0aee3a87e8e3c92f154335` |
| Triton | `3.6.0`, supplied by FlagTree; no standalone `triton` distribution |
| Humming / TorchCodec | `0.1.17` / `0.16.0+cu129` |
| Runtime defaults | FlagGems enabled, FlagTune disabled, native V2 runner |

These FlagGems and FlagTree development revisions are fixed, not floating
`master`/`main` installations. Keep the plugin checkout and its
[NVIDIA dispatch policy](../../vllm_fl/dispatch/config/nvidia.yaml) together.
Do not resolve or upgrade dependencies after installing the pinned FL wheels.
The image records source and wheel provenance in `/opt/fl-ci/provenance.json`.

## Prepare the host and container

Use Linux, Docker with NVIDIA Container Toolkit, and a driver compatible with
CUDA 12.9. Check the selected GPUs with `nvidia-smi`. Run long remote setup and
serving sessions inside `tmux`.

On the host, prepare a checkout containing the 0.32 adaptation and a directory
of complete local checkpoints. Before the adaptation is merged, check out its
PR revision instead of the target branch; `main` is not this deployment's base.

```bash
git clone --branch 0.5.0-dev https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL
git rev-parse HEAD  # Record the exact deployment revision.
export REPO_DIR="$PWD"
export MODELS_DIR=/data/models
export RUN_DIR=/data/vllm-fl-runs
mkdir -p "$RUN_DIR"
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-cuda-empty-ci@sha256:28a000b21c00ba11b9ddd8e01f147cb3d18f24594b071295fcd4921cd30d58c0

docker pull "$IMAGE"
docker run --rm -it --name vllm-fl-nvidia \
  --gpus all --ipc=host --ulimit nofile=65535:65535 \
  -p 127.0.0.1:8000:8000 \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v "$MODELS_DIR:/models:ro" \
  -v "$RUN_DIR:/results" \
  -w /workspace/vllm-plugin-FL \
  --entrypoint /bin/bash "$IMAGE"
```

If the registry requires authentication, run `docker login harbor.baai.ac.cn`
before pulling. The container has the runtime dependencies; install the plugin
from the mounted checkout inside it:

```bash
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1 GEMS_VENDOR=nvidia USE_FLAGTUNE=0
export VLLM_USE_V2_MODEL_RUNNER=1
export FL_STACK_PROVENANCE=/opt/fl-ci/provenance.json

python -I -B .github/scripts/cuda/stack.py --runtime-only
VLLM_VENDOR=cuda MAX_JOBS=4 python -m pip install \
  --break-system-packages --no-build-isolation --no-deps -e .
python -I -B .github/scripts/cuda/stack.py
```

`VLLM_VENDOR=cuda` builds `vllm_fl._C`; an editable Python-only installation is
insufficient. The first check validates the ordinary EMPTY wheel, absence of
upstream CUDA payloads, FL wheel provenance and active Triton provider. The
second also checks the plugin extension, NVIDIA platform, worker imports and
visible GPU. Do not run from an upstream vLLM source checkout or shadow its
installed package through `PYTHONPATH`.

The examples use the default native NCCL communication path. Leave
`FLAGCX_PATH` unset unless intentionally configuring and validating FlagCX.
Keep attention selection automatic: the plugin selects available providers and
installs architecture-specific adapters. Forcing native FlashAttention or
FlashMLA kernels that the EMPTY wheel does not contain is not a replacement
for enabling FlagGems.

## Basic service and native runner selection

The commands below run inside the prepared container. Use one available GPU
and a local `Qwen3-4B` checkpoint. This V2 example uses eager execution:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm serve /models/Qwen3-4B \
  --served-model-name fl-model --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.25 --seed 0 --enforce-eager
```

To exercise native V1 with CUDA graphs, stop that service and run the same
command with `VLLM_USE_V2_MODEL_RUNNER=0` and without `--enforce-eager`:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm serve /models/Qwen3-4B \
  --served-model-name fl-model --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.25 --seed 0 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'
```

Both selections use upstream runners through `NvidiaWorkerFL`. Check the
worker log's runner selection; the environment variable alone is not evidence
that a requested runner was instantiated.

From another host terminal, wait for readiness and send a deterministic request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fl-model","messages":[{"role":"user","content":"What is 2 + 2? Give only the number."}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false,"thinking":false},"return_token_ids":true}'
```

Expect answer `4`. Repeat the same request and compare the generated
`choices[0].token_ids`, as well as the answer. Health or model construction
alone does not validate generation. Stop the foreground service with Ctrl+C
before starting another example.

## Validated model configurations

The following smoke tests were completed on plugin revision `0fbe20f9`
(2026-10-20). They are not minimum hardware requirements or performance
benchmarks. All four new-model runs used native V2, eager execution and the
pinned development dependencies. The regression entry point below was added
after these runs; its execution status is reported separately in the PR.

| Model/checkpoint | Hardware and configuration | Validation scope |
|---|---|---|
| Qwen3-4B | H20, TP1; V1 graph and V2 eager | Four short answers and repeated greedy tokens |
| Qwen3.8-Flash-Next, official BF16 | 8 × H100 80 GB, TP8; length 2048, sequences 8, batch tokens 512, utilization 0.90; Engram on GPU | Four short text answers and repeated greedy tokens; image/video disabled |
| DeepSeek-V4.1-Flash, official mixed-precision checkpoint | 8 × H100 80 GB, TP8; length 1024, sequences 1, batch tokens 1024, utilization 0.94; Engram on GPU, FP8 indexer | Four short answers and repeated greedy tokens; checkpoint quantization preserved |
| MiniCPM-V-4.7-1B | H20, TP1; length 4096, sequences 4, utilization 0.30 | Text, one image, four-frame video and paired images; semantic checks, repeatability and native-reference token agreement |
| Hy4-preview, official BF16 | Two nodes, each 8 × H100 80 GB and about 2 TiB host RAM; TP8 × PP2; selective non-UVA CPU offload | Full 131-shard checkpoint, one arithmetic prompt and deterministic repeat |

The large-model H100 runs used isolated environments with the same verified
0.32 EMPTY runtime and exact FL wheels; those environments were prepared from
an existing official image, rather than by running the published CI image
itself. Unit tests and the basic native-runner checks also ran in the published
0.32 CI image. These results do not establish broad accuracy, long-context
support, all quantization modes, or graph execution for the four new models.

### Qwen3.8-Flash-Next

Make all eight assigned GPUs visible. GPU Engram storage is explicit because
this EMPTY configuration does not provide native host-memory UVA views.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm serve /models/Qwen3.8-Flash-Next \
  --served-model-name fl-model --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 8 --pipeline-parallel-size 1 \
  --engram-config '{"cpu_offload":false}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.90 --seed 0 --enforce-eager
```

Use the chat request above with `enable_thinking:false` for the text smoke test.

### DeepSeek-V4.1-Flash

Use the complete official checkpoint and retain automatic weight/cache dtype
and quantization selection. The tested path uses H100/SM90, FP8 indexing and
no speculative decoding. TP16 is not interchangeable with TP8 for this model's
head/group dimensions.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm serve /models/DeepSeek-V4.1-Flash \
  --served-model-name fl-model --host 0.0.0.0 --port 8000 \
  --dtype auto --kv-cache-dtype auto \
  --tensor-parallel-size 8 --pipeline-parallel-size 1 \
  --engram-config '{"cpu_offload":false}' \
  --attention-config '{"indexer_kv_dtype":"fp8","indexer_sparse_logits":false}' \
  --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.94 --seed 0 --enforce-eager
```

Keep the automatically selected DS attention path. The plugin supplies real
operators for the missing EMPTY kernels; a native-capability flag should not
be forced to claim that DeepGEMM or FlashMLA binaries are present. The measured
weight footprint was about 71.94 GiB per rank before KV-cache/runtime needs.

### MiniCPM-V-4.7-1B: text, images and video

Use the **1B** checkpoint. The following service supports the tested text,
one-image and four-frame-video configuration:

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

Use the basic chat request for text. For an image, run this client where the
image is accessible, with `IMAGE_PATH` set to a PNG file:

```bash
export IMAGE_PATH=/path/to/image.png
python - <<'PY'
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

encoded = base64.b64encode(Path(os.environ["IMAGE_PATH"]).read_bytes()).decode()
body = {
    "model": "fl-model",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        {"type": "text", "text": "Describe the image in English."},
    ]}],
    "temperature": 0, "max_tokens": 96,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = Request("http://127.0.0.1:8000/v1/chat/completions",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
PY
```

For video, use a `video_url` content item with an accessible URL or an MP4 data
URL (`data:video/mp4;base64,...`). For paired images, restart with
`--limit-mm-per-prompt '{"image":2,"video":0}'` and
`--max-num-batched-tokens 4096`, then send two `image_url` items in the same
message. Evaluate the answer against the supplied media; a nonempty response
alone is not a multimodal correctness check.

### Hy4-preview: official BF16 on two nodes

This example uses the complete official BF16 checkpoint, approximately 1.56 TB
on disk. A differently quantized or `Testing` checkpoint is not equivalent.
Each node needs eight H100 80 GB GPUs, sufficient CPU RAM for selective offload
and loading, and working cross-node NCCL/RDMA. Use the same image digest, plugin
revision and checkpoint on both nodes.

Create the container on each node as above, but replace the `-p` option with
`--network host`. For RDMA-capable hosts, also expose the allocated RDMA devices
(e.g. `--device=/dev/infiniband`) and use `--ulimit memlock=-1 --cap-add IPC_LOCK`
according to the cluster's container policy. Repeat the plugin installation and
strict stack check on each node. Verify cross-node communication before loading
the full model.

Inside each container, set the following to the real routable addresses and
interfaces of that node. `MASTER_ADDR` must identify node 0 on both nodes;
`VLLM_HOST_IP` must identify the current node. `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME` select the bootstrap interfaces, not the RDMA HCA names.

```bash
export NODE_RANK=0  # Set to 1 in the second node's container.
export MASTER_ADDR='<node-0-routable-address>'
export VLLM_HOST_IP='<this-node-routable-address>'
export NCCL_SOCKET_IFNAME='<bootstrap-interface>'
export GLOO_SOCKET_IFNAME='<bootstrap-interface>'
export NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

endpoint=(--host 127.0.0.1 --port 8000)
if [ "$NODE_RANK" = 1 ]; then endpoint=(--headless); fi
vllm serve /models/Hy4-preview \
  --served-model-name fl-hy4 \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29575 \
  --distributed-executor-backend mp --data-parallel-backend mp \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 --dtype bfloat16 \
  --cpu-offload-gb 32 --cpu-offload-params w13_weight w2_weight \
  --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 --seed 0 --enforce-eager \
  --reasoning-parser hy_v4 "${endpoint[@]}"
```

Replace the angle-bracket placeholders before running. Start both node commands
in separate persistent sessions. Allow the selected master port and NCCL
communication within the trusted cluster network; the HTTP endpoint remains
bound to node 0's loopback interface.

Offload **only** `w13_weight` and `w2_weight`. Offloading every parameter can
invalidate attention/cache aliases. The 32 GiB threshold operates on whole
parameters: the tested configuration offloaded 33 GiB per worker, with temporary
GPU copies during execution. Preserve the selector and the non-UVA setting.

After node 0 reports readiness, run there:

```bash
curl --fail --max-time 3600 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fl-hy4","messages":[{"role":"user","content":"What is 2 + 2? Reply with exactly the digit 4."}],"temperature":0,"seed":0,"max_tokens":16,"chat_template_kwargs":{"reasoning_effort":"no_think"},"return_token_ids":true}'
```

Repeat and check answer `4` and identical generated token IDs. The 1800-second
execute-model deadline and 3600-second client deadline are specific to this
large-model validation; they do not change CI timeouts. Loading all shards,
finishing profile runs or returning health status does not replace this check.

## Build the pinned runtime image

If the published image is unavailable, build the checked-in EMPTY recipe from
the repository root:

```bash
bash docker/build.sh --platform cuda --target ci \
  --image-name vllm-fl-local --image-tag v0.32.0-cuda-empty-ci
```

This selects [Dockerfile.empty](../../docker/cuda/Dockerfile.empty), checksums
fixed upstream/FL source archives, builds ordinary wheels, replaces the
upstream vLLM and standalone Triton packages inside the image, and validates
its generated provenance. FlagTree compilation and its LLVM dependencies make
this a substantial build. Install the plugin checkout and run the same stack
checks after starting the resulting image. Other CUDA build targets do not
select this EMPTY recipe.

## CI and local regression

The CUDA configuration pins the same image digest. CI setup uses
[setup.sh](../../.github/scripts/cuda/setup.sh), which runs the runtime check,
builds `vllm_fl._C` without dependency resolution, then runs the full stack check
and audio dependency checks. Inside the prepared GPU container:

```bash
bash .github/scripts/cuda/check.sh
bash .github/scripts/cuda/setup.sh
python tests/run.py --platform cuda --scope unit
```

Run a selected checkpoint regression in an independently allocated container:

```bash
python .github/scripts/cuda/run_regression.py --list
python .github/scripts/cuda/run_regression.py \
  --model qwen3 --case 4b_v2_eager \
  --model-path /models/Qwen3-4B --output-dir /results/qwen3-v2
```

| `--model` | `--case` | Required checkpoint and devices |
|---|---|---|
| `qwen3` | `4b_v1_graph`, `4b_v2_eager` | Qwen3-4B; one GPU with at least 48 GiB |
| `minicpm4_7` | `1b_multimodal`, `1b_paired_images` | MiniCPM-V-4.7-1B; one GPU with at least 24 GiB |
| `qwen3_8` | `flash_next_tp8_eager` | Qwen3.8-Flash-Next; eight H100-class GPUs |
| `deepseek_v4_1` | `flash_tp8_eager` | DeepSeek-V4.1-Flash; eight H100 GPUs |

Add `--check-only` for preflight without model inference; it is not a generation
pass. The runner preserves pytest reports, logs and result/worker facts under
`--output-dir`. The new large-model cases are manual selections, not additions
to the automatic H20 matrix. Hy4 remains the two-node procedure above. See
[the test guide](../../tests/README.md) for CI details; unit-test success does
not imply that a model regression was run.

## Troubleshooting

- **Missing `vllm_fl._C`:** reinstall this checkout with `VLLM_VENDOR=cuda` and
  `--no-deps`, then run the full stack check.
- **Wrong vLLM/Triton provider:** restore the pinned image or rebuild its recipe.
  Do not install standalone Triton over FlagTree or use a full native vLLM wheel
  to make an EMPTY check pass.
- **Missing model operator:** confirm FlagGems is enabled and the plugin source,
  dispatch policy and image pins match. Retain the original traceback; do not
  force native-capability flags or disable the model's mathematical operation.
- **Memory failure:** distinguish weight loading, temporary offload copies and
  KV-cache allocation. Reduce request limits or choose a supported topology;
  lowering utilization does not shrink model weights. Do not assume TP16 works
  for a checkpoint validated at TP8.
- **Slow first request:** inspect compiler and worker logs. Retained compilation
  caches can change timings; GPU utilization alone does not identify a hang.
- **Container thread creation fails with `Operation not permitted`:** check the
  host Docker/seccomp compatibility with the image's libc before diagnosing a
  model failure. Do not disable host security or change shared services as a
  deployment workaround.

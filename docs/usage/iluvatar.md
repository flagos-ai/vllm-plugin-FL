# Iluvatar CoreX Deployment Guide

This guide starts an OpenAI-compatible vLLM inference service on Iluvatar CoreX
BI-V150 accelerators with the software stack validated by this repository. The
recommended path uses the prebuilt CI image so that vLLM, FlagTree, Triton, and
FlagGems stay on the tested versions.

## Validated software stack

| Component | Validated version or configuration |
|---|---|
| Prebuilt image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-iluvatar-ci` |
| Image digest | `sha256:eb904d40a92e9769f358f4d81c8043acfbc30b1b077f0a31210be6316ec0cf18` |
| Base image | `harbor.baai.ac.cn/plugin/iluvatar-corex4.5.0-flagtree0.6.0-triton3.6.0-cxnone-vllm_fl0.24.0:20260909` |
| Chip | Iluvatar CoreX BI-V150 |
| vLLM | `0.28.0+empty` (`VLLM_TARGET_DEVICE=empty`, no `vllm._C`) |
| Python | `3.12` (the image ships a `cp312` vLLM wheel) |
| PyTorch | `2.10.0+corex.4.5.0` |
| Driver | `4.5.0` |
| IX-ML | `4.4.0` |
| FlagTree | provides the Triton runtime; the base image tag records `flagtree0.6.0` |
| Triton | `3.6.0` with the `iluvatar` backend; provided by FlagTree, no standalone `triton` package is expected |
| FlagGems | `5.3.5` |
| Dispatch policy | [`vllm_fl/dispatch/config/iluvatar.yaml`](../../vllm_fl/dispatch/config/iluvatar.yaml) |

The Iluvatar dispatch policy is loaded automatically from
[`vllm_fl/dispatch/config/iluvatar.yaml`](../../vllm_fl/dispatch/config/iluvatar.yaml).
It keeps `prefer: flagos` and an empty `flagos_blacklist`, so FlagGems handles the
listed operators with `vendor:iluvatar` and then `reference` as fallbacks. Keep
that file and the plugin source from the same revision; the policy is part of the
validated configuration.

The plugin also applies Iluvatar-specific compilation defaults in
[`vllm_fl/platform.py`](../../vllm_fl/platform.py) whenever
`vendor_name == "iluvatar"`:

- `cudagraph_mm_encoder = false` — the multimodal encoder does not use CUDA Graph
  capture, which is required for correct multi-image concurrency.
- `use_inductor_graph_partition = true` — compiled graph partitioning is enabled,
  which is required for text output to match eager mode.
- `fuse_rope_kvcache_cat_mla = false` — this RoPE/MLA fusion is not loaded on
  CoreX and must stay off while graph partitioning is active.

These three are applied at startup and do not have to be written by hand.

## 1. Check the host

The host needs an installed CoreX driver, the CoreX management tool, and Docker.
Docker lives on the **host**; the adapted container image used for development may
not contain a Docker CLI, so run every Docker command from the host.

```bash
ixsmi
docker version
```

`ixsmi` is the CoreX equivalent of `nvidia-smi`. On CoreX installations that only
expose the compatibility tool, `nvidia-smi` works as a fallback; if neither exists,
probe through PyTorch instead:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Select a tensor-parallel size that matches the model and the number of visible
devices. The Qwen3.6 examples below use four BI-V150 devices with
`TP_SIZE=4`, and the validated configuration requires at least four devices.

## 2. Prepare the repository and model

Use the branch, tag, or commit that you intend to deploy. Keep the plugin
checkout, this guide, and the checked-in Iluvatar dispatch configuration on the
same revision.

```bash
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL

export REPO_DIR="$PWD"
export MODEL_DIR=/data/models/Qwen/Qwen3.6-27B
export MODEL_NAME=qwen3.6-27b
export TP_SIZE=4

test -f "$MODEL_DIR/config.json"
```

The validated models are `Qwen3.6-27B` (dense) and `Qwen3.6-35B-A3B` (MoE). The
image expects the host model directory to be mounted at
`/data/models/Qwen`, so a host directory such as `/mnt/share/user_homes/models`
is mounted to `/data/models/Qwen` and the model is addressed inside the container
as `/data/models/Qwen/Qwen3.6-27B`.

If the model is not already present, it can be downloaded with ModelScope. Keep
weights on the host so that they survive container recreation.

```bash
modelscope download \
  --model Qwen/Qwen3.6-27B \
  --local_dir /data/models/Qwen/Qwen3.6-27B
```

Do not set a network proxy permanently. If a proxy is required for a download,
export it only in the current shell and unset it when the download finishes.

## 3. Pull the validated image

Authenticate first if the Harbor registry requires it. Do not put credentials
in scripts or shell history.

```bash
docker login harbor.baai.ac.cn

export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-iluvatar-ci
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

For a bit-for-bit reproducible deployment, use the manifest digest from the
software-stack table after confirming that the registry still exposes it.

## 4. Verify the image runtime

This check confirms the CoreX stack inside the image before a service is started.
The image was built against a `0.28.0+empty` vLLM wheel, so the build asserts
below are the tested contract; a device must be visible for the accelerator check
to pass.

```bash
docker run --rm \
  --privileged \
  --ipc=host \
  --shm-size=64g \
  -v /dev:/dev \
  -v /lib/modules:/lib/modules \
  -v /usr/local/corex-4.5.0/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so.1 \
  -v /usr/local/corex-4.5.0/lib64/libixml.so:/usr/local/corex/lib64/libixml.so \
  -e GEMS_VENDOR=iluvatar \
  -e VLLM_PLUGINS=fl \
  -e USE_FLAGGEMS=1 \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -euo pipefail
ixsmi || nvidia-smi || true
python3 -c "
import flag_gems
import torch
import triton
import vllm
import vllm_fl

assert vllm.__version__.startswith(\"0.28\"), vllm.__version__
assert \"empty\" in vllm.__version__, vllm.__version__
assert torch.__version__.startswith(\"2.10\"), torch.__version__
assert flag_gems.__version__.startswith(\"5.3\"), flag_gems.__version__
assert triton.__version__.startswith(\"3.6.0\"), triton.__version__
print(f\"vLLM={vllm.__version__}\")
print(f\"PyTorch={torch.__version__}\")
print(f\"Triton={triton.__version__} (FlagTree iluvatar backend)\")
print(f\"FlagGems={flag_gems.__version__}\")
print(f\"vLLM-FL={vllm_fl.__file__}\")
"
python3 -c "
import torch
from vllm.platforms import current_platform

assert torch.cuda.is_available(), \"Iluvatar accelerator is unavailable\"
assert current_platform.vendor_name == \"iluvatar\", current_platform.vendor_name
print(f\"Devices={torch.cuda.device_count()}\")
print(f\"Platform={current_platform}\")
"
'
```

## 5. Start the inference service

The command mounts the current plugin checkout and installs it into the container
without resolving dependencies. This preserves the validated image runtime while
ensuring that the service runs the same plugin revision as the repository
checkout.

```bash
docker run --rm -d \
  --name vllm-fl-iluvatar \
  --privileged \
  --ipc=host \
  --shm-size=64g \
  --hostname vllm-plugin-fl \
  --user root \
  --ulimit nofile=65535:65535 \
  -p 8000:8000 \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v /dev:/dev \
  -v /lib/modules:/lib/modules \
  -v /data:/data \
  -v /mnt/share/user_homes/models:/data/models/Qwen:ro \
  -v /usr/local/corex-4.5.0/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so.1 \
  -v /usr/local/corex-4.5.0/lib64/libixml.so:/usr/local/corex/lib64/libixml.so \
  -e VLLM_PLUGINS=fl \
  -e USE_FLAGGEMS=1 \
  -e GEMS_VENDOR=iluvatar \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e MODEL=/data/models/Qwen/Qwen3.6-27B \
  -e MODEL_NAME=qwen3.6-27b \
  -e TP_SIZE=4 \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -euo pipefail
cd /workspace/vllm-plugin-FL
python3 -m pip install \
  --no-build-isolation \
  --no-deps \
  -e .
exec vllm serve "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.70 \
  --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64,128,256]}" \
  --trust-remote-code
'
```

Notes on this command:

- CoreX devices are exposed through `--privileged`, `-v /dev:/dev`,
  `-v /lib/modules:/lib/modules`, the two CoreX library mounts, and
  `CUDA_VISIBLE_DEVICES`. `--gpus all` belongs to the NVIDIA Container Toolkit
  and is not used here.
- `--compilation-config` must be passed explicitly with a short
  `cudagraph_capture_sizes` list. Leaving it out keeps `PIECEWISE` as the
  upstream default but keeps the upstream capture list of roughly 51 sizes,
  which runs out of memory during graph warm-up.
- `--gpu-memory-utilization 0.70` is the validated value. Higher values make the
  graph warm-up allocation fail.
- Do **not** set `USE_FLAGGEMS=0`. Doing so disables the validated FlagGems
  operator path and is not a supported configuration.
- Do **not** set `VLLM_FL_PREFER=vendor`. The validated policy is the checked-in
  `prefer: flagos` dispatch order.
- Do not modify `is_cuda_alike` or the graph `splitting_ops` to force CUDA code
  paths. CoreX reports `is_cuda_alike() == False` by design, and the validated
  configuration already accounts for it.

The Iluvatar defaults described in the software-stack section are injected by the
plugin, so `cudagraph_mm_encoder`, `use_inductor_graph_partition`, and
`fuse_rope_kvcache_cat_mla` normally do not need to be passed on the command line.

## 6. Verify inference

Follow startup logs in one terminal:

```bash
docker logs -f vllm-fl-iluvatar
```

Wait for the OpenAI-compatible endpoint in another terminal:

```bash
for attempt in $(seq 1 120); do
  if curl --silent --fail http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "service is ready"
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' vllm-fl-iluvatar 2>/dev/null \
      | grep -q true; then
    docker logs vllm-fl-iluvatar
    exit 1
  fi
  if [ "$attempt" -eq 120 ]; then
    echo "service did not become ready" >&2
    docker logs vllm-fl-iluvatar
    exit 1
  fi
  sleep 5
done
```

Send a deterministic chat request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "Reply with exactly: vLLM FL is ready"}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

The deployment is successful when `/v1/models` returns HTTP 200 and the chat
request returns a non-empty `choices[0].message.content` without a server-side
error.

For the full acceptance check, use the repository's official Qwen3.6 test cases
and read their JSON result rather than relaxing the expected terms. A text run
that drifts a few tokens away from the reference wording indicates a compilation
misconfiguration, not a tolerant test.

Stop the example service with:

```bash
docker stop vllm-fl-iluvatar
```

## 7. Build the same stack from the base image

Use this path only when the prebuilt Harbor image is unavailable. The base image
already provides the CoreX 4.5 stack with FlagTree and Triton, so no separate
`triton` install is required. The build needs the `0.28.0+empty` vLLM wheel in
the build context.

```bash
mkdir -p docker/iluvatar/wheels
cp /path/to/vllm-0.28.0+empty-*.whl docker/iluvatar/wheels/

docker build \
  --target ci \
  -t my-registry/vllm-plugin-fl:v0.28.0-iluvatar-ci \
  -f docker/iluvatar/Dockerfile docker/iluvatar
```

The `ci` stage asserts the tested contract while building: vLLM `0.28` with
`empty` in the version, PyTorch `2.10`, and FlagGems `5.3`. Run the runtime
verification from step 4 before starting a service. Moving to a new FlagTree,
FlagGems, or CoreX revision requires re-running the operator tests and real dense
and MoE model inference before calling the environment supported.

## Troubleshooting

- **Out of memory during graph warm-up:** lower `--gpu-memory-utilization`
  (`0.70` is validated), and keep the explicit `--compilation-config` with the
  short `cudagraph_capture_sizes` list.
- **`NameError: MLARoPE...` at startup:** graph partitioning is active while the
  RoPE/MLA fusion is not disabled. Confirm the plugin revision sets
  `fuse_rope_kvcache_cat_mla = false` for Iluvatar, and do not enable that fusion
  manually.
- **Multi-image requests return wrong content or ordering:** multimodal encoder
  CUDA Graph capture is still enabled. Confirm the plugin revision sets
  `cudagraph_mm_encoder = false` for Iluvatar.
- **Text output diverges from eager mode a few tokens in:** compiled graph
  partitioning is not active. Confirm the plugin revision sets
  `use_inductor_graph_partition = true` for Iluvatar.
- **Plugin is not active:** confirm the container has `VLLM_PLUGINS=fl` and that
  `current_platform.vendor_name` reports `iluvatar`.
- **FlagGems is not active:** confirm `USE_FLAGGEMS=1` and
  `GEMS_VENDOR=iluvatar`. Do not replace the checked-in dispatch policy with an
  ad-hoc whitelist, and do not disable FlagGems to work around an operator
  failure.
- **`docker` is not found inside a container:** run Docker commands from the
  host. The adapted development container does not ship a Docker CLI.
- **A model-specific operator fails:** retain the checked-in Iluvatar policy,
  reproduce the failure against native PyTorch/vLLM, and only then update
  [`iluvatar.yaml`](../../vllm_fl/dispatch/config/iluvatar.yaml).

# NVIDIA Deployment Guide

This guide starts an OpenAI-compatible vLLM inference service on NVIDIA GPUs
with the software stack validated by this repository. The recommended path uses
the prebuilt CI image so that vLLM, FlagTree, Triton, and FlagGems stay on the
tested versions.

## Validated software stack

| Component | Validated version or configuration |
|---|---|
| Prebuilt image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-cuda-ci` |
| Image manifest | `sha256:ad7a8dfc9f760d9878b6634a4009aefe98afd58e179f01c62333e935ee130e6f` |
| Base image | `vllm/vllm-openai:v0.28.0-cu129` |
| vLLM | `0.28.0` |
| Python | `3.12.3` |
| PyTorch | `2.13.0+cu129` |
| CUDA runtime | `12.9` |
| FlagTree | `0.6.2a1` |
| Triton | `3.6.0`, provided by FlagTree; no standalone `triton` distribution |
| FlagGems | `5.3.5+g34c6d2ce4` at commit `34c6d2ce416d32a98f1dd2bb6e4cfb67c59342d9` |

The NVIDIA dispatch policy is loaded automatically from
[`vllm_fl/dispatch/config/nvidia.yaml`](../../vllm_fl/dispatch/config/nvidia.yaml).
Keep that file and the plugin source from the same revision. Its blacklist is
part of the validated configuration and must not be replaced with an ad-hoc
whitelist.

## 1. Check the host

The host needs Docker, NVIDIA Container Toolkit, and an NVIDIA driver that can
run CUDA 12.9 containers.

```bash
nvidia-smi
docker version
docker info | grep -i runtime
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

Select a tensor-parallel size that matches the model and the number of visible
GPUs. The Qwen3-8B example below fits on one 80 GB GPU with `TP_SIZE=1`.

## 2. Prepare the repository and model

Use the branch, tag, or commit that you intend to deploy. Keep the plugin
checkout, this guide, and the checked-in NVIDIA dispatch configuration on the
same revision.

```bash
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL

export REPO_DIR="$PWD"
export MODEL_DIR=/data/models/Qwen3-8B
export TP_SIZE=1

test -f "$MODEL_DIR/config.json"
```

If the model is not already present, it can be downloaded with ModelScope. Keep
weights outside the container so that they survive container recreation.

```bash
modelscope download \
  --model Qwen/Qwen3-8B \
  --local_dir /data/models/Qwen3-8B
```

Do not set a network proxy permanently. If a proxy is required for a download,
export it only in the current shell and unset it when the download finishes.

## 3. Pull the validated image

Authenticate first if the Harbor registry requires it. Do not put credentials
in scripts or shell history.

```bash
docker login harbor.baai.ac.cn

export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-cuda-ci
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

For a bit-for-bit reproducible deployment, use the manifest digest from the
software-stack table after confirming that the registry still exposes it.

## 4. Verify the image runtime

This check confirms that Triton comes from FlagTree. A separately installed
`triton` wheel would replace the tested runtime and is treated as an error.

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -euo pipefail
nvidia-smi
if python3 -m pip show triton >/dev/null 2>&1; then
  echo "ERROR: standalone triton is installed; FlagTree must provide Triton" >&2
  exit 1
fi
python3 -c "
from importlib.metadata import version
import flag_gems
import torch
import triton
import vllm

assert vllm.__version__.startswith(\"0.28.0\"), vllm.__version__
assert version(\"flagtree\") == \"0.6.2a1\", version(\"flagtree\")
assert triton.__version__.startswith(\"3.6.0\"), triton.__version__
flagtree_version = version(\"flagtree\")
print(f\"vLLM={vllm.__version__}\")
print(f\"PyTorch={torch.__version__}, CUDA={torch.version.cuda}\")
print(f\"FlagTree={flagtree_version}\")
print(f\"Triton={triton.__version__} (provided by FlagTree)\")
print(f\"FlagGems={flag_gems.__version__}\")
"
'
```

## 5. Start the inference service

The command mounts the current plugin checkout and installs it into the
container without resolving dependencies. This preserves the validated image
runtime while ensuring that the service runs the same plugin revision as the
repository checkout.

```bash
docker run --rm -d \
  --name vllm-fl-nvidia \
  --privileged \
  --gpus all \
  --ipc=host \
  --hostname vllm-plugin-fl \
  --user root \
  --ulimit nofile=65535:65535 \
  -p 8000:8000 \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v "$MODEL_DIR:/models/Qwen3-8B:ro" \
  -e VLLM_PLUGINS=fl \
  -e USE_FLAGGEMS=1 \
  -e GEMS_VENDOR=nvidia \
  -e USE_FLAGTUNE=0 \
  -e TP_SIZE="$TP_SIZE" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -euo pipefail
cd /workspace/vllm-plugin-FL
uv pip install \
  --system \
  --break-system-packages \
  --no-build-isolation \
  --no-deps \
  -e .
exec vllm serve /models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
'
```

The default vLLM 0.28 runner and CUDA Graph mode are used. Add
`--enforce-eager` only when an eager-mode run is specifically required.

The validated NVIDIA policy intentionally keeps vLLM's native CUDA attention
backend. Do not set `VLLM_FL_USE_FLAGGEMS_ATTN=1` unless that path has been
validated separately for the target model.

## 6. Verify inference

Follow startup logs in one terminal:

```bash
docker logs -f vllm-fl-nvidia
```

Wait for the OpenAI-compatible endpoint in another terminal:

```bash
for attempt in $(seq 1 120); do
  if curl --silent --fail http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "service is ready"
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' vllm-fl-nvidia 2>/dev/null \
      | grep -q true; then
    docker logs vllm-fl-nvidia
    exit 1
  fi
  if [ "$attempt" -eq 120 ]; then
    echo "service did not become ready" >&2
    docker logs vllm-fl-nvidia
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
    "model": "qwen3-8b",
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

Stop the example service with:

```bash
docker stop vllm-fl-nvidia
```

## 7. Build the same stack from the official base image

Use this path only when the prebuilt Harbor image is unavailable. Start from
`vllm/vllm-openai:v0.28.0-cu129`, then replace vLLM's standalone Triton package
with the validated FlagTree runtime. Do not install `triton==3.6.0` separately.

```bash
python3 -m pip uninstall -y triton
# Repeat until: WARNING: Skipping triton as it is not installed.

RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.6.2a1 $RES

python3 -m pip install -U scikit-build-core==0.11 pybind11 ninja cmake

git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
git checkout 34c6d2ce416d32a98f1dd2bb6e4cfb67c59342d9
python3 -m pip install --no-build-isolation --no-deps .

cd /workspace/vllm-plugin-FL
VLLM_VENDOR=cuda python3 -m pip install \
  --no-build-isolation \
  --no-deps \
  -e .
```

Run the runtime verification from step 4 before starting a service. Moving to a
new FlagTree or FlagGems revision requires re-running operator-blacklist tests
and real dense and MoE model inference before calling the environment
supported.

## Troubleshooting

- **Plugin is not active:** confirm the container has `VLLM_PLUGINS=fl` and the
  startup log reports the FL platform/plugin registration.
- **FlagGems is not active:** confirm `USE_FLAGGEMS=1` and
  `GEMS_VENDOR=nvidia`; do not define both a FlagGems whitelist and blacklist.
- **Triton import or compilation errors:** run step 4. The supported setup has
  FlagTree `0.6.2a1`, imported Triton `3.6.0`, and no standalone `triton`
  package.
- **Out of memory:** lower `--max-model-len` or
  `--gpu-memory-utilization`, or increase `TP_SIZE` when the model supports that
  tensor-parallel degree.
- **First request is slow:** Triton kernels and CUDA Graphs may compile or
  capture during warm-up. Check container logs before treating this as a
  failure.
- **A model-specific operator fails:** retain the checked-in NVIDIA blacklist,
  reproduce the failure against native PyTorch/vLLM, and only then update
  [`nvidia.yaml`](../../vllm_fl/dispatch/config/nvidia.yaml).

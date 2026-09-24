# Ascend 910C Deployment Guide

This guide starts an OpenAI-compatible vLLM inference service on Huawei
Ascend 910C. It follows the NVIDIA deployment guide's prebuilt-image workflow,
using the Ascend stack validated by this repository. The example uses one 910C
and Qwen3-0.6B; choose other free device IDs and a larger tensor-parallel size
for larger models.

## Validated software stack

| Component | Validated version or configuration |
|---|---|
| Prebuilt CI image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:ascend-vllm0.28.0-a3-ci-20260922-r5-gas-clean` |
| Image manifest | `sha256:d80f4616b04fda507ce6d9dc54d36983d4ef134730bb9305b45409931260d688` |
| Base image | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` |
| CANN | `9.0.0` |
| Python | `3.11.15` on aarch64 |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0` |
| vLLM | `0.28.0+empty` |
| FlagTree / Triton | `0.6.2a1+ascend3.5` / `3.5.1` (Triton supplied by FlagTree) |
| FlagGems | commit `3b406c36212744b98b9720bf6d0a5387c09fe96b` |
| cann-shmem / NumPy | `1.6.0` / `1.26.4` |

The checked-in [Ascend dispatch policy](../../vllm_fl/dispatch/config/ascend.yaml)
loads automatically. Keep it and the plugin checkout on the same revision.
The CI image contains the validated runtime, but its packaged plugin wheel
predates some changes in this guide. Step 5 installs your checkout over that
wheel without changing the runtime dependencies. For build details and the
full 910C validation matrix, see the [Ascend image guide](../../docker/ascend/README.md).

## 1. Check the host

The host needs a working CANN driver, Docker, and an Ascend container runtime.
The commands below use a runtime registered as `ascend`, as on the validated
910C host. Confirm its name and select free physical NPU IDs before starting:

```bash
npu-smi info
docker version
docker info --format '{{json .Runtimes}}'
test -e /dev/davinci_manager
test -f /etc/ascend_install.info
```

Set the visible devices explicitly. `0` is an example; the shared 910C CI
runner uses `14,15`. For tensor parallelism, the number of selected devices
must match `TP_SIZE`.

```bash
export NPU_IDS=0
export TP_SIZE=1
```

The example uses a privileged task container, matching the working 910C
configuration. If your host uses different driver or runtime paths, adjust
the mounts in steps 4 and 5 to match its Ascend installation.

## 2. Prepare the repository and model

Use the branch, tag, or commit that contains the plugin version you intend to
deploy. Keep the checkout, this guide, and the Ascend dispatch policy on that
same revision. Provision model weights on the host outside the container:

```bash
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL

export REPO_DIR="$PWD"
export MODEL_DIR=/data/models/Qwen3-0.6B
test -f "$MODEL_DIR/config.json"
```

The single-NPU example below follows the validated Qwen3-0.6B eager settings.
The shared Ascend CI also validates Qwen3.6-27B and Qwen3.6-35B-A3B on two
910C NPUs for offline and serving E2E. Those models need their own weights,
`TP_SIZE=2`, and the memory/length settings in
[`tests/platforms/ascend.yaml`](../../tests/platforms/ascend.yaml).

## 3. Pull the validated image

Authenticate interactively if Harbor requires it; keep registry credentials
out of scripts and shell history.

```bash
docker login harbor.baai.ac.cn

export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:ascend-vllm0.28.0-a3-ci-20260922-r5-gas-clean
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

For an exact reproduction, pull the image by the manifest digest in the table
after confirming the registry still exposes it. The tag may be republished
independently of this repository.

## 4. Verify the image runtime

This checks NPU access and the pinned empty-vLLM stack before loading a model.
FlagTree must own the `triton` module; a separately installed `triton` or
`triton-ascend` distribution is not part of the validated image.

```bash
docker run --rm \
  --runtime=ascend --privileged --network=host --ipc=host \
  -e ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --entrypoint /bin/bash "$IMAGE" -lc '
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python - <<PY
from importlib.metadata import PackageNotFoundError, version
import triton

expected = {
    "vllm": "0.28.0+empty",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "flagtree": "0.6.2a1+ascend3.5",
    "cann-shmem": "1.6.0",
    "numpy": "1.26.4",
}
for name, pinned in expected.items():
    actual = version(name)
    assert actual == pinned, (name, actual, pinned)
    print(f"{name}={actual}")
assert triton.__version__.startswith("3.5.1"), triton.__version__
for name in ("triton", "triton-ascend"):
    try:
        installed = version(name)
    except PackageNotFoundError:
        continue
    raise RuntimeError(f"Remove standalone {name}=={installed}; use FlagTree")
print(f"Triton={triton.__version__} (from FlagTree)")
PY
'
```

## 5. Start the inference service

The command installs the current plugin checkout in editable mode without
resolving dependencies. This is how the shared CI tests pull-request source
against the prepared image; it also ensures that the service uses your checkout
instead of the older wheel inside this published tag.

```bash
docker run --rm -d \
  --name vllm-fl-ascend \
  --runtime=ascend --privileged --network=host --ipc=host \
  --hostname vllm-plugin-fl \
  --ulimit nofile=65535:65535 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v "$MODEL_DIR:/models/Qwen3-0.6B:ro" \
  -e VLLM_PLUGINS=fl \
  -e VLLM_FL_PLATFORM=ascend \
  -e ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS" \
  -e GLOO_SOCKET_IFNAME=lo \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e TP_SIZE="$TP_SIZE" \
  --entrypoint /bin/bash "$IMAGE" -lc '
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/vllm-plugin-FL
python -m pip install --no-build-isolation --no-deps -e .
exec vllm serve /models/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --no-enable-chunked-prefill \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --trust-remote-code
'
```

`--network=host` makes the service's `127.0.0.1:8000` endpoint available on
the host loopback interface. For remote clients, choose an appropriate bind
address and network policy. `GLOO_SOCKET_IFNAME=lo` is for a single-host job;
multi-host jobs need an interface reachable by every rank. Eager execution is
the supported deployment setting here. NPU graph execution is experimental
and is outside this guide's startup recipe.

## 6. Verify inference

Follow startup logs in one terminal:

```bash
docker logs -f vllm-fl-ascend
```

Wait for the endpoint in another terminal:

```bash
for attempt in $(seq 1 120); do
  if curl --silent --fail http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "service is ready"
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' vllm-fl-ascend 2>/dev/null \
      | grep -q true; then
    docker logs vllm-fl-ascend
    exit 1
  fi
  if [ "$attempt" -eq 120 ]; then
    echo "service did not become ready" >&2
    docker logs vllm-fl-ascend
    exit 1
  fi
  sleep 5
done
```

Send a deterministic completion request:

```bash
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "prompt": "The capital of France is",
    "temperature": 0,
    "max_tokens": 16
  }'
```

The deployment is working when `/v1/models` returns HTTP 200 and the
completion has non-empty `choices[0].text` without a server-side error.
Stop this example with `docker stop vllm-fl-ascend`.

## 7. Optional: enable FlagCX device collectives

The published image does **not** bundle FlagCX. The separately validated
FlagCX v0.13.0 configuration mounted a built FlagCX core library and set
`FLAGCX_PATH`. It used HCCL for PyTorch process groups and FlagCX's C API for
tensor-parallel device collectives; the FlagCX Torch plugin was not installed.
On 910C_174, Qwen3-0.6B and Qwen3.6-27B TP2 eager inference and two-rank
`all_reduce`, `all_gather`, `all_gatherv`, and `reduce_scatterv` checks passed.
Pipeline and expert parallelism were not covered by that FlagCX test.

Build the v0.13.0 core for Ascend in an environment with CANN development
headers, following the [FlagCX build instructions](https://github.com/flagos-ai/FlagCX/blob/v0.13.0/docs/getting_started.md).
The validated build used `make USE_ASCEND=1`; ensure its development
dependencies, including `nlohmann/json.hpp`, are present. Then verify:

```bash
export FLAGCX_DIR=/path/to/FlagCX
test -f "$FLAGCX_DIR/build/lib/libflagcx.so"
```

For a TP2 run, select two free NPUs and set `TP_SIZE=2`. Add these options to
the `docker run` command in step 5, before `--entrypoint`:

```bash
-v "$FLAGCX_DIR:/opt/FlagCX:ro" \
-e FLAGCX_PATH=/opt/FlagCX \
```

Keep the same plugin checkout inside the container. Do not install the FlagCX
Torch plugin to reproduce this tested NPU path; `FLAGCX_PATH` is enough to
select the plugin's FlagCX device communicator.

## 8. Build the same stack from the official base image

When the prebuilt Harbor image is unavailable, build the checked-in Ascend
Dockerfile on an aarch64 machine. It starts from the base image in the table,
removes the inherited vLLM 0.20.2 installation, and installs upstream vLLM
0.28.0 with `VLLM_TARGET_DEVICE=empty` plus the pinned FlagTree and FlagGems
stack. Give the local build an explicit tag:

```bash
docker/build.sh \
  --platform ascend \
  --target ci \
  --image-name vllm-plugin-fl \
  --image-tag ascend-vllm0.28.0-local

export IMAGE=vllm-plugin-fl:ascend-vllm0.28.0-local
```

Run step 4 before serving. The Dockerfile prepares the runtime and CI tools
but does not install this repository's plugin wheel. Keep the step 5 checkout
overlay, or build a separate release image that installs the plugin
non-editably from a fixed revision. The
[image guide](../../docker/ascend/README.md) documents build constraints and
the known OpenCV/NumPy dependency conflict in this pinned stack.

## Troubleshooting

- **No NPU inside the container:** inspect `npu-smi info`,
  `ASCEND_RT_VISIBLE_DEVICES`, the Ascend runtime, driver mounts, and device
  access. The working 910C task container uses `--privileged`.
- **Wrong vLLM or Triton version:** rerun step 4. Non-NVIDIA devices require
  `vllm==0.28.0+empty`; FlagTree supplies Triton 3.5.1 without a standalone
  `triton` or `triton-ascend` distribution.
- **Plugin not active:** check `VLLM_PLUGINS=fl`, `VLLM_FL_PLATFORM=ascend`,
  the editable checkout install, and the platform activation in container logs.
- **Out of memory:** choose free NPUs, lower `--max-model-len` or
  `--gpu-memory-utilization`, or use a supported tensor-parallel size.
- **First request is slow:** FlagTree kernels may JIT compile during warm-up.
  Check logs before treating that delay as a failure.
- **FlagCX initialization fails:** verify `FLAGCX_PATH` points to the mounted
  v0.13.0 source tree containing `build/lib/libflagcx.so`. Leave the variable
  unset when using the default HCCL-only path.

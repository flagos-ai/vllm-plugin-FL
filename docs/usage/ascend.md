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
| Plugin source and dispatch policy | `86d32aaac3dc568014e9734ea17596ce17a2b784` |
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
predates some changes in this guide. Step 5 builds and installs a non-editable
wheel from the selected plugin revision without changing runtime dependencies.
For build details, see the
[Ascend image guide](../../docker/ascend/README.md).

### FlagGems execution mode

The validated image uses FlagGems's Python/Triton operators with
`USE_C_EXTENSION=0` and `flag_gems.config.use_c_extension == False`. The
installation in the Ascend Dockerfile needs no C++ extension:

```bash
python -m pip install --no-build-isolation --no-deps /workspace/FlagGems
```

`use_c_extension` is a FlagGems runtime flag. Enabling it requires building
the optional C++ extension for NPU and setting `USE_C_EXTENSION=1` before
importing FlagGems. See the pinned
[FlagGems source-installation instructions](https://github.com/flagos-ai/FlagGems/blob/3b406c36212744b98b9720bf6d0a5387c09fe96b/docs/content/en/getting-started/install.md#331-install-with-c-extension)
for the build dependencies and the `FLAGGEMS_BACKEND=NPU` and
`FLAGGEMS_BUILD_C_EXTENSIONS=ON` options. That optional mode has not been
validated by this deployment recipe; keep `USE_C_EXTENSION=0` to reproduce
the tested stack.

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

Set the visible devices explicitly. `0` is an example; use the devices
assigned on your host. For tensor parallelism, the number of selected devices
must match `TP_SIZE`.

```bash
export NPU_IDS=0
export TP_SIZE=1
```

The example uses a privileged task container, matching the working 910C
configuration. If your host uses different driver or runtime paths, adjust
the mounts in steps 4 and 5 to match its Ascend installation.

## 2. Prepare the repository and model

The executable example pins the reviewed PR #487 runtime commit. Fetch the PR
ref before checking out that immutable revision; its checked-in Ascend policy
is the policy paired with the image runtime above. Provision model weights on
the host outside the container:

```bash
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL
git fetch origin refs/pull/487/head
export PLUGIN_REVISION=86d32aaac3dc568014e9734ea17596ce17a2b784
git checkout --detach "$PLUGIN_REVISION"
test "$(git rev-parse HEAD)" = "$PLUGIN_REVISION"
test -f vllm_fl/dispatch/config/ascend.yaml

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

export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:d80f4616b04fda507ce6d9dc54d36983d4ef134730bb9305b45409931260d688
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
  -e USE_C_EXTENSION=0 \
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
import flag_gems
from flag_gems.config import use_c_extension

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
assert flag_gems.vendor_name == "ascend", flag_gems.vendor_name
assert not use_c_extension, "Use USE_C_EXTENSION=0 for the validated Ascend stack"
print("FlagGems=Ascend Python/Triton operators (C++ extension disabled)")
PY
'
```

## 5. Start the inference service

The command builds a wheel from the selected plugin revision and installs it
non-editably without resolving dependencies. It checks that the installed
module and Ascend dispatch policy come from site-packages, rather than the
mounted checkout. Shared CI uses an editable checkout for pull-request tests;
this deployment path validates the package that will be delivered.

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
  -e USE_C_EXTENSION=0 \
  -e PLUGIN_REVISION="$PLUGIN_REVISION" \
  -e ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS" \
  -e GLOO_SOCKET_IFNAME=lo \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e TP_SIZE="$TP_SIZE" \
  --entrypoint /bin/bash "$IMAGE" -lc '
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/vllm-plugin-FL
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir /tmp/vllm-fl-wheels .
python -m pip install --no-deps --force-reinstall \
  /tmp/vllm-fl-wheels/vllm_plugin_fl-*.whl
cd /tmp
python - <<PY
from importlib.resources import files
from pathlib import Path
import os
import vllm_fl
from vllm_fl.version import git_version

module_path = Path(vllm_fl.__file__).resolve()
assert "site-packages" in str(module_path), module_path
policy = files("vllm_fl.dispatch.config").joinpath("ascend.yaml")
assert policy.read_bytes() == Path(
    "/workspace/vllm-plugin-FL/vllm_fl/dispatch/config/ascend.yaml"
).read_bytes()
assert git_version != "Unknown" and os.environ["PLUGIN_REVISION"].startswith(git_version)
print(f"Installed plugin: {module_path}")
PY
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
The validated FlagCX setup covers TP2 eager inference and two-rank
`all_reduce`, `all_gather`, `all_gatherv`, and `reduce_scatterv`. Pipeline
and expert parallelism have not been validated.

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
but does not install this repository's current plugin wheel. Use the step 5
wheel build, or publish a separate release image with that fixed wheel. The
[image guide](../../docker/ascend/README.md) documents build constraints and
the known OpenCV/NumPy dependency conflict in this pinned stack.

## Troubleshooting

- **No NPU inside the container:** inspect `npu-smi info`,
  `ASCEND_RT_VISIBLE_DEVICES`, the Ascend runtime, driver mounts, and device
  access. The validated task container uses `--privileged`.
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

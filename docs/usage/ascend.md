# Ascend 910C Deployment Guide

Run an OpenAI-compatible service with vLLM 0.32's empty build and the FL
plugin. Native `torch.npu.NPUGraph` captures decode; prefill, vision encoding
and sampling execute outside the decode graph.

## Software dependencies

Use the pinned [Ascend Dockerfile](../../docker/ascend/Dockerfile) and
[runtime constraints](../../docker/ascend/runtime-constraints.txt):

| Component | Version / source |
|---|---|
| Base image | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` |
| CANN | 9.0.0 |
| Python | 3.11, aarch64 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0 |
| vLLM | 0.32.0+empty, official `v0.32.0` source |
| FlagTree / Triton | 0.6.2a1+ascend3.5 / 3.5.1, supplied by FlagTree |
| FlagGems | 5.4.0, `6ed2f39071ef85db26b7e64f0652a162f434959c` |
| FlagGems-vllm | v0.2.0, installed by the repository's pinned installer |
| NumPy / PyYAML | 2.2.6 / 6.0.1 |
| cann-shmem | 1.6.0 |

Retain the vendor Torch/torch-npu pair when installing vLLM's common Python
dependencies. Disable build isolation for vLLM and the plugin. FlagTree owns
the `triton` module: remove standalone `triton` and `triton-ascend` distributions
before installing FlagTree. Use FlagGems's Python/Triton mode (`USE_C_EXTENSION=0`).
Keep the [Ascend dispatch policy](../../vllm_fl/dispatch/config/ascend.yaml)
and plugin checkout on the same revision.

## Build the runtime image

On an aarch64 host with Docker, check out the desired plugin revision, then
build from the repository root:

```bash
docker/build.sh --platform ascend --target ci \
  --image-name vllm-plugin-fl --image-tag ascend-vllm0.32.0
export IMAGE=vllm-plugin-fl:ascend-vllm0.32.0
export REPO_DIR="$PWD"
export MODEL_DIR=/path/to/models
```

The Dockerfile prepares runtime and test dependencies; install the plugin and
FlagGems-vllm in the container as shown below. For image stages and shared CI
configuration, see the [image guide](../../docker/ascend/README.md).

## Create a container

The host needs a compatible Ascend driver and container runtime. Confirm the
registered runtime name, inspect device utilization, and select two allocated
NPUs for the TP2 recipe. Adjust driver mounts to the host installation.

```bash
npu-smi info
docker info --format '{{json .Runtimes}}'
export NPU_IDS=0,1  # Replace with the physical devices assigned to the job.
test -f "$MODEL_DIR/Qwen3.6-27B/config.json"

docker run --detach --name vllm-fl-ascend \
  --runtime=ascend --network=host --ipc=host \
  -e ASCEND_VISIBLE_DEVICES="$NPU_IDS" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$REPO_DIR:/workspace/vllm-plugin-FL" \
  -v "$MODEL_DIR:/models:ro" \
  --entrypoint /bin/bash "$IMAGE" -lc 'exec sleep infinity'
docker exec -it vllm-fl-ascend bash
```

Run the remaining commands inside this container. Physical device selection is
handled by the container runtime; the two visible devices have logical indices
`0,1`. For a single-host service, `GLOO_SOCKET_IFNAME=lo` avoids hostname
resolution delays. Multi-host communication needs a reachable network interface.

## Install the plugin

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/vllm-plugin-FL
python .github/scripts/install_flaggems_vllm.py
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir /tmp/vllm-fl-wheels .
cd /tmp
python -m pip install --no-deps --force-reinstall \
  /tmp/vllm-fl-wheels/vllm_plugin_fl-*.whl
python -c 'import torch; import vllm_fl; print(vllm_fl.__file__)'
```

The imported plugin should resolve to the installed package. Shared CI instead
uses an editable install to test the submitted checkout.

## Native decode graph mode

This recipe covers BF16 Qwen3.6-27B and Qwen3.6-35B-A3B with TP2 and DP1.
Async scheduling, chunked prefill, prefix caching, expert parallelism and
speculative decoding are outside this configuration. Ascend uses `ModelRunnerFL`
(`VLLM_USE_V2_MODEL_RUNNER=0`); the upstream V2 runner requires CUDA/UVA APIs.

```bash
export VLLM_PLUGINS=fl VLLM_FL_PLATFORM=ascend USE_C_EXTENSION=0
export USE_FLAGGEMS=1 VLLM_FL_FLAGOS_WHITELIST=zeros
export TRITON_ALL_BLOCKS_PARALLEL=1 TORCHINDUCTOR_COMPILE_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=lo
unset VLLM_FL_FLAGOS_BLACKLIST FLAGCX_PATH

vllm serve /models/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.80 --trust-remote-code \
  --allowed-local-media-path /workspace/vllm-plugin-FL/tools/adaptation-gate-cases/images \
  --no-enable-chunked-prefill --no-async-scheduling --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

Substitute `Qwen3.6-35B-A3B` in both the model path and served name for the MoE
model. Native HCCL is selected with `FLAGCX_PATH` unset. This configuration
allows only FlagGems `zeros` ATen replacement; unrestricted FlagGems replacement
is outside its correctness guarantee. A nonempty `VLLM_FL_FLAGOS_WHITELIST` replaces platform selection, including
its blacklist and `VLLM_FL_FLAGOS_BLACKLIST_APPEND`. Do not set
`VLLM_FL_FLAGOS_BLACKLIST` at the same time; the two overrides conflict.

Attention uses native BLHNC packed caches and 128-token attention pages; Mamba
states retain their allocator-owned storage and strides. GDN state copy plans
select only active rows. Padded FIA lengths and MoE routes retain graph shapes.
The Ascend dispatch policy propagates execution failures without retrying an
operator after it may have updated state. Graph failures propagate to the caller.

For an eager control, replace `--compilation-config` with `--enforce-eager`.
The host loopback endpoint is accessible with `--network=host`; choose a bind
address and network policy appropriate for remote clients.

## Verify inference

After startup, query the service and run the existing adaptation gate:

```bash
curl --fail http://127.0.0.1:8000/v1/models
cd /workspace/vllm-plugin-FL/tools/adaptation-gate-cases
MODEL_PATH=/models/Qwen3.6-27B SERVED_MODEL_NAME=Qwen3.6-27B PORT=8000 \
  ./run_test.sh
```

Follow the [gate README](../../tools/adaptation-gate-cases/README.md) for its
fixtures, request budgets and checks. Repeat for each model and eager/graph
configuration. With `VLLM_LOGGING_LEVEL=DEBUG`, verify completed capture and
actual request batches with `cudagraph_mode: FULL` at padded sizes 1, 2, 4 and 8.
Startup alone does not demonstrate replay. Record results in the PR or CI
artifacts for the exact tested revision.

Run regression suites using the same interpreter and operator policy:

```bash
python tests/run.py --platform ascend --device 910c --scope unit
python tests/run.py --platform ascend --device 910c --scope functional
```

## Troubleshooting

- **No NPU:** check the registered container runtime, driver mounts, assigned
  device IDs and permissions with `npu-smi info` inside the container.
- **Dependency mismatch:** compare installed versions to the Dockerfile and
  runtime constraints; a vLLM 0.28 image cannot run this 0.32 adapter.
- **Plugin inactive:** check `VLLM_PLUGINS=fl`, `VLLM_FL_PLATFORM=ascend` and
  the imported package path.
- **Out of memory:** inspect free device memory and adjust context length,
  scheduling limits or memory utilization together. Graph capture reserves
  memory in addition to the model and cache.
- **First request is slow:** inspect compilation and warmup logs before
  treating the delay as a failure.

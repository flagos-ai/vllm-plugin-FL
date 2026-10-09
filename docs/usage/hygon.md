# Hygon Deployment Guide

This guide starts an OpenAI-compatible vLLM service on Hygon DCUs. Use the
prebuilt image to keep the vendor DTK/PyTorch, FlagTree, FlagGems, and plugin
on compatible versions. vLLM, FlagGems, and vLLM-Plugin-FL are ordinary
installed packages; serving does not require a source checkout or editable install.

## Validated software stack

| Component | Validated version or configuration |
|---|---|
| Deployment image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-hygon-ci` |
| Image manifest | Pending publication verification |
| Vendor base | DTK 26.04 / Ubuntu 22.04 / Python 3.10.12 |
| vLLM distribution | `0.28.0+empty`, upstream commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| vLLM runtime version | `0.28.0` |
| Vendor PyTorch distribution | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| PyTorch runtime version | `2.10.0` |
| FlagGems distribution | `0.1.0` |
| FlagTree | `0.6.0+hcu3.6` |
| Triton distribution | `3.6.0+gitc73250c4.staging`, coexists with FlagTree in the vendor baseline |
| Validated plugin code | `04ae0f0906e97d11a4d1f5a9d4a01e87487416d5` |
| Communication | Vendor HIP/RCCL, TP2; FlagCX was not installed |
| Hardware | Two Hygon BW1000 DCUs, 64 GiB each |

The vendor PyTorch version is intentional. The empty vLLM wheel is built
without resolving upstream PyTorch dependencies, which would replace this
vendor stack. Keep the image's existing FlagTree/Triton combination; do not
replace it with an unrelated Triton wheel. The empty wheel does not include `vllm._C` or
`vllm._C_stable`; this deployment uses the FL backend and does not copy native
extensions from an older vLLM installation.

The Hygon dispatch policy is loaded from
[`vllm_fl/dispatch/config/hygon.yaml`](../../vllm_fl/dispatch/config/hygon.yaml).
Keep the image's policy and operator fixes together. The tested configuration
uses `USE_FLAGGEMS=1`, `GEMS_VENDOR=hygon`, and `VLLM_PLUGINS=fl`, with the
repository's existing blacklist.

## 1. Check the host and select two idle DCUs

The host needs the Hygon driver compatible with DTK 26.04, Docker, `/opt/hyhal`,
`/dev/kfd`, `/dev/mkfd`, and the render nodes for two available DCUs. Check
current utilization, memory, and KFD processes with the host's `hy-smi` or DTK
`rocm-smi`, and confirm its reported devices against the render-node mapping.
A small driver memory reservation can remain on an idle card; verify that no
workload owns either selected device before starting a container.

The example below uses host physical DCUs **2 and 3**, whose render nodes were
`renderD130` and `renderD131` on the validated host. Resolve the device mapping
on your own host before reusing these paths. The container exposes only that
pair; its runtime device indices are **0 and 1**. Do not pass host indices
`2,3` as the logical indices inside this isolated container.

```bash
ls -l /dev/kfd /dev/mkfd /dev/dri/renderD130 /dev/dri/renderD131
readlink -f /sys/class/drm/renderD130/device
readlink -f /sys/class/drm/renderD131/device
```

## 2. Prepare the image and local model directory

The tested models are `Qwen3.6-27B` and `Qwen3.6-35B-A3B`. Their directories
must contain the original model configuration, tokenizer, and all checkpoint
shards. Set `MODEL_ROOT` to the parent directory containing these models.

```bash
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-hygon-ci
export MODEL_ROOT=/public-flash/models
export MODEL_NAME=Qwen3.6-27B
export SERVED_MODEL=hygon-qwen36
export CONTAINER=hygon-qwen36-tp2
export PORT=18035

docker pull "$IMAGE"
test -f "$MODEL_ROOT/$MODEL_NAME/config.json"
```

For the MoE model, set `MODEL_NAME=Qwen3.6-35B-A3B` before creating the container.
The 0.24 vendor base image alone does not contain the 0.28 deployment stack.

## 3. Create a container for the selected pair

```bash
docker run --detach --name "$CONTAINER" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/mkfd \
  --device /dev/dri/renderD130 --device /dev/dri/renderD131 \
  --group-add video \
  --security-opt seccomp=unconfined \
  --volume /opt/hyhal:/opt/hyhal:ro \
  --volume "$MODEL_ROOT:/models:ro" \
  --env GEMS_VENDOR=hygon --env USE_FLAGGEMS=1 --env VLLM_PLUGINS=fl \
  --env HIP_VISIBLE_DEVICES=0,1 --env CUDA_VISIBLE_DEVICES=0,1 \
  --env MODEL_PATH="/models/$MODEL_NAME" \
  --env SERVED_MODEL="$SERVED_MODEL" --env PORT="$PORT" \
  --entrypoint /bin/bash "$IMAGE" -lc 'exec sleep infinity'
```

The image places `/opt/vllm-fl/venv/bin` first on `PATH`; the commands below
use `/opt/vllm-fl/venv/bin/python`. Do not mount a plugin
checkout over its installed packages or run `pip install -e .` in this container.

## 4. Verify the installed stack

```bash
docker exec -i "$CONTAINER" /opt/vllm-fl/venv/bin/python -I -B - <<'PY'
import json
import os
from importlib import metadata

import torch
import vllm
import vllm_fl
import flag_gems

expected = {
    "torch": "2.10.0+das.opt1.dtk2604.20260325.g6b060a",
    "vllm": "0.28.0+empty",
    "flag-gems": "0.1.0",
    "flagtree": "0.6.0+hcu3.6",
}
for name, version in expected.items():
    actual = metadata.version(name)
    print(f"{name}: {actual}")
    assert actual == version, (name, actual, version)
for name in ("vllm", "flag-gems", "vllm-plugin-fl"):
    dist = metadata.distribution(name)
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    assert not direct_url.get("dir_info", {}).get("editable", False), name
    print(f"{name}: {dist.version}, installed at {dist.locate_file('')}")
assert torch.__version__ == "2.10.0"
assert vllm.__version__ == "0.28.0"
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
assert "FLAGCX_PATH" not in os.environ
print("Torch:", torch.__file__)
print("vLLM:", vllm.__file__)
print("Plugin:", vllm_fl.__file__)
print("FlagGems:", flag_gems.__file__)
print("DCUs:", [torch.cuda.get_device_name(i) for i in range(2)])
PY
```

Distribution versions and imported module versions are both checked because
`0.28.0+empty` imports as `0.28.0`, and the vendor PyTorch distribution imports
as `2.10.0`. An older global vLLM installation must not shadow the image's wheel.
FlagCX is optional for this deployment. Leave `FLAGCX_PATH` unset to use the
validated vendor RCCL route; setting it to an empty string still selects FlagCX.

## 5. Start the TP2 service

The following graph configuration matches the validated short-context runs.

```bash
docker exec --detach "$CONTAINER" /bin/bash -lc '
  exec /opt/vllm-fl/venv/bin/python -I -B -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 2 --max-model-len 4096 \
    --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
    --compilation-config '\''{"cudagraph_mode":"FULL_AND_PIECEWISE"}'\'' \
    --cudagraph-capture-sizes 1 2 4 8 --max-cudagraph-capture-size 8 \
    > /tmp/vllm-server.log 2>&1
'
docker exec "$CONTAINER" tail -n 40 /tmp/vllm-server.log
curl --fail "http://127.0.0.1:$PORT/health"
```

Wait for model loading and graph capture to finish before sending requests.
For an eager run, replace the three graph-related options
(`--compilation-config`, `--cudagraph-capture-sizes`, and
`--max-cudagraph-capture-size`) with `--enforce-eager`.

## 6. Send a real chat request

```bash
curl --fail --show-error "http://127.0.0.1:$PORT/v1/chat/completions" \
  --header 'Content-Type: application/json' \
  --data "{
    \"model\": \"$SERVED_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 2 + 2?\"}],
    \"temperature\": 0,
    \"max_tokens\": 32,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }"
```

Confirm HTTP 200 and generated text in `choices[0].message.content`. Health
and successful server startup alone do not verify model generation.
For image requests, send an OpenAI `image_url` content item containing a data
URI, or explicitly configure a local media path mounted into the container.

## Validation scope

On the validated two-card stack, both Qwen3.6 models passed the original text,
image, concurrent text/image, and mixed-request cases in eager and graph modes:

| Model | Eager | Graph |
|---|---|---|
| Qwen3.6-27B | 26/26 requests | 26/26 requests |
| Qwen3.6-35B-A3B | 26/26 requests | 26/26 requests |

The four runs cover 20 scenarios and 104 requests. Eager runs used the exact
committed attention source; graph runs used the ordinary plugin wheel
`0.0.0+g04ae0f090`. Native HIP capture begin/end and graph launches were verified
in profiler traces, including replay on both TP ranks. These checks establish
this model/configuration scope; they are not a throughput benchmark or a
complete operator/collective ABI test.

## Build the CI image from this repository

Build on a Linux host from a clean committed checkout. Docker BuildKit and the
buildx plugin are required. The build wrapper records the current Git revision;
use the same committed source for the wheel and image.

```bash
docker buildx version
git diff --exit-code HEAD -- vllm_fl pyproject.toml setup.py README.md LICENSE
bash docker/build.sh --platform hygon --target ci \
  --image-name harbor.baai.ac.cn/plugin/vllm-plugin-fl \
  --image-tag v0.28.0-hygon-ci
```

The Hygon Dockerfile retains the vendor DTK/PyTorch base and its noneditable
FlagGems/FlagTree installation, then builds and installs ordinary wheels for
the empty vLLM and this plugin.
Builds use `--no-deps --no-build-isolation` to preserve the vendor runtime.
Do not apply the upstream CUDA PyTorch requirements to this Hygon image.
If Git needs a proxy, set `HYGON_GIT_CONFIG` to a local Git configuration file.
The builder mounts it as a BuildKit secret for Git only; it does not store the
proxy in image layers. Keep HTTP proxy variables unset for pip and curl.
The existing Hygon CI runner label is `hg-cicd-vllm-plugin-0240`; its name is
an infrastructure label and does not specify the vLLM version inside the image.

## Troubleshooting

- **Imported vLLM is still 0.24:** use the image's Python environment and check
  `vllm.__file__`. Remove conflicting `PYTHONPATH` or source mounts; verify both
  distribution and runtime versions with the check above.
- **No DCUs or the wrong pair:** recheck host render-node mapping, permissions,
  and current ownership. Confirm that the isolated container sees exactly two
  logical devices before serving.
- **HIP compiler or shared-library errors:** preserve the image's DTK paths and
  the host `/opt/hyhal` mount. The validated clang path is
  `/opt/rocm/aillvm/bin/clang-18`; installing CUDA wheels does not fix this stack.
- **Text works but visual answers are wrong:** use the plugin revision containing
  the Hygon vision SDPA fix. That fix makes Q/K/V contiguous after transpose and
  pads head dimension 72 to 128 while preserving the original attention scale.
  Keep the default dispatch policy rather than adding manual operator exclusions.

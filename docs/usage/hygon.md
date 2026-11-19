# Hygon Deployment Guide

Run an OpenAI-compatible service with the pinned image below. The image contains
ordinary installed vLLM, FlagGems, and plugin packages; no source checkout or
editable installation is required. The recipe uses two Hygon BW1000 64 GiB DCUs.

> This guide targets the `v0.32.0-hygon-ci` image on the FlagOS 2.2 base. The
> 0.28 image and recipe were superseded; see
> [vLLM 0.32 implementation notes](#vllm-0.32-implementation-notes) for what
> changed with the move.

## Deployment stack

| Component | Version or configuration |
|---|---|
| Image (Linux AMD64) | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-hygon-ci@sha256:558066eb6677cd33d60bc97010d0b91c991941eb05631c9a9265372c8e319f84` |
| Vendor base | DTK 26.04 (`/opt/dtk-26.04`) / Ubuntu 24.04 / Python 3.10.20 |
| Host kernel driver | `hycu` kernel module `6.4.0-V1.7.0c` (vermagic `6.8.0-87-generic`) |
| HyHAL | `6.3.3-V1.7.0c` (20260630-175451), installed at `/opt/hyhal` |
| RCCL | `2.22.3` (`/opt/dtk/lib/librccl.so.1`, reported by `ncclGetVersion`) |
| HIP runtime | `6.3.26045` (`torch.version.hip`); `hipcc` from DTK `dcc 25.10.0`, clang 17.0.0 |
| vLLM | Distribution `0.32.0+empty`; runtime `0.32.0` |
| PyTorch | Distribution `2.9.0+das.opt1.dtk2604`; runtime `2.9.0` |
| FlagGems distribution | `5.4.0` |
| FlagTree distribution | `0.7.0+hcu3.6` |
| Triton distribution | `3.6.0`, coexists with FlagTree |
| Model runner | Upstream `GPUModelRunner` (v2) by default; `ModelRunnerFL` (v1) with `VLLM_USE_V2_MODEL_RUNNER=0` |
| Communication | Vendor HIP/RCCL `2.22.3`, TP2; `FLAGCX_PATH` unset |

The rows above were read from the pinned image's exact digest
(`558066eb...`) on the validation host: DTK from `/opt/dtk-26.04/.info/version-dev`,
HyHAL from `/opt/hyhal/.info/version`, RCCL from the version string in
`librccl.so.1`, and the kernel module from `modinfo hycu`. The same package set
ships in every `v0.32.0-hygon-ci` rebuild, because they all derive from the
FlagOS 2.2 base image.

Scope of this recipe and of the validation behind it:

| Item | BW1000 64 GiB (this recipe) | BW1101 144 GiB (gate validation) |
|---|---|---|
| Cards per run | 2 (TP2) | 2 of 8 (TP2, host cards 6 and 7) |
| Card memory | 65520 MiB usable, reported by `hy-smi --showmeminfo vram` | BW1101 144 GiB, 800 W power cap; eight cards on `renderD128`-`renderD135` |
| Host | CI runner `hg-cicd-vllm-plugin-0240` | `bm-srwl-sh-zone9-d-bw1100-144g-1-80`, kernel `6.8.0-87-generic` |
| Verified by | CI unit, functional, inference and the four-group adaptation gate | four-group adaptation gate, eager and graph |
| Driver / RCCL | same image and driver series | `hycu 6.4.0-V1.7.0c`, HyHAL `6.3.3-V1.7.0c`, RCCL `2.22.3` |

The 64 GiB memory figure comes from `--showmeminfo vram`, which reports usable
device memory rather than the marketing capacity. The bandwidth, capacity and
power numbers of the two hosts differ, so a passing run on one does not by
itself certify the other; the graph-memory headroom is therefore configurable
(see [vLLM 0.32 implementation notes](#vllm-0.32-implementation-notes)) instead
of being hard-coded for one card size.

Keep the image's vendor Torch, FlagTree/Triton, and
[Hygon dispatch policy](../../vllm_fl/dispatch/config/hygon.yaml) together.
The empty vLLM wheel omits `vllm._C` and `vllm._C_stable_libtorch`; do not copy
extensions from an older installation. The image enables `USE_FLAGGEMS=1`,
`GEMS_VENDOR=hygon`, and `VLLM_PLUGINS=fl`, and does not pin
`VLLM_USE_V2_MODEL_RUNNER`: 0.32 defaults to the V2 runner and the plugin keeps
that default. The empty build has no `_C::get_cuda_view_from_cpu_tensor`, so the
plugin reports UVA as unavailable and V2 takes upstream's own `NonUvaBuffer`
fallback instead of selecting `ModelRunnerFL`.

## 1. Check the host and select two idle DCUs

The validated host combination is the `hycu` kernel module `6.4.0-V1.7.0c` with
HyHAL `6.3.3-V1.7.0c` (20260630-175451) and RCCL `2.22.3`; both numbers are the
values the image itself reports, so mismatched host and container HyHAL is the
first thing to rule out when a device is not visible. The host also needs
`/opt/hyhal`, `/dev/kfd`, `/dev/mkfd`, and the render nodes for two available
DCUs. Check current utilization, memory, and KFD processes with the host's
`hy-smi` or DTK `rocm-smi`, and confirm its reported devices against the
render-node mapping. A small driver memory reservation can remain on an idle
card; verify that no workload owns either selected device before starting a
container.

```bash
modinfo hycu | grep -E '^version|^vermagic'
head -1 /opt/hyhal/.info/version
uname -r
```

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
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-hygon-ci@sha256:558066eb6677cd33d60bc97010d0b91c991941eb05631c9a9265372c8e319f84
export MODEL_ROOT=/public-flash/models
export MODEL_NAME=Qwen3.6-27B
export SERVED_MODEL=hygon-qwen36
export CONTAINER=hygon-qwen36-tp2
export PORT=18035

docker pull "$IMAGE"
test -f "$MODEL_ROOT/$MODEL_NAME/config.json"
```

For the MoE model, set `MODEL_NAME=Qwen3.6-35B-A3B` before creating the container.
The 0.24 vendor base image alone does not contain the 0.32 deployment stack.

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

The image keeps the vendor virtualenv at `/flagos` (`VIRTUAL_ENV=/flagos`) and
puts `/flagos/bin` first on `PATH`; the commands below use
`/flagos/bin/python`. Do not mount a plugin
checkout over its installed packages or run `pip install -e .` in this container.

## 4. Verify the installed stack

```bash
docker exec -i "$CONTAINER" /flagos/bin/python -I -B - <<'PY'
import json
import os
from importlib import metadata

import torch
import vllm
import vllm_fl
import flag_gems

expected = {
    "torch": "2.9.0+das.opt1.dtk2604",
    "vllm": "0.32.0+empty",
    "flag-gems": "5.4.0",
    "flagtree": "0.7.0+hcu3.6",
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
assert torch.__version__ == "2.9.0"
# The wheel is built as 0.32.0+empty, so vllm.__version__ keeps the local
# suffix; compare the base version, as the CI image self-check does.
assert vllm.__version__.split("+")[0] == "0.32.0"
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
assert "FLAGCX_PATH" not in os.environ
print("Torch:", torch.__file__)
print("vLLM:", vllm.__file__)
print("Plugin:", vllm_fl.__file__)
print("FlagGems:", flag_gems.__file__)
print("DCUs:", [torch.cuda.get_device_name(i) for i in range(2)])
PY
```

Distribution versions and imported module versions are both checked because the
imported `vllm.__version__` carries the wheel's local suffix while the
distribution check pins the exact `0.32.0+empty`, and the vendor PyTorch
distribution imports as `2.9.0`. An older global vLLM installation must not
shadow the image's wheel.
FlagCX is optional for this deployment. Leave `FLAGCX_PATH` unset to use the
validated vendor RCCL route; setting it to an empty string still selects FlagCX.

## 5. Start the TP2 service

The following graph configuration is the one the Hygon CI matrix runs
(`tests/platforms/hygon.yaml`, `device_overrides.bw1000.llm`), so a local run
and a CI run start from the same engine settings.

```bash
docker exec --detach "$CONTAINER" /bin/bash -lc '
  exec /flagos/bin/python -I -B -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
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

### Why each non-default flag is set

The flags below are the only overrides; everything else stays at the vLLM
default. They fall into two groups: values the DCU forces, and values that
reproduce the measured gate workload. Remove or relax the second group when
you are not reproducing the gate.

| Flag | Applies to | Why it is set | Effect of changing it |
|---|---|---|---|
| `--tensor-parallel-size 2` | both models | Two DCUs, TP2 over vendor RCCL. The 35B MoE weights in bf16 exceed a single 64 GiB card, and the 27B leaves no usable KV space on one card. | Requires re-running with a matching card count; collectives must stay on RCCL. |
| `--gpu-memory-utilization 0.85` | both models | Keeps a 15% device margin for the host driver reservation and capture growth. This matters on this platform because `is_cuda_alike()` is `False`, so upstream's cudagraph memory estimator never runs and the KV cache would otherwise take the entire budget; the plugin subtracts its own fixed headroom of 4 GiB (never more than 30% of one card). 0.85 is the value the recorded runs used; the headroom, not the ratio, is the part the platform requires. | Raising it shrinks the margin below the recorded footprint and risks a HIP OOM during the first capture, which leaves DCU memory unreclaimable; lowering it cuts KV blocks and therefore the maximum context that fits. |
| `--cudagraph-capture-sizes 1 2 4 8` and `--max-cudagraph-capture-size 8` | graph runs only | Required on this platform, not optional tuning. vLLM 0.32 generates the table from `max_num_seqs`/`max_num_batched_tokens` (up to 512 tokens) before the platform hook runs, and every captured size costs DCU memory on top of the KV cache. The upstream estimator that would account for that never runs here (`is_cuda_alike()` is `False`), so an unconstrained table runs out of memory during the first capture and leaves DCU memory unreclaimable. Eight entries cover the concurrency this recipe allows. | A larger table increases the capture footprint and can OOM during startup; a smaller one re-uses a bigger graph for small batches, which costs latency but not correctness. |
| `--max-model-len 4096` | both models | Gate workload constraint. The adaptation gate sends short text and image prompts, and 4096 was the limit used for the recorded results on two 64 GiB cards. | Longer limits consume more KV blocks per sequence and change the measured memory profile; re-measure before publishing numbers. |
| `--max-num-seqs 8` | both models | Gate workload constraint. The gate's concurrency scenario sends eight simultaneous requests, so eight sequences bound both the scheduler batch and the KV peak for the reported run. | Larger values raise concurrency and KV pressure; the `1 2 4 8` capture table then no longer covers every batch size. |
| `--max-num-batched-tokens 8192` | both models | Gate workload constraint. Caps one prefill step at 8192 tokens so a long prompt cannot spike activation memory beyond the budget the capture table assumes. | Raising it allows larger prefills at higher peak memory; lowering it splits prefills into more steps. |
| `--no-enable-prefix-caching` | both models | Gate workload constraint. The gate requests are independent, so caching would let later requests reuse blocks from earlier ones and shift both latency and memory numbers away from the recorded baseline. It also keeps the hybrid MoE case consistent with the V1 run, which must use `--mamba-cache-mode none`. | For a production workload with repeated prefixes, enabling it is the expected default; only the gate numbers assume it is off. |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` | graph runs only | Both capture modes are exercised: piecewise graphs for prefill and decode, plus full graphs where the scheduler can replay a whole batch. This is the mode the recorded `PIECEWISE`/`FULL` capture counts refer to. | Other modes change which batches are replayed and therefore both latency and the capture footprint. |

Two rows are worth calling out because they are often mistaken for tuning. The
capture-size pair is **mandatory** on this platform: it is the only thing that
bounds graph-capture memory, because the upstream estimator does not run here.
`--gpu-memory-utilization 0.85` is the additional device margin the recorded
runs used and should be re-measured rather than copied blindly.
`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens` and
`--no-enable-prefix-caching` exist only to reproduce the gate workload and can
be relaxed for real traffic.

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

## Build the CI image

Build on a Linux host from a clean committed checkout. Docker BuildKit and the
buildx plugin are required. The build wrapper records the selected Git revision;
use the same committed source for the wheel and image. The wrapper archives the
selected committed Git revision (default: `HEAD`) as the build context;
uncommitted files are not included. Set `PLUGIN_SOURCE_SHA` to select another
existing 40-character commit SHA.

```bash
docker buildx version
git diff --exit-code HEAD -- vllm_fl pyproject.toml setup.py README.md LICENSE
bash docker/build.sh --platform hygon --target ci \
  --image-name harbor.baai.ac.cn/plugin/vllm-plugin-fl \
  --image-tag v0.32.0-hygon-ci
```

The Hygon Dockerfile starts from the FlagOS 2.2 release image
(`HYGON_BASE_IMAGE`, DTK 26.04 / Python 3.10 / Torch 2.9), keeps its noneditable
FlagGems/FlagTree installation, adapts the upstream vLLM source to Python 3.10
(`.github/scripts/hygon/patch_vllm_py310.py`), then builds and installs ordinary
wheels for the empty vLLM (`0.32.0+empty`) and this plugin.
Current source builds and CI also install **FlagGems-vllm v0.2.0**, commit
[`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`](https://github.com/flagos-ai/FlagGems-vllm/commit/b5c2c2249319277ccda4d41aa0f65164bd9b2a19),
through the shared installer, alongside the vendor FlagGems package. Its
package metadata still reports `0.1.0`; the commit identifies the release.
The pinned 0.28 image and the [historical validation in PR #576](https://github.com/flagos-ai/vllm-plugin-FL/pull/576)
predate this integration.
Builds use `--no-deps --no-build-isolation` to preserve the vendor runtime.
Do not apply the upstream CUDA PyTorch requirements to this Hygon image.
If Git needs a proxy, set `HYGON_GIT_CONFIG` to a local Git configuration file.
The builder mounts it as a BuildKit secret for Git only; it does not store the
proxy in image layers. Keep HTTP proxy variables unset for pip and curl.
For an offline upstream install, set `HYGON_VLLM_WHEEL_DIR` to a directory
containing exactly one previously built `vllm-0.32.0+empty-*.whl`, and set
`HYGON_VLLM_WHEEL_SHA256` to its verified SHA256. The build checks that digest
and installs the wheel normally; it does not copy old native vLLM extensions.
Use a wheel built from upstream vLLM commit
[`5df63a7203cdb9328d824a27b682e77c7b9928be`](https://github.com/vllm-project/vllm/commit/5df63a7203cdb9328d824a27b682e77c7b9928be).
Leave these variables unset to build vLLM from the official Git source.

The existing Hygon CI runner label is `hg-cicd-vllm-plugin-0240`; its name is
an infrastructure label and does not specify the vLLM version inside the image.

## vLLM 0.32 implementation notes

The 0.32 line moves upstream to Python `>=3.11` and a newer Torch, while the
FlagOS 2.2 Hygon release image ships Python 3.10 / Torch 2.9 / DTK 26.04. The
plugin is ported to 0.32 on top of that image; the validated stack is:

| Component | Version |
|---|---|
| Base image (FlagOS 2.2) | `harbor.baai.ac.cn/flagos-app/vllm0.24.0-hygon-dtk26.04:2.2.0-0.3.0` |
| Python | 3.10.20 |
| Torch | 2.9.0+das (DTK 26.04) |
| vLLM | `0.32.0+empty` (built from the upstream `v0.32.0` source) |
| FlagGems / FlagTree | `5.4.0` / `0.7.0+hcu3.6` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch290` |
| FlagGems-vllm | `0.2.0` (`v0.2.0` tag; metadata reports `0.1.0`) |

Steps that differ from the legacy 0.28 recipe:

1. **Make vLLM 0.32 importable on Python 3.10.** The image build runs
   `.github/scripts/hygon/patch_vllm_py310.py`, which rewrites the 3.11-only
   typing imports (`Self`, `Never`, `override`, `assert_never`,
   `LiteralString`, `NotRequired`, `Required`, `TypeVarTuple`, `Unpack`,
   `CapsuleType`, `TypeAliasType`, `TypeIs`, `deprecated`, `reveal_type`,
   `ReadOnly`) to `typing_extensions` (37 files), backports `datetime.UTC`
   (2 files), and relaxes `requires-python` to `>=3.10,<3.15`. The empty wheel
   is then built with `VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.32.0+empty`,
   `VLLM_USE_PRECOMPILED=0`, `VLLM_USE_PRECOMPILED_RUST=0`,
   `CARGO_NET_OFFLINE=true`, and `pip wheel --no-deps --no-build-isolation`.
   Monkeypatching `typing` globally does not work: `typing_extensions` detects
   PEP 695 via `hasattr(typing, "TypeAliasType")` and then calls
   `typing.ParamSpec(..., infer_variance=...)`, which Python 3.10 rejects.
2. **Install FlagGems-vllm** (`v0.2.0`). Without `flaggems_vllm` the FlagGems
   backend cannot register and dispatch falls back to `reference.torch`.
3. **Model runner selection follows upstream.** 0.32 defaults to the V2 model
   runner and Hygon keeps that default, matching the NVIDIA path. The plugin
   ships no V2 runner of its own, so the default runs vLLM's own V2 runner.
   `VLLM_USE_V2_MODEL_RUNNER=0` selects the plugin's `ModelRunnerFL` (a fork of
   the v1 runner) and goes through upstream's v1 feature validation. Do not pin
   the variable in images or launch scripts.
4. **Blacklist `zeros` / `zero_` in the Hygon dispatch policy.** FlagGems'
   Triton zero kernels pass the element count as int32, so zeroing the multi-GB
   KV-cache backing buffer (`> 2**31` int8 elements) fails with
   `Triton Error [HIP]: invalid argument`. The native ATen memset is used
   instead (`torch.zeros` lowers to `empty` + `zero_`).
5. **Shim the Torch 2.9 accelerator attribute.** 0.32 queries
   `torch.accelerator.get_memory_info` on the worker startup path
   (`MemorySnapshot`) and for the KV-cache memory budget; the Hygon patch
   module binds it to `torch.cuda.mem_get_info`. The other accelerator entry
   points 0.32 uses (`current_device_index` / `set_device_index`) already ship
   with DTK Torch 2.9, and 0.32 guards its only `empty_host_cache` call with
   `try/except AttributeError`, so no other shim is installed.
6. **Filter IR-op providers per operator.** 0.32 asserts that every provider in
   the priority list is registered for that op. The DCU wheel ships no
   `vllm._C`, so `vllm_c` is only kept when the op actually registered it and
   the rest fall back to `native`; CUDA behaviour is unchanged because it
   registers both.
7. **Bound graph-capture memory.** 0.32 generates the capture table from
   `max_num_seqs`/`max_num_batched_tokens` (up to 512 tokens) before the
   platform hook runs, and Hygon DCUs report `is_cuda_alike() == False`, so the
   upstream graph-memory estimator never runs and the KV cache takes the whole
   budget. The plugin reserves a fixed 4 GiB of graph headroom, clamped to 30%
   of the DCU budget so a smaller card is never starved, but it does not resize
   the table: pass `--cudagraph-capture-sizes` together with
   `--max-cudagraph-capture-size` whenever graph mode is on. An unconstrained
   table does not fit in that headroom.

Runtime removals that the plugin had to drop (0.32 has no replacement):

- XD-RoPE (`XDRotaryEmbedding`, `supports_xdrope`, `uses_xdrope_dim`).
- Legacy routed-experts capture (`RoutedExpertsLists` / `RoutedExpertsTensors`);
  0.32 reports routed experts through `AuxOutputConfig`, and
  `enable_return_routed_experts` is currently a warning no-op in the FL runner.
- Sequence-parallel collective fusion (`PassConfig.enable_sp`) and the
  residual-scattered-for-SP padding.
- kv-connector uniform KV cache (`use_uniform_kv_cache` /
  `allocate_uniform_kv_caches`); KV cache allocation now delegates to 0.32's
  `allocate_kv_cache` (single backing buffer plus per-layer views).
- `get_total_cp_world_size`, `InputBatch(kernel_block_sizes)`,
  `MultiModalRegistry.get_dummy_mm_inputs` / `supports_multimodal_inputs`,
  `_reshape_kv_cache_tensors`, `_seq_lens_cpu` on `CommonAttentionMetadata`.

Validated on two BW1000 64 GiB DCUs with `Qwen3.6-27B` (hybrid GDN attention):
TP1 eager, TP2 eager, TP1 graph (`PIECEWISE` 5 + `FULL` decode 4 graphs), and
`vllm serve` with `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`. The unit suite reports 1912 passed / 194 skipped.

The V2 runner runs vLLM's own `GPUModelRunner`; the plugin adds no V2 runner of
its own. A 27B TP2 serve on two BW1100 DCUs captured `PIECEWISE` 11/11 and
`FULL` 11/11 graphs (1.31 GiB of graph memory) and passed the text, image and
mixed request checks.

Not exercised on this stack: Ascend dense hybrid KV cache (Ascend-only),
`enable_return_routed_experts`, XD-RoPE models, and GDN / mamba hybrid models on
the V2 runner. Graph-mode performance has not been tuned on this stack.

Not supported / not verified on this stack:

- `USE_FLAGGEMS=0` (or `VLLM_FL_PREFER=vendor`) for MoE models. The DCU
  wheel has no native unquantized MoE backend, so selection ends in
  `NotImplementedError: No Unquantized MoE backend supports the deployment
  configuration.` The dispatch fallback only makes the failure explicit; it does
  not provide a working non-FlagGems path.
- Collectives inside piecewise graphs. `enable_sp` is gone upstream and the
  plugin runs with SP disabled; no experiment has checked whether any collective
  gets captured into a piecewise graph on this stack.
- Multi-node TP/EP and inter-node collectives.

## Validation scope

The 0.32 line is validated by the `vllm-plugin-FL` CI on the Hygon runner and by
the four-group adaptation gate described above; see
[PR #711](https://github.com/flagos-ai/vllm-plugin-FL/pull/711) for the run
identifiers, per-scenario reports and the exact source and image digests.
[PR #576](https://github.com/flagos-ai/vllm-plugin-FL/pull/576) covers the
superseded 0.28 line and is kept only as history. Results apply to the recorded
source and image artifacts; they do not establish complete
operator/collective ABI coverage or performance.

## Troubleshooting

- **Imported vLLM is an older release:** use the image's Python environment and check
  `vllm.__file__`. Remove conflicting `PYTHONPATH` or source mounts; verify both
  distribution and runtime versions with the check above.
- **No DCUs or the wrong pair:** recheck host render-node mapping, permissions,
  and current ownership. Confirm that the isolated container sees exactly two
  logical devices before serving.
- **HIP compiler or shared-library errors:** preserve the image's DTK paths and
  the host `/opt/hyhal` mount. The image compiles with DTK `dcc 25.10.0`
  (`clang version 17.0.0` under `/opt/dtk/bin`); installing CUDA wheels does not
  fix this stack.
- **Text works but visual answers are wrong:** use the plugin revision containing
  the Hygon vision SDPA fix. That fix makes Q/K/V contiguous after transpose and
  pads head dimension 72 to 128 while preserving the original attention scale.
  Keep the default dispatch policy rather than adding manual operator exclusions.

# AMD Deployment Guide

Deploying vllm-plugin-FL on AMD ROCm GPUs. Validated on AMD Radeon PRO W7900D
(gfx1100, RDNA3) with the full FlagOS stack — FlagGems, FlagTree and FlagCX
active simultaneously.

## Deployment stack

| Component | Version or configuration |
|---|---|
| Accelerator | AMD Radeon PRO W7900D 48 GB, **gfx1100 (RDNA3)**, 8 cards |
| Host base | Ubuntu 24.04 / ROCm 7.2.4 |
| Container base | `vllm/vllm-openai-rocm:latest` (ROCm 7.2.3, Python 3.12) |
| vLLM | `0.32.0+empty` — rebuilt from tag `v0.32.0` with `VLLM_TARGET_DEVICE=empty` |
| PyTorch | `2.13.0+git733fca1` (ROCm, HIP 7.2.53211) |
| Triton | `3.6.0`, **supplied by FlagTree** from `/opt/flagtree` |
| FlagTree distribution | `0.7.0+amd3.6` |
| FlagGems | `main`, post-5.4.0 — needs commit [a8fdfa0](https://github.com/flagos-ai/FlagGems/commit/a8fdfa082487e7ef73e65adcdaa79c134e6d790f) |
| FlagGems-vllm | Commit [`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`](https://github.com/flagos-ai/FlagGems-vllm/commit/b5c2c2249319277ccda4d41aa0f65164bd9b2a19) (`v0.2.0`; package metadata reports `0.1.0`) — the project-wide pin |
| FlagCX | `0.15.0.dev52+g591cfbf`, built with `USE_AMD=1` (RCCL backend) |
| Attention backend | `TRITON_ATTN` (dense), `TRITON_MLA` (MLA); ViT uses Torch SDPA |
| Communication | FlagCX over RCCL; custom all-reduce disabled |
| Compilation | **Eager only** — see [Validation scope](#validation-scope) |

Two properties of this hardware drive several choices below:

- **gfx1100 is RDNA3, not CDNA.** Composable Kernel's flash attention is built
  for CDNA and segfaults here, so the multimodal encoder uses Torch SDPA.
- **64 KB LDS.** Some kernel autotuning configurations ask for 120 KB and are
  rejected; Triton recovers by selecting a smaller configuration, which costs
  first-run time rather than correctness.

## Build the image

The quickest path is the AMD Dockerfile, which performs every step in section 4
(FlagTree, the empty vLLM, FlagGems, FlagGems-vllm, FlagCX and the plugin) and
fails the build if the resulting stack is not the validated one:

```bash
./docker/build.sh --platform amd --target release
```

It needs two source images, because neither is sufficient alone: the ROCm base
image has a Torch built for gfx1100 but ships upstream Triton 3.8 and no
FlagGems, while the FlagOS runtime image carries FlagTree but its Torch cannot
see the GPUs. Override either with `AMD_BASE_IMAGE` /
`AMD_FLAGOS_RUNTIME_IMAGE`; `./docker/build.sh --help` lists the rest.

Note that the base image's entrypoint is `vllm serve`, so use
`--entrypoint python3` (or `bash`) when inspecting the image rather than
serving from it.

Sections 1–7 below are the same stack assembled by hand, which is how the
results in [Validation scope](#validation-scope) were produced. Follow them to
work incrementally, or to adapt the stack to another ROCm image.

## 1. Check the host and the GPUs

```bash
cat /opt/rocm/.info/version          # expect 7.2.x
rocm-smi --showproductname | head -20
rocm-smi --showpids                  # confirm no stale processes hold VRAM
```

The host render and compute nodes must be present, and note their group IDs —
they are needed for the container:

```bash
ls -l /dev/kfd /dev/dri/renderD*
getent group video render            # e.g. video:x:44, render:x:993
```

## 2. Prepare the base image

```bash
# Docker Hub may be unreachable; a mirror works.
docker pull docker.m.daocloud.io/vllm/vllm-openai-rocm:latest
```

> **Torch must be built for gfx1100.** The FlagOS runtime image
> (`flagos-runtime-amd-rocm7.2.4`) carries FlagGems and FlagTree, but its Torch
> reports an empty `torch.cuda.get_arch_list()`, so no GPU is visible even
> though `rocminfo` inside the container sees gfx1100. Always check:
>
> ```bash
> python3 -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.device_count())"
> ```
>
> `gfx1100` must appear in the list and the device count must be non-zero.

## 3. Create the container

Pass both device nodes and add the host's `video`/`render` groups by **numeric
GID** — the group names do not exist inside the image.

```bash
docker run -d --name vllm-fl-amd --entrypoint bash \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 993 \
  --ipc=host --shm-size=16g \
  -v /data/fl-amd:/work -v /data/models:/models \
  docker.m.daocloud.io/vllm/vllm-openai-rocm:latest -c 'sleep infinity'
```

## 4. Install the stack

### 4.1 Install FlagTree

FlagTree `0.7.0+amd3.6` is **not on PyPI** — `pip install flagtree` fails with
`No matching distribution found`. The AMD build is distributed inside the
FlagOS runtime image, so take `/opt/flagtree` from there. It is self-contained
(1.1 GB, ships its own `triton`), which is why its Torch being unusable on
gfx1100 does not matter here: only this directory is used.

Run on the **host**, into the directory mounted at `/work`:

```bash
docker create --name flagtree-src \
  harbor.baai.ac.cn/flagos-runtime/flagos-runtime-amd-rocm7.2.4:2.3.0
docker cp flagtree-src:/opt/flagtree /data/fl-amd/flagtree
docker rm flagtree-src
```

Then, inside the container, put it at `/opt/flagtree`:

```bash
cp -r /work/flagtree /opt/flagtree
ls /opt/flagtree        # bin  flagtree  flagtree-0.7.0+amd3.6.dist-info  triton
```

### 4.2 Make FlagTree own the `triton` module

FlagGems requires Triton ≤ 3.6; the base image ships 3.8. Move the image's
Triton aside (reversibly) and put FlagTree first on the path:

```bash
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
for d in triton pytorch_triton; do
  [ -d "$SP/$d" ] && mv "$SP/$d" "$SP/.$d-image-backup"
done
printf '/opt/flagtree\n' > "$SP/zz-flagtree.pth"

python3 -c "import triton, importlib.metadata as md;
print(triton.__version__, triton.__file__);
print('flagtree', md.version('flagtree'))"
```

`triton.__file__` must resolve under `/opt/flagtree`, and the reported
`flagtree` version must be `0.7.0+amd3.6`. Upstream Triton 3.8 is not a
supported substitute.

### 4.3 Rebuild vLLM as an empty distribution

Non-NVIDIA backends use empty builds uniformly.

```bash
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm /work/vllm-src
cd /work/vllm-src
pip install --no-cache-dir "setuptools>=77,<81" setuptools-scm wheel pybind11 ninja cmake
VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0 \
  pip wheel --no-deps --no-build-isolation -w /work/wheels .
pip install --no-cache-dir /work/wheels/vllm-0.32.0+empty-*.whl
python3 -c "import vllm; print(vllm.__version__)"   # 0.32.0+empty
```

### 4.4 FlagGems, FlagGems-vllm and FlagCX

FlagGems must come from `main`: the AMD fix that sets
`triton_extra_name="hip"` for the `_amd` backend is not in the 5.4.0 release,
and without it every FlagGems kernel emits CUDA `__nv_*` symbols that the AMD
Triton backend rejects.

```bash
# FlagGems (main)
git clone https://github.com/flagos-ai/FlagGems /work/FlagGems
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r /work/FlagGems/src/flag_gems "$SP/flag_gems"
pip install -q sqlalchemy

# FlagGems-vllm: the project-wide pin (v0.2.0), the same commit
# .github/scripts/install_flaggems_vllm.py installs for every other backend.
# Its package metadata still reports 0.1.0; the commit identifies the release.
git clone https://github.com/flagos-ai/FlagGems-vllm /work/FlagGems-vllm
git -C /work/FlagGems-vllm checkout b5c2c2249319277ccda4d41aa0f65164bd9b2a19
pip install --no-deps --no-build-isolation /work/FlagGems-vllm

# FlagCX with the AMD adaptor
apt-get update -qq && apt-get install -y -qq nlohmann-json3-dev
git clone https://github.com/flagos-ai/FlagCX /work/FlagCX
cd /work/FlagCX && make USE_AMD=1 -j"$(nproc)"
FLAGCX_ADAPTOR=amd USE_AMD=1 FLAGCX_HOME=/work/FlagCX \
  pip install --no-deps --no-build-isolation .
```

> GitHub clones from inside the container can fail intermittently
> (`curl 16 HTTP2 framing layer`, `curl 28 timed out`). Set
> `git config --global http.version HTTP/1.1` and retry; `ls-remote` succeeding
> does not guarantee that `clone` will.

### 4.5 Install the plugin

```bash
cd /work/vllm-plugin-FL
pip install --no-deps --no-build-isolation -e .
```

## 5. Verify the installed stack

```bash
python3 - <<'PY'
import importlib.metadata as md
import torch, triton
print("torch   :", torch.__version__, "| hip", torch.version.hip)
print("triton  :", triton.__version__, triton.__file__)
print("backends:", list(triton.backends.backends.keys()))
for d in ("flagtree", "flag_gems", "flaggems_vllm", "flagcx", "vllm"):
    print(f"{d:14}:", md.version(d))
print("GPUs    :", torch.cuda.device_count())
PY
```

Expected: `triton.__file__` under `/opt/flagtree`, backends include `amd`,
`vllm` is `0.32.0+empty`, and the device count matches the cards passed in.

Confirm the platform resolves to AMD:

```bash
VLLM_PLUGINS=fl python3 -c "
import vllm_fl; vllm_fl.register()
from vllm_fl.platform import PlatformFL as P
print('vendor', P.vendor_name, '| device_type', P.device_type,
      '| is_rocm', P().is_rocm())"
```

Confirm FlagCX is genuinely carrying collectives rather than falling back to
RCCL. A `backend=flagcx` log line alone only shows which backend was
requested, so check that `libflagcx.so` is mapped into the process and that a
real collective is correct:

```bash
cat > /tmp/check_flagcx.py <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group(backend="flagcx")
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
with open("/proc/self/maps") as f:
    libs = sorted({l.split()[-1] for l in f if "flagcx" in l.lower()})
t = torch.full((4,), float(rank + 1), device=f"cuda:{rank}")
dist.all_reduce(t)
expected = float(sum(range(1, ws + 1)))
ok = bool((t == expected).all().item())
print(f"rank {rank}: reduce {t[0].item()} expected {expected} -> "
      f"{'OK' if ok else 'FAIL'}; flagcx libs {libs or 'NONE'}", flush=True)
# Every rank must reach this barrier before any of them exits, otherwise a
# fast rank tears its communicator down mid-collective and the job hangs.
dist.barrier()
# FlagCX's destroy_process_group segfaults on ROCm, so exit without it.
os._exit(0)
PY

FLAGCX_PATH=/work/FlagCX LD_LIBRARY_PATH=/work/FlagCX/build/lib \
  torchrun --nproc-per-node=2 /tmp/check_flagcx.py
```

`/work/FlagCX/build/lib/libflagcx.so` must appear and the reduction must match.
`librccl.so` also being mapped is expected: FlagCX's AMD adaptor uses RCCL
underneath.

## 6. Start the service

FlagCX is selected by the presence of `FLAGCX_PATH`. Without it vLLM silently
uses `backend=nccl` (RCCL) instead.

```bash
export VLLM_PLUGINS=fl USE_FLAGGEMS=1
export FLAGCX_PATH=/work/FlagCX
export LD_LIBRARY_PATH=/work/FlagCX/build/lib:$LD_LIBRARY_PATH

vllm serve /models/Qwen3.6-27B \
  --served-model-name qwen36-27b \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --trust-remote-code \
  --enforce-eager
```

`--enforce-eager` is required; see [Validation scope](#validation-scope).

Confirm all three components are active in the service log:

```
backend=flagcx                       # once per TP rank
using 'default.flagos'               # FlagGems executing the FL operators
Using Torch SDPA backend for ViT     # correct for RDNA
Application startup complete
```

## 7. Send a request

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-27b",
       "messages":[{"role":"user","content":"Name the capital of France."}],
       "max_tokens":32,"temperature":0}' | python3 -m json.tool
```

## Validation scope

See [PR #659](https://github.com/flagos-ai/vllm-plugin-FL/pull/659) for the
full evidence. In summary:

**Passing** — the adaptation gate's eager groups, on 8× W7900D at TP8, with
FlagGems, FlagTree and FlagCX active simultaneously:

| Group | Scenarios | Requests |
|---|---|---|
| `Qwen3.6-27B` eager | 5/5 | 26/26 |
| `Qwen3.6-35B-A3B` eager | 5/5 | 26/26 |

covering single and 8-way-concurrent text, image, and mixed text-image
requests, on a dense model and an MoE model.

FlagCX was confirmed to be in the data path rather than merely requested:
those runs log `backend=flagcx` on all 8 ranks with no `backend=nccl`, the
platform resolves the communicator to `CommunicatorFL`, and the check in
section 5 shows `libflagcx.so` mapped into the process with a correct
`all_reduce`.

**Not supported in this release — graph mode.** Graph capture fails for two
reasons: FlagGems GDN kernels are traced into by Dynamo and need an opaque
custom-op registration ([FlagGems #8095](https://github.com/flagos-ai/FlagGems/issues/8095)),
and a lazily-resolved dispatch lock is entered during tracing. Always pass
`--enforce-eager`.

**Not covered**: TP sizes other than 8, FP8/quantised paths, performance
characterisation, and CI (AMD is intentionally not in automatic CI). Results
apply to the recorded artifacts and do not establish complete operator or
collective ABI coverage.

## Troubleshooting

- **No GPUs visible even though `rocminfo` sees them:** the image's Torch was
  not built for gfx1100. Check `torch.cuda.get_arch_list()`; an empty list is
  the signature. Use a Torch build that lists `gfx1100`.
- **`cannot import name 'is_hip_gfx1250'`:** vLLM's bundled `triton_kernels`
  expect AMD predicates that FlagTree's `target_info` does not define. The
  plugin adds them for AMD in `vllm_fl/patches/flagtree_target_info.py`; make
  sure the plugin is actually imported (`VLLM_PLUGINS=fl`).
- **`Implicit conversion of CUDA __nv_powf device function has been dropped`:**
  FlagGems is resolving CUDA libdevice. Use FlagGems from `main` — the released
  5.4.0 lacks `triton_extra_name="hip"` for the `_amd` backend.
- **`All providers in priority must be registered implementations`:** an empty
  vLLM build registers only the `native` IR-op provider. The plugin probes the
  registry; if you see this, confirm the plugin revision includes that probe.
- **Segfault inside `ck_tile::FmhaFwdKernel`:** Composable Kernel flash
  attention is being used for the vision encoder. It is CDNA-only; the plugin
  routes AMD to Torch SDPA, so check that `Using Torch SDPA backend for ViT`
  appears in the log.
- **`Free memory on device cuda:0 (3.1/47.98 GiB)` on startup:** a previous run
  left workers holding VRAM. They appear as bare
  `python3.12 -c from multiprocessing.spawn ...` processes and match neither
  `vllm serve` nor `VLLM::Worker`. Find them with `rocm-smi --showpids` and
  clear them before restarting.
- **Long first-run startup:** Triton autotuning logs
  `out of resource: shared memory, Required: 122880, Hardware limit: 65536` on
  gfx1100's 64 KB LDS, retries with smaller configurations, and caches the
  result. Subsequent starts are faster.
- **`destroy_process_group()` segfaults at exit:** a known FlagCX-on-ROCm issue.
  Collectives complete correctly beforehand; only teardown is affected.

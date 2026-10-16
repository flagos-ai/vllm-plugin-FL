# T-Head PPU Deployment Guide

This guide runs vLLM with the FL plugin on T-Head PPU hardware using the
prepared runtime below. FlagGems is installed as a complete ordinary wheel from
the pinned upstream commit, which includes merged
[PR #6894](https://github.com/flagos-ai/FlagGems/pull/6894). The host driver must
match the vendor PPU runtime. For a custom image, preserve the vendor
PyTorch/SDK and follow section 3; ordinary CUDA wheels do not reproduce this stack.

## Software stack

The updated image uses the following stack. Acceptance of the complete upstream
FlagGems wheel is pending the T-Head CI run; earlier results are recorded
separately below.

| Component | Version or configuration |
|---|---|
| Accelerator | T-Head PPU-ZW810E; four visible cards for model tests |
| Python | `3.12.3` |
| PyTorch | Vendor PPU build, version `2.10.0` |
| vLLM | `0.28.0+empty`, source `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| FL plugin | CI installs the tested checkout as an ordinary wheel |
| FlagTree | `0.7.0+ppu3.6` |
| Imported Triton | `3.6.0`, supplied by the FlagTree PPU wheel |
| FlagGems | Complete upstream commit `317b763204391d530d6024b52b6235de150443d8`; wheel metadata `5.5.0.dev0+g317b76320` |
| Model execution | BF16, TP=4, eager and graph; maximum model length 32768, maximum sequences 8, memory utilization 0.85 |

The vLLM distribution version is `0.28.0+empty`, while its generated module
version (`vllm.__version__`) is `0.28.0`. The empty build suffix is added to
wheel metadata. Verify both versions and confirm that the imported module
belongs to that distribution; checking the module string for `+empty` rejects
this valid build.

The upstream commit contains the split-KV combine correction. Install its
complete wheel; no post-install source replacement is needed. Preserve the
source revision and wheel metadata with the image build record. The packaged
`flag_gems-5.5.0.dev0+g317b76320-py3-none-any.whl` has SHA256
`fdbbc9b6afa7b8ef8ca4bcdf832d69622162db9b067b1bec4ef8ef3779ed8a95`.

The T-Head policy is loaded from
[`vllm_fl/dispatch/config/thead.yaml`](../../vllm_fl/dispatch/config/thead.yaml).
Keep the plugin and this policy together. SiLU, RMSNorm, and rotary embedding
use FlagGems followed by the shared PyTorch reference backend. Only attention
has a T-Head vendor registration. Text paged attention uses the vendor
`flash_attn_3` bridge; the vision encoder also needs the corrected FlagGems
SDPA path. Passing text attention tests alone does not validate image inference.

## 1. Check the host and vendor runtime

Use the published runtime image:

```bash
export PPU_IMAGE="harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-thead-ci"
```

The published image digest is
`sha256:1ec87ed25bce8a523bd9a24f20695821363b385f7b58a878bb87df7bef58c414`.
The image includes vendor PyTorch, the PPU SDK and `flash_attn_3`, plus the
ordinary Python stack in the table.
The following versions were confirmed on 2026-10-14 on the same host and
container used for model acceptance. SDK/runtime versions come from the
container; VBIOS was checked on all four reserved host cards (12–15).

| Component | Recorded version |
|---|---|
| Host kernel | `5.10.134-013.13.4.kangaroo.al8.x86_64` |
| PPU driver | `1.3.2-d7f5a2` |
| PPU-SMI | `1.28` |
| HGGC | `13.0` |
| PPU SDK | `2.1.0-a5f865` |
| clang | `13.0.1 (2.1.0-a5f865)` |
| HGGC Driver / Runtime | `4.2` / `3.0` |
| PCCL | `2.1.0` |
| VBIOS (host cards 12–15) | `1.6.12-af69f0` |

These identify the recorded acceptance environment, not minimum supported
versions or a kernel/firmware compatibility range. The exact vendor base build
and PyTorch/FA3 build identities are recorded in section 3. Keep that SDK,
PyTorch backend, and FA3 build together; obtain the hardware provider's
compatibility matrix before using a different combination.

```bash
docker version
uname -r
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi
# Record the selected host card's details, including VBIOS; change its index as needed.
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi -q -i 12
ls -l /dev/alixpu /dev/alixpu_ctl /dev/alixpu_ppu*
```

Inside the inference container, record its SDK and runtime module versions:

```bash
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi -q -d VERSION
```

Use `ppu-smi` to check utilization, allocated memory, and owning processes.
Select cards that are free immediately before each experiment. Mounting a
card into a container does not reserve it. Keep long installs, builds, and
experiments in dedicated `tmux` sessions so that an SSH disconnect does not
terminate them; retain the session logs and exit status.

## 2. Mount devices, source, and models

`PPU_IMAGE` defaults to the versioned image tag without replacing an existing
selection. Set it to your own vendor image when following the self-build path.
`WORK_DIR` is persistent writable storage for environments, source builds,
caches, and results. `MODEL_DIR` contains model directories with `config.json`
and all weight shards.

```bash
export PPU_IMAGE="${PPU_IMAGE:-harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-thead-ci}"
export REPO_DIR="$PWD"
export WORK_DIR=/data/thead-work
export MODEL_DIR=/data/models
mkdir -p "$WORK_DIR"

device_args=()
for device_node in /dev/alixpu /dev/alixpu_ctl /dev/alixpu_ppu*; do
    test -c "$device_node" || exit 1
    device_args+=(--device "$device_node:$device_node:rw")
done

docker run -d --name vllm-fl-thead \
    "${device_args[@]}" \
    --read-only --cap-drop ALL --security-opt no-new-privileges \
    --shm-size 8g --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
    -v "$WORK_DIR:/work" \
    -v "$REPO_DIR:/workspace/vllm-plugin-FL:ro" \
    -v "$MODEL_DIR:/models:ro" \
    --entrypoint /bin/bash "$PPU_IMAGE" -lc 'sleep infinity'

docker exec -it vllm-fl-thead bash
```

This example makes all host PPU devices available and selects the cards later
with `CUDA_VISIBLE_DEVICES`. It keeps model weights read-only. The service
examples bind to container loopback, so run the Gate client inside the same
container. Exposing an API outside the container additionally requires an
appropriate bind address and Docker port mapping.

Keep `TMPDIR=/tmp` on a local filesystem. The runtime's Triton cache also
stores generated `.so` libraries under `/tmp/thead-runtime` by default, so its
filesystem must allow executable mappings. Specify `exec` for the `/tmp`
tmpfs, retaining `nosuid,nodev`; a `noexec` mount can fail with
`failed to map segment from shared object` when Triton loads a generated
library. If `THEAD_RUNTIME_ROOT` selects another cache mount, that mount must
also permit executable mappings. A writable bind mount can still use a
filesystem such as FUSE that does not support the Unix-domain sockets needed
by multiprocessing. Using such a mount for temporary sockets failed during
validation.

## 3. Prepare the Python stack without replacing vendor packages

For the published image, select its supplied ordinary environment and verify
the package hashes/providers:

```bash
export ENV_DIR=/opt/thead/venv
source /opt/thead/runtime-env.sh
"$ENV_DIR/bin/python" -I -B /opt/thead/verify-image.py --full
```

Then skip the remaining manual installation commands in this section and
continue with section 4. The image's `/opt/thead/bin/runtime-entrypoint`
initializes the same environment before executing a command.

The remaining commands are for preparing a custom vendor image. The validated
vendor base was
`egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun/inference-xpu-pytorch:26.04-v2.1.0-vllm0.23.0-torch2.10-cu130-20260710`.
Keep that vendor build's PyTorch backend and SDK together. The following build
identifiers distinguish its vendor packages from public wheels with the same
version strings:

| Vendor package | Tested build identity |
|---|---|
| PyTorch | `torch-2.10.0-cp312-cp312-linux_x86_64.whl`; wheel SHA256 `9583af61316841ba1b1f8c3eb3c5e8fd2102b05ef934c0263b3520581326c4ad` |
| PyTorch native extension | `torch/_C.cpython-312-x86_64-linux-gnu.so`; SHA256 `7cb47fd88a702f6d724a784dc1b08e6bb572cd785f1bc70091d62975ad0c7d17` |
| PPU backend extension | `acext==1.0.0`; `lib/libacext.so` SHA256 `8d1788d3a04fb3f40ff39eff08e2e41b17b47410ed0df1770bce84dcd5c433d8` |
| PPU FlashAttention | Distribution `flash-attn-3==3.0.0b1`; `flash_attn_3/_C.abi3.so` SHA256 `5543e2443de27f42bee8345ca1cb09a391ea4bc2fa4ad1d79058b13469a344c2` |

Use the prepared image digest in section 1 to reproduce these vendor artifacts,
or obtain the matching base image and packages from the hardware provider.

For a fresh installation, create a private environment that can read the
vendor's system packages. The path below is an example, not a required
installation location.

```bash
export ENV_DIR=/work/thead-env
python3 -m venv --system-site-packages --copies "$ENV_DIR"
source "$ENV_DIR/bin/activate"
unset PYTHONPATH PYTHONHOME
export TMPDIR=/tmp
```

Preserve vendor PyTorch, its backend extensions, SDK libraries, and the PPU
`flash_attn_3` package. Install ordinary Python packages into the private
environment. Use `--no-deps` for the pinned accelerator packages and source
builds; review missing Python dependencies separately. An unrestricted
`pip install -U` or installing the repository's test extras may pull in a
non-PPU PyTorch/Triton stack.

The official PPU FlagTree wheel for CPython 3.12 is:

```bash
python -m pip install --no-deps --ignore-installed \
  'https://resource.flagos.net/repository/flagos-pypi-hosted/packages/flagtree/0.7.0+ppu3.6/flagtree-0.7.0+ppu3.6-cp312-cp312-linux_x86_64.whl#sha256=98fa0c4488e06c904adfb55a8059ef64f2c1bddf5938f6faa7edfc8650dead09'
```

Its SHA256 is pinned in the URL. Follow the
[FlagTree PPU manual](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ppu)
for vendor prerequisites. Do not uninstall or overwrite a system Triton
package merely because its metadata is still visible: in an inherited
environment, the actual imported `triton.__file__` and `triton.__version__`
determine which compiler is used.

Build vLLM with the empty target from the pinned source, using the build
prerequisites already prepared for the vendor runtime. Install the desired
plugin checkout without a CUDA extension build. A CUDA-oriented
`VLLM_VENDOR=cuda` command from the NVIDIA guide is not the validated PPU
installation.

```bash
mkdir -p /work/sources
git clone https://github.com/vllm-project/vllm.git /work/sources/vllm
git -C /work/sources/vllm checkout --detach 2cf0a6915ce544dc493a0990f2ea38d81601128a
VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 VLLM_USE_PRECOMPILED_RUST=0 \
    VLLM_VENDOR= python -m pip install --no-build-isolation --no-deps /work/sources/vllm

# Build from a writable copy of the checkout being validated.
cp -a /workspace/vllm-plugin-FL /work/sources/vllm-plugin-FL
VLLM_VENDOR= python -m pip install --no-build-isolation --no-deps /work/sources/vllm-plugin-FL
```

Keep a record of your installed plugin revision and re-run the acceptance
matrix after changing it. Upstream vLLM's generic dependency declarations target
a different PyTorch version; they are not permission to replace the vendor PPU build. A successful
`--no-deps` install alone does not prove that all runtime dependencies or vendor
ABI requirements have been met.

### Install the complete upstream FlagGems wheel

Build the pinned upstream commit in a separate environment, then install its
ordinary wheel into the active runtime. Keep Git history and tags: the project
uses `setuptools-scm` to generate its version, so a source archive or a shallow
checkout without reachable tags cannot reproduce the version metadata.

```bash
git clone https://github.com/flagos-ai/FlagGems.git /work/sources/FlagGems
git -C /work/sources/FlagGems checkout --detach 317b763204391d530d6024b52b6235de150443d8
python3 -m venv /work/flaggems-build
/work/flaggems-build/bin/python -m pip install \
    'setuptools>=64,<77' 'setuptools-scm>=8,<10' 'wheel==0.46.2'
/work/flaggems-build/bin/python -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir /work/flaggems-wheels /work/sources/FlagGems
python -m pip install --no-deps --ignore-installed /work/flaggems-wheels/*.whl
python -c "from importlib.metadata import version; print(version('flag-gems'))"
```

Use a new wheel directory containing only this build's wheel. `--ignore-installed`
places the wheel in the private environment while preserving inherited system
packages. Do not use the `[thead]` extra or run `flaggems-setup`: this upstream
revision's extra pins an older FlagTree version. Keep the installed PPU
FlagTree `0.7.0+ppu3.6` and vendor PyTorch stack unchanged.

### Package a reusable runtime image

A distributable image must contain the ordinary installed Python packages,
including the complete FlagGems wheel, and the vendor runtime in its own
filesystem.
Install the plugin non-editably; an editable checkout or a `.pth` path into a
host-only source directory is not a self-contained deployment.

A plain `docker commit` does not capture files in mounted volumes, including a
bind-mounted `/work` environment. Copy the required installed package trees
into the image filesystem, resolve any inherited paths, and verify imports and
installed package integrity in a new container without the build workspace
mount before publishing the image. This follows the
[Docker commit behavior](https://docs.docker.com/reference/cli/docker/container/commit/).
Keep model weights in a separate read-only mount and use writable caches at
runtime. Record the resulting image digest with the software/source manifest.
The prepared image published for this stack is the digest in section 1; use
the packaging steps only when producing your own image.

The packaged runtime layout uses `/opt/thead/venv` for the merged ordinary
packages, `/opt/thead/runtime-env.sh` for environment selection, and
`/opt/thead/bin/runtime-entrypoint` to initialize that environment and execute
a command. Its `thead-vllm` wrapper invokes
`python -I -B -m vllm.entrypoints.cli.main`. Runtime caches default to
`/tmp/thead-runtime`; set `THEAD_RUNTIME_ROOT` to a writable cache mount when
persistence is needed. The runtime preserves the caller's device-selection
variables. After producing an image, run its CPU-only
`/opt/thead/verify-image.py --full` through the image Python to check package
hashes and actual providers before inference validation.

## 4. Select the runtime and free cards

Use one writable cache root per software stack and reuse it between serial
experiments. Do not reuse a different FlagGems/FlagTree stack's compilation
cache.

```bash
if [[ "$ENV_DIR" = /opt/thead/venv ]]; then
    source /opt/thead/runtime-env.sh
else
    source "$ENV_DIR/bin/activate"
fi
unset PYTHONPATH PYTHONHOME
export VLLM_PLUGINS=fl
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export GEMS_VENDOR=thead
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR=/tmp
export RUNTIME_DIR="${THEAD_RUNTIME_ROOT:-/work/thead-runtime}"
export XDG_CACHE_HOME="$RUNTIME_DIR/cache"
export TRITON_CACHE_DIR="$RUNTIME_DIR/triton"
export FLAGGEMS_CACHE_DIR="$RUNTIME_DIR/gems"
export HF_HOME="$RUNTIME_DIR/hf"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_DIR/inductor"
export VLLM_CACHE_ROOT="$RUNTIME_DIR/vllm"
export FLAGGEMS_ENABLE_OPLIST_PATH="$RUNTIME_DIR/flag-gems-enabled-ops.txt"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$FLAGGEMS_CACHE_DIR" \
    "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi
# Example only: replace these four physical IDs with currently idle cards.
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

The validated runs used `CUDA_VISIBLE_DEVICES`; avoid a conflicting inherited
`PPU_VISIBLE_DEVICES` restriction. Environment activation should preserve your
chosen visibility. Inside the process, the selected physical cards are
renumbered to logical devices 0 through 3.

Check imports after selecting free cards, using the interpreter that will
start the service:

```bash
python -I -B - <<'PY'
from importlib.metadata import version
import flag_gems
import torch
import triton
import vllm

assert torch.__version__ == '2.10.0'
assert version('flagtree') == '0.7.0+ppu3.6'
assert triton.__version__ == '3.6.0'
assert vllm.__version__.startswith('0.28.0')
for module in (torch, triton, flag_gems, vllm):
    print(module.__name__, getattr(module, '__version__', None), module.__file__)
print('FlagGems distribution:', version('flag-gems'))
print('visible devices:', torch.cuda.device_count())
PY
```

Confirm that Torch is still the vendor package, Triton comes from the private
FlagTree installation, and FlagGems resolves to the installed ordinary wheel.
Check its distribution version against the image's build record.

## 5. Start eager and graph services serially

Run long commands through `tmux` on the Docker host. For example, save the
service command below as `/work/serve-thead.sh` inside the container, including
environment activation and the exports from step 4. Then, on the host:

```bash
tmux new-session -d -s thead-serve \
    'docker exec vllm-fl-thead bash /work/serve-thead.sh > /data/thead-work/thead-serve.log 2>&1'
```

Adjust the host log path to `WORK_DIR`. This does not require tmux to be
installed in the container.

```bash
export MODEL_PATH=/models/Qwen3.8-27B
export SERVED_MODEL_NAME="$MODEL_PATH"
export PORT=8000
export GATE_DIR=/workspace/vllm-plugin-FL/tools/adaptation-gate-cases

python -I -B -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 4 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --safetensors-load-strategy prefetch \
    --allowed-local-media-path "$GATE_DIR/images" \
    --trust-remote-code --enforce-eager
```

For the graph run, stop the owned eager service, confirm that its workers have
exited and the selected cards are idle, then remove `--enforce-eager` and add
`--cudagraph-metrics`. Repeat both modes for `/models/Qwen3.6-35B-A3B`. Do not run
multiple model services on the same selected cards during this matrix.

Use `python -I -B -m vllm.entrypoints.cli.main`, including when the environment
inherits packages and has no `bin/vllm` console script. This vLLM source does not
provide a `python -m vllm` entry point.

The first startup/request can take minutes for kernel autotuning, compilation,
and graph capture. Follow logs and wait for `/v1/models`; model loading alone
is not readiness. Retain complete logs instead of truncating autotuning output.

## 6. Run unit tests and the original adaptation Gate

Stop model services before GPU unit tests. Select an idle card and run the
checked-in T-Head cache and attention tests through the same interpreter:

```bash
cd /workspace/vllm-plugin-FL
python -I -B -m pytest -o addopts= -p no:cacheprovider -ra \
    --basetemp=/tmp/thead-unit-tests \
    tests/unit_tests/dispatch/test_thead_cache.py \
    tests/unit_tests/dispatch/test_thead_attention.py
```

A missing PPU vendor wheel or a skipped GPU test is not an acceptance pass.
The two files above cover 5 CPU cache cases and 45 GPU cache/attention cases.
The CI unit runner also includes the shared reference operator tests, as
described below.

For each running service, use the
[adaptation Gate cases](../../tools/adaptation-gate-cases/README.md). Set
`MODEL_PATH`, `SERVED_MODEL_NAME`, and `PORT` to the service values in a second
tmux session. Use a different `RESULTS_DIR` for every model/mode. The original
pytest modules can be run directly with the active interpreter, including in
an environment without a local `pytest` executable:

```bash
cd /workspace/vllm-plugin-FL/tools/adaptation-gate-cases
export MODEL_PATH=/models/Qwen3.8-27B
export SERVED_MODEL_NAME="$MODEL_PATH"
export PORT=8000
export BASE_URL="http://127.0.0.1:$PORT/v1"
export REQUEST_TIMEOUT=300
export RESULTS_DIR=/work/results/qwen27b-eager
curl --fail "$BASE_URL/models"

status=0
for test_file in test_text.py test_image.py test_mix_text_image.py; do
    python -I -B -m pytest -o addopts= -p no:cacheprovider -sv "$test_file" || status=1
done
# In a test runner script, finish with: exit "$status"
```

The packaged `run_test.sh` also waits for readiness and runs all three files;
use it when its `pytest` executable belongs to the intended environment.
Each model/mode sends 26 requests: 1 text, 8 concurrent text, 1 image,
8 concurrent image, and 8 mixed requests. It evaluates 260 response checks.
The workflow runs the shared Gate files without changing their prompts,
images, keyword checks, or thresholds. The detailed LLM introduction must
contain at least 256 characters and the required terms in order; the other
requests retain their exact answer/OCR checks. All ten checks must pass for
every response, including repetition, encoding, and control-character checks.

## CI configuration

The reusable workflow is [`.github/workflows/_thead_test.yml`](../../.github/workflows/_thead_test.yml).
It uses the same configuration loader as the other platforms, with
[the T-Head configuration](../../.github/configs/thead.yml). Use the published
versioned image below and reserve four PPU cards on a registered runner. Device mounts
grant access; they do not reserve cards or prove that the cards are idle.

The versioned THead configuration supplies the approved image, runner, four
reserved cards, model mount, and image Python. This also supports PR runs that
receive empty repository variables. Maintainers may override these defaults
with the following repository variables; defaults and overrides undergo the
same validation:

| Variable | Value |
|---|---|
| `THEAD_CI_IMAGE` | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-thead-ci` |
| `THEAD_CI_VISIBLE_DEVICES` | Exactly four distinct, exclusively reserved PPU indices, comma-separated |
| `THEAD_CI_MODEL_27B` | Absolute container path to the complete `Qwen3.8-27B` model |
| `THEAD_CI_MODEL_35B` | Absolute container path to the complete `Qwen3.6-35B-A3B` model |
| `THEAD_CI_BASE_PYTHON` | Python in the image's ordinary environment; `/opt/thead/venv/bin/python` for the packaged runtime layout |
| `THEAD_CI_RUNNER_LABELS` | Optional JSON array overriding the configured runner labels |
| `THEAD_CI_CONTAINER_VOLUMES` | JSON array of Docker volume specifications; mount the two models read-only |
| `THEAD_CI_CONTAINER_OPTIONS` | Optional Docker options overriding the explicit device grants and shared-memory settings |

The published image provides `/opt/thead/venv` and the complete ordinary
stack. Ensure that the runner can pull this image and access the
configured model mounts. Keep any registry authentication in the runner/CI
secret configuration. The workflow accepts explicit v-version tags or SHA256
digests and rejects missing or invalid image references, invalid
card selection, and privileged container options, including invalid explicit
overrides instead of silently using defaults.

The image must contain vendor PyTorch 2.10.0 and SDK/extensions, vLLM
0.28.0+empty, FlagTree 0.7.0+ppu3.6, the complete FlagGems wheel from
`317b763204391d530d6024b52b6235de150443d8`, and runtime/test/build dependencies.
The [setup script](../../.github/scripts/thead/setup.sh) creates a fresh child
venv, inherits the image-owned package directory, and builds/installs this
checkout as an ordinary plugin wheel with `--no-deps --no-build-isolation`.
It does not download accelerator packages or replace the vendor stack.
[The check script](../../.github/scripts/thead/check.sh) validates installed
package identities, providers, and package integrity against the runtime
snapshot before and after tests.

To exercise the same scripts manually inside a prepared image, keep setup
and tests in one Bash session. Set `CUDA_VISIBLE_DEVICES` to the reserved
cards first. Each output directory must be new:

```bash
cd /workspace/vllm-plugin-FL
export THEAD_BASE_PYTHON=/opt/thead/venv/bin/python
source .github/scripts/thead/setup.sh
bash .github/scripts/thead/check.sh
RESULT_DIR="$(mktemp -d /tmp/thead-ci-results.XXXXXXXX)"
"$THEAD_CI_PYTHON" -I -B .github/scripts/thead/run_tests.py \
    --output-dir "$RESULT_DIR/unit"
"$THEAD_CI_PYTHON" -I -B .github/scripts/thead/run_gate.py \
    --model /models/Qwen3.8-27B --mode eager --tensor-parallel-size 4 \
    --output-dir "$RESULT_DIR/qwen27b-eager"
```

Run `graph` next, then repeat `eager` and `graph` for
`/models/Qwen3.6-35B-A3B`. The workflow runs those four Gates serially. Each
Gate starts its own server, verifies readiness, executes the original Gate
files, records all 26 requests/260 checks, and cleans up its own process group
before the next model/mode. Graph mode also requires graph-execution evidence.

The unit runner uses the existing
[`tests/unit_tests/ops/test_numerical.py`](../../tests/unit_tests/ops/test_numerical.py)
suite for shared reference operators, plus 5 CPU cache tests and 45 native PPU
cache/attention tests, with zero skips. Reference tensors run on CPU using the
normal package imports and the test suite's existing device fixture. Retain the
JUnit reports, per-request JSON results, source/provider snapshots, and logs.

## Recorded acceptance

Acceptance of the complete upstream FlagGems wheel at `317b7632` is pending
T-Head CI. The following results belong to the earlier repaired runtime,
with FlagGems metadata `5.4.0rc2.post1+gb4b37a751` and image digest
`sha256:fc75bd7811813ac824773f4719a76821e749f5f30ecb3f22fe7fc07bb7a107bb`.
The earlier CI run passed all 93 unit tests (43 shared reference, 5 CPU cache,
and 45 PPU cache/attention cases) and all four model Gate modes. These historical
results do not validate the updated wheel.

That earlier runtime passed the serial TP4 matrix on 2026-10-09 with the
original shared Gate checks:

| Model | Eager requests | Graph requests |
|---|---:|---:|
| `Qwen3.8-27B` | 26/26 | 26/26 |
| `Qwen3.6-35B-A3B` | 26/26 | 26/26 |

All 104 requests and 1040 response checks passed. The available dense model was
`Qwen3.8-27B`; this result does not claim that the Gate's originally named
`Qwen3.6-27B` weights were tested. These are results for the stated software,
models, and TP4 configuration, not a blanket result for every PPU/model setup.

## Troubleshooting

- **Images produce wrong answers or repeated punctuation:** verify that the
  installed FlagGems wheel includes upstream commit `317b7632`. The diagnosed
  bug in split-KV combine treated scratch
  rows flattened as `[batch, head, query]` as if they had the output tensor's
  storage order. The fix decodes all three indices, uses the real output
  strides, and passes the head stride rather than the channel stride. See
  [FlagGems #6872](https://github.com/flagos-ai/FlagGems/issues/6872) and
  [the repair](https://github.com/flagos-ai/FlagGems/pull/6894). The same D72 SDPA
  inputs changed from four failing profiles to four passing profiles without
  changing the numerical tolerances.
- **Torch or Triton import changes after installation:** inspect actual module
  paths as well as distribution metadata. Preserve vendor Torch and verify that
  the intended FlagTree wheel supplies the imported compiler.
- **Socket creation or multiprocessing startup fails on a mounted path:** set
  `TMPDIR=/tmp`, not a FUSE-backed workspace directory.
- **Cold compilation fails with `Resource temporarily unavailable` or a
  `log_file` `NameError`:** on the validated 176-CPU host, a container PID/thread
  limit of 512 made `ppu-llc` fail to create threads (`EAGAIN`). FlagTree's error
  handler then referenced an undefined `log_file`, hiding the compiler error.
  Removing that added limit restored the original nine-call D72 SDPA smoke
  test without changing package bytes or numerical tolerances. Size the thread
  budget for the host and concurrent compilation; the baseline does not impose
  a low PID limit. Keep the other container isolation settings. This diagnosis
  does not claim that FlagTree's error handler has been fixed.
- **GPU tests skip or attention imports fail:** verify device mounts, visibility,
  vendor SDK, and the PPU `flash_attn_3` package. Standard NVIDIA FA3 is not a
  substitute for the PPU implementation.
- **Out of memory:** inspect competing jobs first. Increase tensor parallelism
  only where the model supports it, or lower memory/sequence limits and record
  the new settings. Re-run both modes rather than presenting a changed setup as
  the TP4 acceptance result above.

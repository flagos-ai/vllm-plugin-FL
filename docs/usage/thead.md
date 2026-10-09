# T-Head PPU Deployment Guide

This guide runs vLLM with the FL plugin on T-Head PPU hardware using the
published prepared runtime below. It contains the pinned ordinary Python stack
and repaired FlagGems sources. The host driver must still match its vendor PPU
runtime. For a custom image, preserve the vendor PyTorch/SDK and follow the
self-build path in section 3; ordinary CUDA wheels do not reproduce this stack.

## Validated software stack

The following stack passed the T-Head acceptance matrix on 2026-10-09.

| Component | Validated version or configuration |
|---|---|
| Accelerator | T-Head PPU-ZW810E; four visible cards for model tests |
| Python | `3.12.3` |
| PyTorch | Vendor PPU build, version `2.10.0` |
| vLLM | `0.28.0+empty`, source `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| FL plugin | Installed source `a0752c3a70158df1e6de953eae33d4338fe193a4`; metadata `0.0.0+ga0752c3a7` |
| FlagTree | `0.7.0+ppu3.6` |
| Imported Triton | `3.6.0`, supplied by the FlagTree PPU wheel |
| FlagGems base | `b4b37a7518d511548ed562277e1972a0b5d9986d`; metadata `5.4.0rc2.post1+gb4b37a751` |
| Required FlagGems fix | Three production files from [PR #6894](https://github.com/flagos-ai/FlagGems/pull/6894), fixed head `4e330227564ccc7f6a373be2d48a5d2cb7f2823e`, applied to the base above |
| Model execution | BF16, TP=4, eager and graph; maximum model length 32768, maximum sequences 8, memory utilization 0.85 |

The vLLM distribution version is `0.28.0+empty`, while its generated module
version (`vllm.__version__`) is `0.28.0`. The empty build suffix is added to
wheel metadata. Verify both versions and confirm that the imported module
belongs to that distribution; checking the module string for `+empty` rejects
this valid build.

The tested FlagGems code is **the pinned base plus the three-file fix**.
Checking out the entire PR head, installing a moving `master`, or relying on
package version strings alone does not identify the tested code. The repaired
installation retains its base distribution metadata and records the changed
source hashes separately.

The T-Head policy is loaded from
[`vllm_fl/dispatch/config/thead.yaml`](../../vllm_fl/dispatch/config/thead.yaml).
Keep the plugin and this policy together. Text paged attention uses the vendor
`flash_attn_3` bridge; the vision encoder also needs the corrected FlagGems
SDPA path. Passing text attention tests alone does not validate image inference.

## 1. Check the host and vendor runtime

Use the published runtime by immutable digest:

```bash
export PPU_IMAGE="harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:fc75bd7811813ac824773f4719a76821e749f5f30ecb3f22fe7fc07bb7a107bb"
```

The corresponding tag is `v0.28.0-thead-ci-20261009`. The image includes vendor
PyTorch, the PPU SDK and `flash_attn_3`, plus the repaired stack in the table.
Obtain matching host driver instructions from the hardware provider.

The packaged payload passed normal imports and a nine-call numerical smoke in
a separate container without the old build-workspace mounts. The published
digest and anonymous registry metadata were verified. GPU execution after
pulling this final digest has not yet been validated; the CI job checks that
artifact on the configured PPU runner.

```bash
docker version
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi
ls -l /dev/alixpu /dev/alixpu_ctl /dev/alixpu_ppu*
```

Use `ppu-smi` to check utilization, allocated memory, and owning processes.
Select cards that are free immediately before each experiment. Mounting a
card into a container does not reserve it. Keep long installs, builds, and
experiments in dedicated `tmux` sessions so that an SSH disconnect does not
terminate them; retain the session logs and exit status.

## 2. Mount devices, source, and models

`PPU_IMAGE` defaults to the published digest without replacing an existing
selection. Set it to your own vendor image when following the self-build path.
`WORK_DIR` is persistent writable storage for environments, source builds,
caches, and results. `MODEL_DIR` contains model directories with `config.json`
and all weight shards.

```bash
export PPU_IMAGE="${PPU_IMAGE:-harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:fc75bd7811813ac824773f4719a76821e749f5f30ecb3f22fe7fc07bb7a107bb}"
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

The remaining commands are for preparing a custom vendor image. Rebuild only
when creating a new environment; the repair script intentionally rejects
already changed preimages.

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

The test record used the plugin commit in the software table. Keep a record of
your installed plugin revision and re-run the acceptance matrix after changing
it. Upstream vLLM's generic dependency declarations target a different PyTorch
version; they are not permission to replace the vendor PPU build. A successful
`--no-deps` install alone does not prove that all runtime dependencies or vendor
ABI requirements have been met.

### Apply the pinned FlagGems repair

Build and install FlagGems from the base commit in a private environment. Keep
a pristine copy before applying the fix. Apply only these production files
from the fixed PR head, with their preimage and postimage checksums verified:

| Relative path under `flag_gems/` | Repaired SHA256 |
|---|---|
| `ops/flash_api.py` | `bd6e3daec6af0d72f5eec25168d1e09cbd7c8ee577896086f6650c800c782485` |
| `ops/flash_kernel.py` | `dfb99850150f9c734373247ea005ed3957109347dbc1ce9f5472b3eef8ecb850` |
| `runtime/backend/_enflame/gcu400/ops/flash_api.py` | `52a2d999ef981349542229bf599c9003740bbb63524302bec9e27dccddd4dce6` |

Use a separate build environment for the ordinary FlagGems wheel, then install
it into the active private runtime. The wheel directory below should be new
and contain only this build's wheel.

```bash
git clone https://github.com/flagos-ai/FlagGems.git /work/sources/FlagGems
git -C /work/sources/FlagGems checkout --detach b4b37a7518d511548ed562277e1972a0b5d9986d
python3 -m venv /work/flaggems-build
/work/flaggems-build/bin/python -m pip install \
    'setuptools>=64,<77' 'setuptools-scm>=8,<10' 'wheel==0.46.2'
/work/flaggems-build/bin/python -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir /work/flaggems-wheels /work/sources/FlagGems
python -m pip install --no-deps --ignore-installed /work/flaggems-wheels/*.whl
```

Apply the three pinned source changes after the base wheel is installed. This
checks every preimage and downloaded postimage before writing; it saves the
original files and a manifest outside `site-packages`. It fails if the active
FlagGems package is inherited from outside the private environment.

```bash
python -I -B - <<'PY'
from pathlib import Path
from importlib.util import find_spec
import hashlib
import json
import sys
import urllib.request

head = '4e330227564ccc7f6a373be2d48a5d2cb7f2823e'
files = {
    'ops/flash_api.py': (
        '9d6f1e1ec802d132b074a618c173a9ea424f4d5793fd0cbcead687796155a70f',
        'bd6e3daec6af0d72f5eec25168d1e09cbd7c8ee577896086f6650c800c782485'),
    'ops/flash_kernel.py': (
        'eef0202e6c0ed08bef8098adede1a603c63354c54b54a19414e210b675157265',
        'dfb99850150f9c734373247ea005ed3957109347dbc1ce9f5472b3eef8ecb850'),
    'runtime/backend/_enflame/gcu400/ops/flash_api.py': (
        'df6d4e554b04f6bf06a57346ac5c613f07db2fbf3703328c66920abfe68c39ec',
        '52a2d999ef981349542229bf599c9003740bbb63524302bec9e27dccddd4dce6'),
}
root = Path(find_spec('flag_gems').origin).resolve().parent
assert root.is_relative_to(Path(sys.prefix).resolve()), root
prepared = []
for name, (before_sha, after_sha) in files.items():
    path = root / name
    before = path.read_bytes()
    assert hashlib.sha256(before).hexdigest() == before_sha, name
    url = f'https://raw.githubusercontent.com/flagos-ai/FlagGems/{head}/src/flag_gems/{name}'
    with urllib.request.urlopen(url, timeout=60) as response:
        after = response.read(131073)
    assert hashlib.sha256(after).hexdigest() == after_sha, name
    prepared.append((name, before, after))
backup = Path('/work/flaggems-source-repair')
backup.mkdir(exist_ok=False)
manifest = {'base_commit': 'b4b37a7518d511548ed562277e1972a0b5d9986d',
            'fix_head': head, 'files': []}
for name, before, after in prepared:
    original = backup / name
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(before)
    (root / name).write_bytes(after)
    assert (root / name).read_bytes() == after
    manifest['files'].append({'path': name, 'before_sha256': files[name][0],
                              'after_sha256': files[name][1]})
(backup / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
```

The third file keeps the other caller compatible with the shared kernel's new
arguments; its presence does not mean GCU was tested. These are Python/Triton
sources, so this repair does not require rebuilding vendor Torch or the PPU
SDK. Retain the base commit, patch, and source-hash manifest with the environment.

### Package a reusable runtime image

A distributable image must contain the ordinary installed Python packages,
vendor runtime, and the repaired FlagGems sources in its own filesystem.
Install the plugin non-editably; an editable checkout or a `.pth` path into a
host-only source directory is not a self-contained deployment.

A plain `docker commit` does not capture files in mounted volumes, including a
bind-mounted `/work` environment. Copy the required installed package trees
into the image filesystem, resolve any inherited paths, and verify imports and
patch hashes in a new container without the build workspace mount before
publishing the image. This follows the
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
FlagTree installation, and FlagGems resolves to the repaired ordinary package.
Also verify the three source hashes above. Metadata can remain unchanged when
Python source is patched.

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
The full recorded unit result was 74/74: 5 CPU cases, 36 GPU cache cases,
9 attention cases, and an additional 24-case FA3 numerical oracle. The two
pytest files above cover the first 50 cases. Additional regressions passed
72 packed-GDN calls and 36 public vision SDPA calls across four layout/head
profiles; these counts are separate from model requests.

For each running service, use the unchanged
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
Keep every original semantic, OCR, length, repetition, and encoding check.

## CI configuration

The reusable workflow is [`.github/workflows/_thead_test.yml`](../../.github/workflows/_thead_test.yml).
It uses the same configuration loader as the other platforms, with
[the T-Head configuration](../../.github/configs/thead.yml). Use the published
digest below and reserve four PPU cards on a registered runner. Device mounts
grant access; they do not reserve cards or prove that the cards are idle.

The versioned THead configuration supplies the approved image, runner, four
reserved cards, model mount, and image Python. This also supports PR runs that
receive empty repository variables. Maintainers may override these defaults
with the following repository variables; defaults and overrides undergo the
same validation:

| Variable | Value |
|---|---|
| `THEAD_CI_IMAGE` | `harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:fc75bd7811813ac824773f4719a76821e749f5f30ecb3f22fe7fc07bb7a107bb` |
| `THEAD_CI_VISIBLE_DEVICES` | Exactly four distinct, exclusively reserved PPU indices, comma-separated |
| `THEAD_CI_MODEL_27B` | Absolute container path to the complete `Qwen3.8-27B` model |
| `THEAD_CI_MODEL_35B` | Absolute container path to the complete `Qwen3.6-35B-A3B` model |
| `THEAD_CI_BASE_PYTHON` | Python in the image's ordinary environment; `/opt/thead/venv/bin/python` for the packaged runtime layout |
| `THEAD_CI_RUNNER_LABELS` | Optional JSON array overriding the configured runner labels |
| `THEAD_CI_CONTAINER_VOLUMES` | JSON array of Docker volume specifications; mount the two models read-only |
| `THEAD_CI_CONTAINER_OPTIONS` | Optional Docker options overriding the explicit device grants and shared-memory settings |

The published image provides `/opt/thead/venv` and the complete repaired
ordinary stack. Ensure that the runner can pull this digest and access the
configured model mounts. Keep any registry authentication in the runner/CI
secret configuration. The workflow rejects a missing image digest, invalid
card selection, and privileged container options, including invalid explicit
overrides instead of silently using defaults.

The image must contain the complete repaired ordinary stack: vendor PyTorch
2.10.0 and SDK/extensions, vLLM 0.28.0+empty, FlagTree 0.7.0+ppu3.6, the pinned
FlagGems base plus the three-file fix, and runtime/test/build dependencies.
The [setup script](../../.github/scripts/thead/setup.sh) creates a fresh child
venv, inherits the image-owned package directory, and builds/installs this
checkout as an ordinary plugin wheel with `--no-deps --no-build-isolation`.
It does not download accelerator packages or replace the vendor stack.
[The check script](../../.github/scripts/thead/check.sh) validates actual
providers and patch bytes against the vendor snapshot before and after tests.

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

The unit runner requires 202 isolated backend CPU tests, 5 CPU cache tests,
and 45 native PPU cache/attention tests, with zero skips. The isolated backend
CPU fixture is separate from the normal provider/import check. The manual
24-case FA3 oracle and 72 packed-GDN calls are additional numerical probes;
the CI unit runner does not claim that coverage. Retain its JUnit, per-request
JSON results, source/provider snapshots, and complete logs.

## Recorded acceptance

The exact repaired stack above passed the following serial TP4 matrix:

| Model | Eager requests | Graph requests |
|---|---:|---:|
| `Qwen3.8-27B` | 26/26 | 26/26 |
| `Qwen3.6-35B-A3B` | 26/26 | 26/26 |

All 104 requests and 1040 response checks passed. The available dense model was
`Qwen3.8-27B`; this result does not claim that the Gate's originally named
`Qwen3.6-27B` weights were tested. These are results for the stated software,
models, and TP4 configuration, not a blanket result for every PPU/model setup.

## Troubleshooting

- **Images produce wrong answers or repeated punctuation:** check the actual
  FlagGems source hashes. The diagnosed bug in split-KV combine treated scratch
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

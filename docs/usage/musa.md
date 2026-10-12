# MUSA deployment and adaptation Gate

This branch uses vLLM **0.28.0**. Build the upstream `empty` wheel and the FL
plugin as ordinary wheels in a private environment that inherits the MUSA
image's vendor packages. The older MUSA guide using vLLM 0.24.0 describes the
same installation pattern; its vLLM version and image are separate baselines.

## Ordinary-stack image and CI dispatch

The new image recipe is [docker/musa](../../docker/musa/README.md). It installs
exactly six verified ordinary wheels: empty vLLM 0.28.0, the FL plugin, the
Gate observer, the corrected FlagGems profile, the patched MUSA FlagTree profile
and PyYAML 6.0.1. The interpreter is `/opt/musa/venv/bin/python`; the image keeps
the wheels under `/opt/musa/wheelhouse`, the six-wheel stack manifest at
`/opt/musa/stack-manifest.json`, and the three-dependency manifest at
`/opt/musa/dependency-manifest.json`. Vendor Torch, TorchMUSA and SDK files retain
their protected fingerprints. No editable install or source checkout enters
the new layer; validation rejects active editable distributions, egg-links
and editable `.pth` hooks.

The new FL and observer ordinary source builds completed from committed source
`c35aade3c23e59533260578081a8161c7fb88179`, with actual versions
`0.0.0+gc35aade3c` and `0.1.0`. The ordinary image was published as
`harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-musa-ci-20261011-c35aade3c23e-p03`.
Its immutable RepoDigest is
`harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06`.
The normal Docker push completed successfully, and the published manifest/config
matched the accepted local image `sha256:c9c56cb3553eefab5657074d66c6ce288ae56f63e92f00c4af70da14aad3c9e4`
with the same embedded source and six-wheel stack. Ordinary-stack CPU installation
checks passed. MUSA is enabled in the existing Job 4e and uses this image for
PR CI. Full new-image GPU acceptance depends on the actual suite and original
Gate results; publication and CPU checks alone do not establish it.

The completed [9484c6a attempt](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38264910638/job/114865271015)
passed unit, functional, both existing E2E groups and benchmark smoke. Both
original eager Gate routes passed their five scenarios (52 logical requests
in total), and all four routes confirmed owned-process cleanup. Both graph
servers exited before readiness because Dynamo could not trace the rotary
layout patch's `storage_offset()` query. The missing accepted observer summary
was a consequence of the incomplete graph routes. Full Gate acceptance remains
failed for this attempt; no original assertion or graph requirement was relaxed.

The completed [8224362 attempt](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38242389032/job/114788780567)
passed unit, functional and both existing E2E groups, including the original
35B TP4 text and JPEG checks with the unchanged image oracle. Benchmark failed
at startup: the CUDA V2 model runner called `torch.cuda.Stream(musa:0)` and
raised `ValueError: Expected a cuda device, but got: musa:0`.

The 27B eager Gate route passed preflight, model readiness and all five
scenarios (26 requests; three pytest commands exited 0). During owned cleanup,
reading a process's `ns/mnt` returned `ENOENT`; an immediate PID recheck did
not confirm exit, so the cleanup barrier blocked the remaining three routes.
This shutdown sampling race's exact exit ordering remains unknown. Final
container cleanup succeeded, but that does not establish Gate cleanup success.
The Gate and full-suite acceptance remain failed for this attempt.

The earlier [da92963 attempt](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38232767793/job/114756829319)
passed unit (763 passed, 42 skipped), functional and 27B inference. The 35B
JPEG response stopped after 195 completion tokens but missed `Hello VLM` and
`blue rectangle`; E2E failed and benchmark was skipped. All four original Gate
routes and container cleanup passed. Full suite acceptance remains failed.

The MUSA rotary fix materializes non-contiguous offset half views before
broadcast multiplication. An isolated run with the same image and CI wheel
plus this single-file change passed the original 35B TP4 text and JPEG checks;
the image answer included both required phrases and stopped after 142 tokens.
All five owned children were reaped without forced kills. The compiler cause
remains unknown; this isolated result does not establish full-suite CI acceptance.

The earlier [71db9a7 attempt](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38221809408/job/114721763129)
passed unit, functional and 27B inference but failed the strict 35B JPEG check.
Its 27B eager Gate quality checks passed before an owned-cleanup `ns/mnt`
process-exit race blocked the remaining three routes. It did not pass the Gate.
The cleanup fix passed two actual Linux exit-race cases, six rejection controls
and CPU regressions.

The [hardware CI record](musa-ci-gate-runner.md#ordinary-image-and-explicit-ci-dispatch)
now includes the [e78c890 hardware attempt](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38205773888/job/114669187190):
unit and functional stages passed, as did 27B inference. The four-route original
Gate passed all 104 logical outcomes and 1040 quality checks, with 20 passing
JUnit cases. The 35B serving text check passed, but its image response failed the
unchanged oracle; benchmark was skipped and container cleanup succeeded. The
complete suite and full new-image acceptance remain failed for this attempt.

The `.github/configs/musa.yml` default selects this immutable RepoDigest;
`MUSA_CI_IMAGE` can select a separately validated image. Its software defaults are:

| Variable | Default |
|---|---|
| `MUSA_CI_EMPTY_WHEEL` | `/opt/musa/wheelhouse/vllm-0.28.0+empty-cp310-cp310-linux_x86_64.whl` |
| `MUSA_CI_EMPTY_WHEEL_SHA256` | `2306c2ea119182a5690eb05776cdffa4ba097a92ba13114a6cb49722d0b53fec` |
| `MUSA_CI_DEPENDENCY_PROFILE` | `latest` |
| `MUSA_CI_DEPENDENCY_MANIFEST` | `/opt/musa/dependency-manifest.json` |
| `MUSA_CI_DEPENDENCY_MANIFEST_SHA256` | `cd5b1e7a1c232c60df49c1384278a924e17e2a0bc71f9546fe4f8516ad9d5cf2` |

The manifest SHA covers the whole JSON file. Software artifacts and the
four-card runner defaults are resolved by the existing platform setup.
Repository variables or `musa_allocation` can override the defaults.
The image's build commit describes its embedded FL/observer wheels. CI setup
rebuilds both as ordinary wheels from the actual checked-out CI commit and
creates a fresh venv, rather than accepting an older embedded plugin as the
current candidate. Because a nested `--system-site-packages` venv does not
inherit `/opt/musa/venv`, setup explicitly installs the three pinned image
dependencies again from `/opt/musa/wheelhouse`. Wheel hashes, complete `RECORD` files,
module origins and the protected vendor inputs are checked in that venv.

The CI entry is `.github/workflows/ci.yml`, using the same reusable platform
workflow as main. Automatic MUSA discovery is enabled for PR CI. An explicit
dispatch can select only MUSA; read-only inventory remains available for
diagnostics on a committed ref:

```bash
gh workflow run ci.yml --repo flagos-ai/vllm-plugin-FL --ref "$COMMITTED_REF" \
  -f platform=musa -F musa_probe_only=true
```

The `musa-runner-probe` artifact contains sanitized runtime names, PCI identities,
device/memory observations and bounded device-FD counts; it does not create a
GPU container or run Torch, inference or model checks. Missing access, an
incomplete scan or a queued probe does not establish an idle allocation. The
normal lint/build jobs still have their own results, so an inventory success
alone cannot certify the complete workflow.

The inventory-only
[run 38108902958](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38108902958)
completed successfully at `2b6bee63ad1b852021a9e166195aa3fac75443f1`:
lint (Ruff check/format, ShellCheck and Typos), ordinary package build,
discovery and the read-only runner probe all passed. `test-musa` was skipped
by the inventory-only selection; no full MUSA GPU CI result is claimed. The earlier
[run 38108076987](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38108076987)
retains its original queued/failed stages.

Normal PR CI selects MUSA through `.github/configs/platforms.yml`. The registered
`mt-cicd-vllm-plugin` runner defaults to Docker runtime `mthreads`, physical
devices `4,5,6,7`, logical devices `0,1,2,3`, and PCI IDs
`0000:83:00.0,0000:a3:00.0,0000:c3:00.0,0000:e3:00.0`. The model defaults are
`/data/models/Qwen/Qwen3.6-27B` and `/data/models/Qwen/Qwen3.6-35B-A3B`,
matching the existing CI model catalog. Before Torch
initialization, the executing container verifies the four device nodes and
requires a complete successful vendor query with zero selected memory use
and utilization. Busy or unknown resources fail this run before GPU tests.
This admission snapshot is not an exclusive scheduler lease.

A direct full run also works without an allocation input:

```bash
gh workflow run ci.yml --repo flagos-ai/vllm-plugin-FL --ref "$COMMITTED_REF" \
  -f platform=musa -F musa_probe_only=false
```

To override the runner defaults, supply a reviewed allocation JSON with
exactly four physical
device ordinals, logical devices `0,1,2,3`, four canonical PCI IDs in that
logical order and the installed Docker runtime name. Optional keys are a list
of runner labels and the two `/data/` model paths. The allowed field names are
`runtime`, `physical_devices`, `logical_devices`, `pci_bus_ids`, `runner_labels`,
`model_27b` and `model_35b_a3b`; duplicate or unknown fields are rejected. The
allocation applies to this dispatch and does not rewrite shared repository
variables. Overrides still undergo the executing job's device and idle checks.

The existing allocation keys `model_27b` and `model_35b_a3b` set
`MUSA_GATE_MODEL_27B` and `MUSA_GATE_MODEL_35B_A3B`, so common MUSA E2E
and the original Gate use the same model source paths. If those environment
variables are absent, the common catalog paths remain the defaults.

```bash
gh workflow run ci.yml --repo flagos-ai/vllm-plugin-FL --ref "$COMMITTED_REF" \
  -f platform=musa -f musa_allocation="$(cat "$ALLOCATION_JSON")" \
  -F musa_probe_only=false
gh run watch "$RUN_ID" --repo flagos-ai/vllm-plugin-FL --exit-status
```

The full platform workflow uses one MUSA runner and one container for unit,
functional, serial e2e, benchmark and the original adaptation Gate. Its native
PCI check verifies the actual four-card mapping before the test stages; the
allocation must belong to this executing runner, rather than an earlier
ephemeral probe. Workflow concurrency serializes these MUSA runs, while the
executing container rejects busy or unknown selected cards before initialization.
Keep the final workflow
URL, actual checkout commit, image RepoDigest, setup evidence and complete
suite/Gate artifact. A workflow is qualified only by its actual jobs and unchanged
four-route Gate: 104 requests, 1040 checks, 20 JUnit cases, 32 complete streams
and 52/208 graph joins. Probe-only, ordinary build or an earlier private Gate
does not substitute for this new image/CI result.

## Additional FlagGems and FlagTree modifications

The ordinary stack uses explicit private dependency profiles. A public commit
pin alone does not include these modifications. Install the verified wheels
and retain their actual hashes and source/repack lineage; do not label the
modified wheels as unpatched upstream builds.

FlagGems starts from `bfc6c49d2c4d06e5deac9784f203730c8372a84a` and changes
exactly six original source files:

| Source file | Modification |
| --- | --- |
| `src/flag_gems/ops/flash_api.py` | Carry PR #6894's split-KV head stride and combine dimensions; additionally query properties using `q.device` instead of a CUDA literal. |
| `src/flag_gems/ops/flash_kernel.py` | Carry PR #6894's decoded batch/head/query output addressing with actual BSHD output strides. |
| `src/flag_gems/runtime/backend/_enflame/gcu400/ops/flash_api.py` | Carry the third PR #6894 production postimage. This preserves the reviewed patch scope; it does not certify Enflame hardware. |
| `src/flag_gems/ops/_pin_memory.py` | Validate explicit accelerators against the active runtime device; retain CPU/non-strided rejection. |
| `src/flag_gems/ops/mul.py` | Restrict MUSA's shared `mul_generic_nd_kernel` to block 512, four warps and three stages, with FlagTune expansion disabled for this entry. Preserve its kernel body and other backend configurations. |
| `src/flag_gems/utils/libentry.py` | Reraise native MUSA `illegal memory access` before the compile-failure fallback benchmarks another configuration. |

The three reviewed production changes from
[PR #6894](https://github.com/flagos-ai/FlagGems/pull/6894) are pinned at
`4e330227564ccc7f6a373be2d48a5d2cb7f2823e`; the entire PR head is not installed.
The shared `flash_api.py` also carries the separate device correction.
[FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973) tracks the
device literals and is not an extra Gate prerequisite.
[FlagGems #7005](https://github.com/flagos-ai/FlagGems/issues/7005) tracks native
error handling. The observed generic fixed-1024 NDmul failure is tracked in
[FlagTree #1346](https://github.com/flagos-ai/FlagTree/issues/1346); Torch/native
reference, public autotuned and fixed-256/fixed-512 checks passed in that
reduced matrix. The block-512 setting is a workaround, not proof of a compiler
lowering root-cause fix.

FlagTree starts from `69148faf5cc03edb0a5737abd4b75b691132cea9` and changes
only `third_party/mthreads/bin/RegisterTritonDialects.h`. Before the six
optional Proton headers, `#else` becomes `#elif defined(__PROTON__)`. Build
with `FLAGTREE_BACKEND=mthreads`, `TRITON_BUILD_FLAGPRISM=OFF` and
`TRITON_BUILD_PROTON=OFF`. The MUSA registration and enabled-profiler branches
retain their text. [FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344)
tracks this optional-profiler build guard; it does not fix compiler lowering
for #1346 or certify an unmodified/default-main wheel.

The recorded FlagGems `musafix2` wheel was an ordinary repack of the verified
`musafix1` source-built wheel: only `mul.py`, `libentry.py`, assigned version,
METADATA and RECORD changed; 5588 other normalized member bodies matched.
The FlagTree `profilefix1` wheel was built from its explicit patched source
profile. Source preparation, source builds, ordinary repacks and installing
an already verified wheel are distinct operations. New builds must record
their actual digests. The full recipe, pre/postimage pins and build flags are
in the [portable source recipe](../../tools/musa_dependency_recipe/README.md).

## CPU07 NDmul safe512 candidate

Use an isolated Linux x86_64 Python 3.10 environment with MUSA SDK 4.3.5,
driver 3.3.5-server and Torch/TorchMUSA 2.9.0. The observed TP4 communication
path uses MCCL; FlagCX was not installed. Preserve the vendor packages and
use ordinary wheels in a fresh private venv.

Prepare Gems from the pinned bfc6 archive with the explicit
`latest-bfc6-ndsafe512` source recipe. Build Tree from pinned 69148 with
`TRITON_BUILD_FLAGPRISM=OFF`, `TRITON_BUILD_PROTON=OFF` and the single-header
Proton guard. Follow the [portable recipe](../../tools/musa_dependency_recipe/README.md)
for complete commands, offline inputs and RECORD validation. CPU07 used an
ordinary `musafix2` Gems wheel repacked from its verified `musafix1` source-built
wheel; it did not execute the new source-build command. Its other six wheel
bodies and vendor/image inputs stayed unchanged. The safety profile limits
only the shared MUSA NDmul entry to block 512 and disables that entry's
FlagTune expansion; it is a workaround, not a compiler root-cause fix.

The recipe profile selects source preparation. CI instead consumes the
generic `latest` interface and an actual SHA-bound manifest containing exactly
Gems, Tree and PyYAML wheels, their actual versions/bytes/SHA256, five fixed
vendor fingerprints and honest `source_lineage` for source builds and repacks.
It does not fetch HEAD or derive lineage from a wheel version. CI defaults
to the image's manifest and literal whole-file SHA listed above. For a separately
prepared profile, set its actual manifest before workflow setup:

```bash
export MUSA_CI_DEPENDENCY_PROFILE=latest
export MUSA_CI_DEPENDENCY_MANIFEST=/data/verified/dependency-profile.json
export MUSA_CI_DEPENDENCY_MANIFEST_SHA256="$(sha256sum "$MUSA_CI_DEPENDENCY_MANIFEST" | cut -d' ' -f1)"
```

The setup step builds current FL and observer ordinary wheels, installs the
three manifest wheels into its fresh venv and verifies their installed payloads.
Keep the current FL capture-input staging and collective split, and the
observer's strict semantic joins. Optional diagnostics remain disabled and
excluded from the Gate plugin allowlist.

CI setup must use committed FL and observer changes: `MUSA_CI_EXPECTED_COMMIT`
and, when set, `GITHUB_SHA` must identify that commit. `git rev-parse HEAD`
does not give uncommitted working-tree edits committed provenance; the local
CPU07 ordinary repack is not a new Git commit.

After that setup, inside the already isolated four-device container, run:

```bash
export MUSA_CI_PYTHON=/path/to/verified-private-venv/bin/python
export MUSA_CI_EXPECTED_COMMIT="$(git rev-parse HEAD)"
export MUSA_GATE_MODEL_27B=/data/models/Qwen3.6-27B
export MUSA_GATE_MODEL_35B_A3B=/data/models/Qwen3.6-35B-A3B
bash .github/scripts/musa/run-adaptation-gate.sh
```

The runner retains TP4, bf16, maximum 8 sequences, 4096 batch tokens and the
original 26 logical cases and ten checks per route. Graph routes use GPU
memory utilization 0.92 and the following capture buckets; eager routes retain
`--enforce-eager` and utilization 0.6:

```bash
--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,256,1024,4096]}'
```

No graph mode is forced. The sparse capture buckets cover every integer
batch count through 4096, including the previously observed mixed-prefill
batches 967, 194, 722 and 527. Both graph routes completed at utilization 0.92
with the unchanged 32768-token model limit. This is an observed fit for these
exact runs; other models, inputs and shared-memory budgets need qualification.

Earlier trials remain preserved: runtime71 passed quality under maximum
capture size 16 but failed strict replay coverage; runtime76 had 25/26 quality
passes because its long-text output omitted the required `limitations`.
Runtime74/77/78 failed KV allocation before READY or a model client, and
runtime79 was unexecuted. See the [runner record](musa-ci-gate-runner.md#earlier-cpu07-trials)
for those separate scopes; none is rewritten as a successful run.

Separate runtime67r2 numerical diagnostics passed all four native/public
cases on the recorded 27B and 35B geometries, using synthetic tensor values.
The public entry used block 512 with zero benchmark calls; the child checked
the complete independent CPU oracle, output sentinels and unchanged inputs.
These synthetic checks establish that numerical scope, not compiler
root-cause acceptance or a substitute for the model Gate.

| CPU07 route / actual evidence | Original quality | JUnit / complete telemetry |
| --- | --- | --- |
| 35B-A3B eager / runtime68r2 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |
| 27B eager / runtime70 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |
| 35B-A3B graph / runtime81 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |
| 27B graph / runtime80 | 26/26; 260/260 checks true | 5 passing cases; 8 clean ends |

**The CPU07 four-route Gate passed for these exact runs.** All 104 original
logical outcomes passed all 1040 original checks, with zero false or absent
checks. The 12 JUnit XML files contain 20 passing cases, zero failures, errors
or skips. All 32 process streams have normal final ends with no errors,
drops, cap or I/O failures; all four owned jobs stopped with PID 0, no OOM
and empty cleanup errors.

The unchanged strict consumer ran once in full-matrix scope and exited 0.
It accepted all 32 byte-identical original streams (465,877,818 bytes), all
104 logical results and 20 scenarios, and certified 52 graph-request joins
and 208 request/rank replay joins. Its five full-matrix completion flags are
true and errors are empty. The consumer's separate source/GPU/ABI/model/Gate
acceptance flags remain false by design; the combined Gate conclusion uses
the independently checked original quality, JUnit and lifecycle evidence.
No cases, retries, check names or thresholds were changed.

This establishes the recorded CPU07 model/observer scope, not a compiler
root-cause fix, a general memory-fit guarantee or GitHub runner qualification.
This CPU07 record does not qualify the new image or GitHub hardware workflow.
Current ordinary-image/CI progress is recorded [above](#ordinary-stack-image-and-ci-dispatch).
The finite evidence budgets are 2 GiB per stream and 12 GiB total with
streaming aggregation hashes; they are not a process-memory bound.

## Historical vendor environment observed during this upgrade

| Component | Observed version or configuration |
|---|---|
| Accelerator | Four allocated Moore Threads S5000 devices, TP4 |
| Upstream | vLLM 0.28.0, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| Empty wheel | `vllm 0.28.0+empty`; SHA256 `2306c2ea119182a5690eb05776cdffa4ba097a92ba13114a6cb49722d0b53fec` |
| Historical candidate | `f80f78c466f4d55feed60120d1d6848f259991b9` |
| Runtime-tested candidate | `f867afb0c7f026bcabc8f80c2a68277a993ae800` |
| Vendor image ID | `sha256:863675633eb0ffd7055d60351ca9f208f271a60dc7cae811a7c68548627fdde9` |
| MUSA SDK / driver | 4.3.5 / 3.3.5 |
| PyTorch / TorchMUSA | 2.9.0 / 2.9.0 |
| FlagGems | `5.3.2.post1.dev22+gb1f939eb5.d20260804` |
| FlagTree | `0.6.1a2+mthreads3.6` |
| Communication | Actual four-rank initialization used MCCL |
| Resource limits | 24 CPUs, 256 GiB memory, 4096 PIDs, private IPC, 16 GiB shared memory |
| JIT cache | Private 8 GiB tmpfs with `rw,exec,nosuid,nodev` |

These values identify the historical environment that was inspected. The
historical vendor tag
`harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.20.2-musa-ci` remains a baseline
reference and requires validation with the selected empty wheel and candidate.
The current CI default is the published ordinary image and immutable RepoDigest
[above](#ordinary-stack-image-and-ci-dispatch).

FlagGems and FlagTree were present and imported from the vendor image. No
usable FlagCX library/wrapper pair was found in the bounded directories
inspected in that environment, and `FLAGCX_PATH` was unset. This is a statement
about that inspection, not every directory on a host. The observed MCCL route
does not require a guessed `FLAGCX_PATH`. To exercise FlagCX, install and verify
its matching library and wrapper separately and record that route's result.

## Build without changing vendor packages

Start from a MUSA image with compatible TorchMUSA, FlagTree, FlagGems, and the
build/client dependencies already installed. Network downloads are unnecessary
in the isolated installation step. Produce the empty wheel once from the exact
upstream source revision:

```bash
git -C /path/to/vllm rev-parse HEAD
# Required result: 2cf0a6915ce544dc493a0990f2ea38d81601128a
mkdir /path/to/new-empty-wheelhouse
cd /path/to/vllm
VLLM_TARGET_DEVICE=empty python -m pip wheel \
  --no-index --no-deps --no-build-isolation \
  --wheel-dir /path/to/new-empty-wheelhouse .
sha256sum /path/to/new-empty-wheelhouse/*.whl
```

Verify that the wheel METADATA names `vllm` with version `0.28.0+empty`.
Supply its **actual** SHA256 when using another independently built wheel;
do not copy the observed hash above onto different bytes.

Build the plugin and install into a fresh venv. Use `--system-site-packages` to
inherit vendor dependencies, and `--ignore-installed` to ensure vLLM and FL are
installed in the private venv despite any inherited distributions:

```bash
cd /path/to/vllm-plugin-FL
python -I -B -m venv --system-site-packages --without-pip /path/to/new-musa-venv
PY=/path/to/new-musa-venv/bin/python
mkdir /path/to/new-plugin-wheelhouse
"$PY" -I -B -m pip --isolated wheel --no-index --no-deps --no-build-isolation \
  --wheel-dir /path/to/new-plugin-wheelhouse .
"$PY" -I -B -m pip --isolated install --ignore-installed --no-index --no-deps \
  --no-cache-dir --no-compile \
  /path/to/verified-vllm-0.28.0+empty.whl \
  /path/to/verified-vllm_plugin_fl.whl
```

Replace the wheel placeholders with real complete wheel filenames. Do not set
`VLLM_VENDOR=cuda` for this MUSA build. Do not uninstall TorchMUSA, Torch,
FlagTree/Triton or FlagGems to make pip dependency resolution succeed. Import
provenance must show `vllm` and `vllm_fl` inside the new venv and vendor modules
in the image's existing site-packages. A successful wheel build alone does not
prove accelerator inference.

## Select the CI dependency profile

`MUSA_CI_DEPENDENCY_PROFILE=latest` is the CI default, with the published
image's fixed `/opt/musa/dependency-manifest.json` and literal whole-file SHA
listed above. The historical `baseline` profile remains an explicit comparison
choice. For separately prepared dependencies, supply the actual manifest path
through `MUSA_CI_DEPENDENCY_MANIFEST` and its complete JSON file SHA256 through
`MUSA_CI_DEPENDENCY_MANIFEST_SHA256`.
The manifest supplies exactly three existing ordinary wheels (FlagGems, a
**MUSA** FlagTree wheel, and PyYAML 6.0.1), with each exact version, byte size,
whole SHA256 and source lineage. It also pins the vendor Torch/TorchMUSA 2.9.0
source/metadata and SDK 4.3.5 version file. A corrected private FlagGems wheel
requires a distinct version and its actual new SHA; do not reuse a master pin.

Setup installs the three additional wheels only in its fresh private venv,
without index access or dependency resolution. Complete wheel and installed
`RECORD` files, original payload bytes, direct wheel URLs and actual module origins
are verified. Vendor Torch/TorchMUSA and SDK inputs are rechecked unchanged;
no SDK, driver, Torch or NumPy upgrade is requested. This option does not fetch
"latest" automatically or select a CUDA FlagTree wheel. The name describes an
explicit dependency profile, not a statement of compatibility.

The inspected public artifacts were FlagGems master
`7599dd130046f84df099aa771e91d35d5991e69a` with explicitly assigned archive
version `5.4.0+archive.g7599dd130046`, and FlagTree `0.7.0+mthreads3.6`.
The archive version is not a Git-derived release version. The unmodified
master profile failed model startup at MUSA pinned-memory validation; do not
claim it passes the model or Gate. Component probes alone do not replace the
unchanged four-route 20-scenario/104-request Gate and its 32 complete observer
streams. New profile results must retain that full-matrix contract.

## Historical CPU06 musafix1 dependency profile

The CPU06 private environment used FlagGems `bfc6c49d2c4d06e5deac9784f203730c8372a84a` and FlagTree `69148faf5cc03edb0a5737abd4b75b691132cea9`. The [portable recipe](../../tools/musa_dependency_recipe/README.md) supplies an explicit latest Gems profile and Tree header-patch manifest, while retaining the historical fixed profile. PR #6894 contributes the reviewed split-KV production changes; the separate device checks tracked by [FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973) are not a model graph-cause claim or a Gate prerequisite. [FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344) records the optional OFF/OFF header build guard, not general GPU compatibility.

| Scope | Actual result |
| --- | --- |
| Gems/Tree CPU source builds | Ordinary wheels and complete RECORD checks passed; explicit patched source profiles |
| CPU06 private installation | Seven ordinary wheels verified; only Gems/Tree replace CPU05; other five wheel bodies and vendor fingerprints unchanged |
| CONFIG06 | Seven metadata rows, original 19 Gate files, four routes × 26 requests and four ranks bound successfully |
| Current captured-input implementation | Strong capture-input references and checked copies before replay; 34 CPU contract cases passed |
| CPU06 original four-route GPU Gate | 87/104 quality passes; strict full-matrix consumer FAIL; 35B-A3B eager remains unresolved |
| Historical migrated baseline | 35/104 quality passes; full Gate FAIL |

The CPU06 Tree wheel used explicit TRITON_BUILD_FLAGPRISM=OFF/TRITON_BUILD_PROTON=OFF plus the header guard, not a pristine default build. CPU checks do not establish native ABI, graph numerical or model acceptance. The observed TP4 communication route uses MCCL; FlagCX was not installed. Diagnostics remain disabled by default. At that historical snapshot, automatic discovery was `musa.enabled: false` and GitHub CI was NOT_RUN. Requests, retries, check names/thresholds and strict observer joins are unchanged. Detailed run/candidate boundaries are centralized in [the runner validation history](musa-ci-gate-runner.md#validation-history-and-candidate-boundaries); successful resource release never repairs failed quality, missing end events or cleanup errors.

## Allocate devices and executable caches

`MTHREADS_VISIBLE_DEVICES` selects physical devices when the MUSA runtime
creates the container. `MUSA_VISIBLE_DEVICES` selects the device ordinals
visible to processes inside that container. Validate the actual native PCI
mapping before inference; the two masks are not interchangeable.

For the inspected allocation, physical devices `4,5,6,7` became logical devices
`0,1,2,3`. Their respective PCI bus IDs were `0000:83:00.0`, `0000:a3:00.0`,
`0000:c3:00.0`, `0000:e3:00.0`. Use the allocation assigned to your runner;
do not copy these host-specific numbers onto a different machine.

Configure all three installed cache APIs before importing the model:

```bash
export GEMS_VENDOR=mthreads
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
export VLLM_NO_USAGE_STATS=1
export TRITON_CACHE_DIR=/root/.cache/triton
export FLAGGEMS_CACHE_DIR=/root/.cache/flaggems
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
"$PY" -I -B .github/scripts/musa/cache_check.py
```

The effective cache mount must permit execution. On the inspected container,
`/tmp` had `noexec`; putting generated shared libraries there produced loader
errors. Triton, FlagGems and TorchInductor each need their supported cache
variable pointing to the private executable mount. The cache check records
actual module paths/hashes, effective directories, mount flags, and a temporary
write/read/remove check. Keep `nosuid,nodev` on the cache mount. Do not change
global loader paths or modify vendor Python files to redirect these caches.

## Run the complete Gate

The unchanged [adaptation cases](../../tools/adaptation-gate-cases/README.md)
contain five scenarios per route: one long text, eight concurrent text, one
image, eight concurrent image, and eight mixed requests. Each route has
26 logical SDK requests; all four routes have 104, comprising 52 text and
52 image requests. SDK retries may produce additional HTTP attempts.

| Model | Execution | Example distinct port |
|---|---|---:|
| Qwen3.6-27B | eager | 18062 |
| Qwen3.6-27B | graph | 18063 |
| Qwen3.6-35B-A3B | eager | 18064 |
| Qwen3.6-35B-A3B | graph | 18065 |

Use TP4 and read-only model files. The server must permit the **existing** Gate
`images` directory through `--allowed-local-media-path`; do not regenerate the
PNG fixtures during acceptance. Eager uses `--enforce-eager`. Graph acceptance
requires actual MUSA capture and replay associated with original requests,
engine IDs, batches, ranks, and graph generations. Merely removing the eager
flag or seeing a capture progress message is insufficient.

Execute all three original pytest files even when an earlier file fails. Keep
the original messages, SDK timeout/retry behavior, sampling, response reports,
and all ten quality checks. A missing scenario or graph observation fails the
Gate. Preserve complete server/client stdout, stderr, actual exits, readiness,
all 20 scenario JSON reports and provenance. Unknown exits must remain unknown.

For a remote run, use tmux with a distinct session name and an outer deadline
so SSH disconnection does not terminate the test. Start one route at a time,
then stop only its owned process/container and confirm it has exited before
the next route. Avoid clearing caches or stopping processes belonging to other
users. A ready endpoint or a single successful text request is a smoke result,
not a complete Gate result.

## Existing CI integration

The MUSA jobs use `.github/configs/musa.yml` and
`.github/scripts/musa/{check,setup}.sh`. The loader resolves explicit repository
variables into concrete Docker options; it does not leave shell `$VARIABLE`
references in GitHub's container options.

| Repository variable | Required value |
|---|---|
| `MUSA_CI_RUNTIME` | Installed Docker MUSA runtime name, for example `mthreads` |
| `MUSA_CI_PHYSICAL_DEVICES` | Exactly four unique physical device IDs assigned to the runner |
| `MUSA_CI_LOGICAL_DEVICES` | `0,1,2,3` after validated runtime remapping |
| `MUSA_CI_PCI_BUS_IDS` | Four canonical PCI IDs in that logical order |
| `MUSA_CI_EMPTY_WHEEL` | Complete absolute `/data/...whl` path, or the verified image `/opt/musa/wheelhouse/...whl` artifact |
| `MUSA_CI_EMPTY_WHEEL_SHA256` | Actual 64-character lowercase SHA256 |
| `MUSA_GATE_MODEL_27B` | Complete absolute `/data/.../Qwen3.6-27B` directory |
| `MUSA_GATE_MODEL_35B_A3B` | Complete absolute `/data/.../Qwen3.6-35B-A3B` directory |
| `MUSA_CI_IMAGE` | Optional validated image tag or digest |

The `/data` bind is read-only. Setup builds and installs ordinary wheels into
a fresh private venv under `RUNNER_TEMP`; it does not edit inherited vendor
packages. MUSA runs the existing unit, functional, e2e and benchmark commands
serially in one runner/container job, with separate stage report directories.
Other backends retain their reusable jobs. Docker's private PID namespace keeps
the existing case cleanup within this job's own container. The wrapper records
owned-child reaping and blocks further GPU stages on timeout or uncertain cleanup.
The graph functional helper now exercises TorchMUSA rather than
skipping the MUSA platform. Its small tensor test is additional component
validation; it does not replace model Gate acceptance.

The single self-hosted MUSA job has a 1200-minute outer budget for the current
suite and Gate. Functional and benchmark stages keep their 15/30-minute bounds;
each E2E group keeps the timeout from its existing matrix, including model
checking. The original Gate step keeps its 500-minute budget and all four
6900-second route limits, readiness and pytest deadlines. The workflow's
[job timeout setting](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes)
is separate from these step/group bounds. A failed earlier test remains a failed
job even when the subsequent original Gate executes to preserve its evidence.

MUSA discovery is enabled in `.github/configs/platforms.yml`. The executing
runner's image, allocation, wheels, model paths and full Gate must still pass
validation. The selected models must be available through the existing read-only
`/data` mount. Configure the variables above for the actual runner.

## Validation record

The following historical runs use candidate
`f80f78c466f4d55feed60120d1d6848f259991b9`. Its isolated wheel build, installation,
static source binding and unchanged vendor comparison passed. Its first full
27B eager attempt stopped before client tests because an owned-process FD
snapshot was incomplete. The container was stopped with PID 0 and no OOM;
that attempt is retained as a failure. The subsequent 27B eager run passed
all five scenarios, 26 logical requests and all ten checks per response, with
three pytest exits of 0. Device/cache/probe and job exits were 0; process
monitoring and cleanup had no errors, with no OOM or PID limit events. Its
container stopped with PID 0. The observed server exit of 137 followed explicit
container shutdown and is not a natural-exit result. The full collection has
357 files, 23,418,880 bytes, SHA256
`72f4e2a8e299c3e8d9299d13108f2efcba2d1e8678531d9f16cb0f1c4da0d204`.
The 35B-A3B eager route completed with **25/26 responses passing** and pytest
exits `[1, 0, 0]`. Its long-text answer omitted the required literal
`limitations`, so the original `expected_semantics` and `expected_order` checks
failed despite coherent text. The other four scenarios passed. No original
quality check or fixture was weakened. There were no OOM/PID-limit events or
monitoring errors, and the owned container stopped with PID 0. The complete
collection has 370 files, 31,795,200 bytes, SHA256
`edca6091fbe9581dac77424fc49f66f0a90ad581703314b45529c995a310cf45`.

The 27B graph route **failed during startup**, before readiness or the
original client. It executed **zero original logical requests**. The first
worker exception was `profile_cudagraph_memory` in `model_runner.py:6953`,
calling `torch.accelerator.synchronize()` and raising
`RuntimeError: Backend doesn't support synchronizing all streams on device.`
The profiling cleanup at line 6803 repeated that error. The API exited
naturally with code 1. Its cleanup report preserves that unsuccessful exit;
the later explicit stop of the owned container returned 0, with final
container exit 137, PID 0 and no OOM. Process monitoring reported no errors or
PID-limit events. Four source-verified worker observer activations do not
prove successful capture/replay associated with any original request. The
complete collection has 471 files, 8,325,120 bytes, SHA256
`705f89cbfd9c8f8e61b7a270efda680082ce1a500024b6151df48945e143b0c5`.

The prepared f80 35B-A3B graph attempt remains **NOT_RUN / ON_HOLD** after that
startup failure. The historical matrix is therefore:

| Route | Original requests executed | Result |
|---|---:|---|
| 27B eager | 26 | PASS: 26/26, all original checks |
| 27B graph | 0 | FAIL: startup synchronization error |
| 35B-A3B eager | 26 | FAIL: 25/26; long text omits `limitations` |
| 35B-A3B graph | 0 | NOT_RUN / ON_HOLD |

The MUSA backend now binds `torch.accelerator.synchronize` to the real
`torch_musa.synchronize` alongside the existing allocator API bindings. It
validates all required vendor callables before changing the bindings and
preserves default/explicit device forwarding, vendor exceptions and
idempotence. The targeted CPU Mock regression passed 48 cases with process
exit 0. No graph-disable workaround or vendor file edit is used. The
historical f80 results, including the 35B eager quality failure, remain
recorded and do not certify a later candidate.

Candidate `f867afb0c7f026bcabc8f80c2a68277a993ae800` was then built and installed
in the isolated CPU09 environment as three ordinary wheels: `vllm
0.28.0+empty`, `vllm-plugin-fl 0.0.0` and `musa-gate-observer 0.1.0`. The current
FL wheel is 542,550 bytes, SHA256
`f3a90b458382c16483da92e763cc4b68ebf1195f069ceed2670e84d7d8423b55`;
the observer wheel is 24,375 bytes, SHA256
`0d9ad617f2204ea53ba6210c12f4529b7cef4b86c76d729cbcac5a54f158ae17`.
Installed metadata, payload, RECORD, direct wheel origins and candidate source
binding were verified. Inherited vendor files and metadata were unchanged.
The successful installation and static binding do not establish GPU or model
acceptance.

The subsequent 27B graph run, runtime30, reached readiness with the actual
TorchMUSA synchronization binding and began generating tokens. It then stopped
progressing, and EngineCore terminated after `sample_tokens` RPC timed out at
23:17:11 CST on 2026-10-09. Later worker shutdown reported native MUSA
`unknown error` or `operation not permitted` at the cleanup synchronization
point. Those later errors do not establish the underlying cause of the stall.
The old generic unsupported-all-streams startup error was not the failure in
this run.

All three original pytest invocations finished with exit 1. Their real JUnit
files contain two text, two image and one mixed testcase, all five failed.
All five original JSON reports were retained: **26 logical attempts, 0 quality
successes**, comprising 13 text and 13 image inputs. Each failed request has
`APIConnectionError` and the original `checks={}`; the ten quality checks were
unavailable, and no check result was synthesized. SDK retries and original
request settings were unchanged; 26 is not a count of wire attempts.

The eight complete traces contain one API, four TP workers and three original
pytest clients. Actual native graph capture/replay events were recorded, but
the unchanged offline consumer exited 1 because of error events. It marked the
partial route and every full-matrix/acceptance flag false. Capture/replay
counts include warmup and do not prove successful joins for all 26 original
requests. The API's recorded natural exit was 0 despite the failed inference;
that exit does not make the Gate pass. The later owned-container stop returned
0; fresh inspection shows container exit 137, PID 0, no OOM and an empty
cleanup-error list. The complete collection contains 566 payload files plus
its manifest, 88,893,440 tar bytes, SHA256
`3051c1d68a4d99a7d939a5f8214d7ca2386497c5b6b9fd7cf65117fad94ed42c`.

The 35B-A3B graph run, runtime27, also reached readiness and generated tokens,
then terminated with `sample_tokens` RPC timeout at 23:47:46 CST on
2026-10-09. Its later native MUSA errors were reported during worker cleanup;
the specific cause of the stalled native work remains unestablished. All
three original pytest exits were 1, with all five real JUnit testcases failed.
Its five JSON reports retain **26 logical attempts and 0 quality successes**,
13 text and 13 image inputs, and `APIConnectionError` rows with `checks={}`.
Its eight complete process traces include actual graph events, but the
unchanged partial-route observer consumer exited 1 on error events and kept
all completion/acceptance flags false. The natural API exit was 0; the owned
container stop returned 0, with final container exit 137, PID 0, no OOM and
an empty cleanup-error list. Its complete collection has 599 payload files
plus its manifest, 83,261,440 tar bytes, SHA256
`e7c4501dcfd525017e1bbaa42a51b6f688cdaa02fdfa75fa049fa62cdb95f59a`.

The preserved first 27B eager run, runtime28, completed all five original
scenarios with **26/26 quality successes**, all ten checks per response, and
13 text plus 13 image inputs. Its three original pytest exits were 0 and its
five real JUnit testcases (2/2/1) passed. The actual server command contains
`--enforce-eager`, uses port 18062, and records compilation and graph modes
`NONE`. The job and the owned graceful API shutdown returned 0. The owned
container stop returned 0; fresh inspection shows container exit 137, PID 0,
no OOM and an empty cleanup-error list. Its complete collection has 411
payload files plus its manifest, 55,470,080 tar bytes, SHA256
`eda15a016c9c3f545dde384bb5c66a4966dfdb839c884f8cf9f7824415825ab1`.

The **complete runtime28 route was FAIL** because observer closure did not
pass. API and all three client traces have normal ending events, but the four
worker traces each contain activation and 234 scheduler begin/return pairs,
ending at `scheduler_return` without `telemetry_end`. The unchanged original
consumer exited 1 with `ContractError: activation first and telemetry_end last
required`; partial and full-matrix completion remained false. Successful
process cleanup does not replace these missing events. The original traces
were retained unchanged, and the consumer's closure requirement was preserved.

The preserved first 35B-A3B eager run, runtime29, produced **25/26 quality
successes** with original pytest exits `[1, 0, 0]` and four of five real JUnit
testcases passing. Only `text_single` failed: its response omitted the required
literal `limitations`, so `expected_semantics` and `expected_order` were false;
the other eight checks, including minimum length, passed. All 26 requests had
responses and all ten original checks; the remaining 25 quality results passed.
Four workers each ended at `scheduler_return` after 297 begin/return pairs,
without `telemetry_end`. The unchanged original consumer exited 1 on the missing
boundary. The API exited 0 after an owned SIGTERM; the container stop returned 0,
with final exit 137, PID 0, no OOM and no cleanup errors. Its complete collection
has 424 payload files plus its manifest, 58,460,160 tar bytes, SHA256
`1563385462dacbe8b9f58e3e8be161c3ae185f58e2759d817a55e5bdfa97e4df`.

The missing worker ends in runtime28/29 accompanied the upstream default
`--shutdown-timeout 0`: the API selected abort mode and immediately killed the
EngineCore tree. The later eager reruns add **only `--shutdown-timeout 60`** to
the server command, allowing drain mode and real worker cleanup. The original
requests, SDK retries, quality thresholds, model, TP4, observer and vendor code
remain unchanged. Historical traces and failed consumer results are preserved;
no ending event is synthesized or closure requirement relaxed.

The 27B eager rerun, runtime31, is a **complete single-route PASS**. All five
original scenarios pass, with 26/26 quality successes, 260 true original checks,
13 text plus 13 image inputs, three pytest exits of 0 and five passing real JUnit
testcases. Its actual command includes `--enforce-eager`, port 18062 and
`--shutdown-timeout 60`, with compilation and graph modes `NONE`. The eight
source-bound traces contain 2,023 events; each of the four workers records
activation, 234 scheduler begin/return pairs and a real final `telemetry_end`
with no errors. The unchanged original partial-route consumer exits 0 with
`route_observer_complete=true`, all 26 original response-to-engine joins, and no
errors. Its scope remains `route_only`; all full-matrix and acceptance flags
remain false. Independent reads verify the original returned rows, all 260
quality checks and joins to each of the four schedulers.

Runtime31 records drain mode with timeout 60s and `all workers exited
gracefully`. Its job and API exit 0. The owned container stop returns 0, with
final container exit 137, PID 0, no OOM and no cleanup errors. The complete
collection has 411 payload files plus its manifest, 55,316,480 tar bytes,
SHA256 `418d8d9a533700688a056052fc5f0f57adfe1b008627a853d71ab07e1c014632`.
The 35B-A3B eager rerun, runtime32, is **FAIL on original quality**, with
**25/26 quality successes**. All 26 requests have responses and ten original
checks, with 13 text and 13 image inputs. The single-text response contains
1,578 characters but omits the required literal `limitations`; only
`expected_semantics` and `expected_order` fail. Its remaining eight checks and
the other 25 requests pass. Independent execution of the original pure check
functions against the saved responses confirms 258 true and two false checks.
The three original pytest exits are `[1, 0, 0]`, and four of five real JUnit
testcases pass. No original prompt, token limit, threshold or fixture was changed.

Runtime32's real observer endings now pass the boundary requirement: its eight
complete traces contain 2,527 events, and each of the four workers has
activation, 297 scheduler begin/return pairs and a final `telemetry_end` with no
errors. The unchanged original partial-route consumer still exits 1, now at
`joins` with `ContractError: original logical quality failure`. Its route and
full-matrix completion flags remain false. The actual server uses
`--enforce-eager`, port 18064 and `--shutdown-timeout 60`, with compilation and
graph modes `NONE`. It records drain mode and all workers exiting gracefully.
The job exits 1; the API exits 0 and the owned container stop returns 0. Fresh
inspection shows container exit 137, PID 0, no OOM and no cleanup errors. This
resource success does not override the quality failure. The complete collection
has 420 payload files plus its manifest, 57,692,160 tar bytes, SHA256
`67a05fca76319677f74b325da36ae00069b82f39e075cc2a5c0f140f733c3392`.

| Route | Current original logical attempts | Current result |
|---|---:|---|
| 27B eager, runtime31 | 26 | PASS: 26/26 quality and complete single-route observer |
| 27B graph, runtime30 | 26 | FAIL: 0/26 quality successes; RPC timeout |
| 35B-A3B eager, runtime32 | 26 | FAIL: 25/26 quality; literal `limitations` missing |
| 35B-A3B graph, runtime27 | 26 | FAIL: 0/26 quality successes; RPC timeout |

All four historical runtime30/27/31/32 routes were executed, retaining **20 original JSON reports
and 104 logical requests**: 52 text and 52 image inputs, **51 quality successes
and 53 failures**. Their 12 original pytest invocations produce 20 real JUnit
testcases: nine pass, 11 fail, with no error or skip testcase. Failed graph
requests retain `APIConnectionError` and `checks={}`; unavailable quality checks
were not fabricated. These counts describe logical results, not SDK wire attempts.

The unchanged original full consumer was executed once in default full-matrix
mode over byte-verified copies of all 32 unfiltered traces from runtime30/27/31/32;
old runtime28/29 results were not substituted. It exited 1 at `events` with
`ContractError: unsupported/error/unknown event`. Its first rejected event is
the runtime30 graph API's actual `api_exception`, status `error`, at sequence 7.
The consumer stops at that error; verified copies do not mean it parsed all
32 streams to completion. All full-matrix completion and acceptance flags remain
false, and the required 208 graph request/rank joins are not verified complete.

The historical runtime30/27/31/32 full four-route Gate acceptance is **FAIL**.
At that snapshot, the CI runner's shutdown-window fix was applied locally and
its 26 CPU helper tests passed. GitHub CI had not executed, and automatic MUSA
discovery was disabled in `.github/configs/platforms.yml` (`musa.enabled: false`).

The dedicated job also builds `tools/musa-gate-observer` as a third independent,
ordinary wheel, without index access, dependencies, or build isolation. It
verifies the observer's local metadata, wheel payload and RECORD; vendor
packages remain inherited. Only the Gate server and its three separate original
pytest processes activate the observer. The [CI runner contract](musa-ci-gate-runner.md)
describes its full-matrix aggregation and cleanup gates. A partial observer
diagnostic cannot pass the complete CI matrix.

The origin check pins all 19 Gate files to the **current checked-out candidate**.
This includes the updated Gate README. The request implementation, original
three tests and eight PNG fixtures retain their own committed-byte identities;
the 19-file table is not claimed to be identical to a historical upstream kit.

The validated Debian-based vendor image has no `ensurepip`. The venv therefore
uses `--without-pip`; `python -I -B -m pip --isolated` reuses the inherited
protected pip module without replacing it or writing vendor packages.

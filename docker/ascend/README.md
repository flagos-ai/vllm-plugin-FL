# Ascend 910C: vLLM 0.32.0 empty

This recipe targets the `0.5.0-dev` plugin branch and installs upstream
`v0.32.0` as a non-editable `0.32.0+empty` wheel. Ascend selects the upstream
V2 model runner for configurations supported by upstream vLLM. The primary
0.32 checks explicitly set `VLLM_USE_V2_MODEL_RUNNER=1`. With the variable
unset, vLLM can select V1 for features such as suffix speculative decoding.
An explicit value of `0` retains the legacy FL runner. The recipe reuses the
Ascend runtime from the 910C application image listed on the
[FlagOS release page](https://flagos-ai.github.io/release-info/application/):

```text
harbor.baai.ac.cn/flagos-app/vllm0.24.0-ascend-cann9.0.0-910c:2.2.0-0.3.0
sha256:fc83d07a4fbecd090dc471e9caa5d978dc34715fcfcca537b7bbf720a298dd51
```

The manifest was checked on 2026-11-04. The application image is a base for
this build; its packaged vLLM 0.24.0 and FL plugin are replaced. The image is
for aarch64 / Ascend 910C. Its
[published prerequisites](https://flagos-ai.github.io/release-info/application/vllm0.24.0-ascend-cann9.0.0-910c/)
list host driver `26.0.rc1` and Ascend container runtime `>= 6.0.RC3` when
using that runtime. Confirm the host prerequisites before running it.

The previous vLLM 0.28 recipe and its validation notes are preserved in
[README.v0.28.0.md](README.v0.28.0.md). The
[deployment guide](../../docs/usage/ascend.md) also describes that historical
validated stack; its image and results do not validate this 0.32 recipe.

## Runtime preservation

The selected image uses `/flagos/bin` and `VIRTUAL_ENV=/flagos`. The image
provides FlagTree/Triton compiler selection. Use its FlagTree environment for
both dependency installation and inference. The derived image also records
FlagTree/CANN Python paths and driver/CANN library paths in Docker `ENV` so
`sh` job steps and direct Python entrypoints work without sourcing
`/etc/bash_env.sh`. The inherited library paths remain available. Inspection
of this application image on 2026-11-04 found the following packages:

| Component | Installed version in the selected base | Source identity |
| --- | --- | --- |
| PyTorch | `2.10.0+cpu` | `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` from `torch/version.py` |
| torch-npu | `2.10.0` | `94f8a8e6b523d7ba553e1b80d5b5248478391526` from `torch_npu/version.py` |
| FlagTree / active Triton | `0.7.0+ascend3.5` / `3.5.1` | No Git commit exposed by the installed package; retained from the pinned base digest |
| FlagGems | `5.4.0` | `_version.py` reports `g6ed2f3907`, resolving to `6ed2f39071ef85db26b7e64f0652a162f434959c` |
| NumPy | `1.26.4` | Retained from the pinned base digest |
| compressed-tensors | `0.17.0+flagos` | Retained from the pinned base digest |
| OpenCV | `5.0.0.93+flagos` | Retained from the pinned base digest |
| xgrammar | `0.2.3+flagos` | Retained from the pinned base digest |

The upstream vLLM `v0.32.0` tag resolves to
`5df63a7203cdb9328d824a27b682e77c7b9928be`; its source archive is locked
by SHA256 `bf592c132f27bbb5f65cff0bbd2ee873eae722b79c8286934fd5b5cf88b43d61`
in the Dockerfile. CI installs FlagGems-vllm `v0.2.0` commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19` (its package metadata still
reports `0.1.0`). The active FlagTree Triton entrypoint has SHA256
`4ec1df745c6bfd03257370a2aab1534e7a31eaa62110cad2ff6c3b56a67cfc50`.

These are observed base-image versions, rather than a new model-validation
matrix. The older runtime documentation's package table does not describe
this application image. Preserve the installed FlagOS builds alongside the
native NPU stack when resolving vLLM's Python dependencies.

The build records the selected interpreter's native distributions with
[`runtime_constraints.py`](runtime_constraints.py), including CPU PyTorch,
`torch-npu`, FlagTree, FlagGems, CANN Python packages and NumPy. It writes:

| File in `/opt/ascend-stack` | Purpose |
| --- | --- |
| `native-stack.json` | Exact native distribution versions from the base |
| `compiler-entrypoint.json` | Selected Python interpreter, Triton entrypoint, owning distribution and file hashes |
| `native-overrides.txt` | uv overrides that preserve those versions |
| `uv.toml` | Excludes transitive Triton distributions absent from the selected environment |

uv overrides allow vLLM's Python dependencies to be installed while retaining
the image's vendor binaries. Build isolation remains disabled. The build does
not install NVIDIA runtime requirements or replace FlagTree with a PyPI
Triton wheel. A separate compiler environment elsewhere in the image does not
need to be removed.

The verifier rejects changes to the recorded packages or compiler
entrypoint, an editable or non-empty vLLM installation, vLLM CUDA extensions,
and a residual `vllm-ascend` installation. The compiler snapshot matches the
active Triton file against distribution RECORD ownership and hashes, so
unchanged package versions cannot hide a replaced entrypoint. A successful
metadata check establishes package preservation; NPU imports and inference
need a separate hardware check.

## Build

From the repository root on a Linux/aarch64 Docker host:

```bash
docker/build.sh \
  --platform ascend \
  --target ci \
  --image-name harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend
```

The generated tag is `vllm0.32.0-a3-ci`. The build also accepts
`--index-url <URL>` and `--extra-index-url <URL>` for package mirrors. These
options are forwarded to the Ascend Docker build without changing its native
package overrides.

The `dev`, `ci` and `release` stages share the empty-vLLM runtime. The CI stage
adds the test dependencies.
The image prepares the runtime; install the selected plugin checkout before
running inference. Shared CI overlays its checkout using the Ascend setup
script. A deployable installation should use an ordinary plugin wheel:

```bash
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir /tmp/vllm-fl-wheels .
python -m pip install --no-deps --force-reinstall \
  /tmp/vllm-fl-wheels/vllm_plugin_fl-*.whl
```

Run remote image pulls, builds, installations and tests inside a named
`tmux` session so an SSH disconnect does not interrupt them.

### Connect the shared CI

Use the following naming pattern for new backend image publications:
`vllm-plugin-fl-<backend>:vllm<version>-<hardware>-<stage>-<build-id>`.
The backend has its own repository, the tag records the vLLM version,
and CI pins the full `repository:tag@sha256:...` reference. The Ascend
build helper defaults to `harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend`.

The validated dependency image has been published as:

```text
harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend:vllm0.32.0-a3-ci-20261104-v2-0b420101
```

[The Ascend CI configuration](../../.github/configs/ascend.yml) now selects
this immutable reference for `ci_image`:

```yaml
ci_image: harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend:vllm0.32.0-a3-ci-20261104-v2-0b420101@sha256:eadc6f0fa34d2b8291db76c2ff24f4dddfadfb449b9f5c282147ef7c229a5ad5
```

Registry manifest inspection confirmed that its config ID matches the
validated local image recorded below. Pulling by this digest also succeeded;
the pulled image is `linux/arm64` and its ID matches the validated build.
The backend-specific repository is a retag of that exact image; its
native stack and plugin-installation procedure are unchanged.

The [platform registry](../../.github/configs/platforms.yml) enables Ascend
for PR routing and manual, nightly and weekly discovery. The PR workflow
runs the full selected stack by default: functional/unit, eight eager/graph
E2E cases and the Ascend benchmark. Existing CI tier labels can narrow
those stages. Workflow results are separate from the recorded hardware
acceptance below.

The platform workflow reads `ci_image` and passes it directly to the
functional, unit, E2E and benchmark job containers. GitHub Actions starts
that container before checkout and platform setup. The repository's build
workflow publishes a Python wheel artifact; `docker/build.sh` only builds a
local Docker image. Neither publishes the dependency image automatically.

For future updates, build and validate a new dependency image, then publish
it under a new tag with the following procedure:

```bash
export LOCAL_IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend:vllm0.32.0-a3-ci
export PUBLISH_TAG="vllm0.32.0-a3-ci-$(date -u +%Y%m%d)-$(git rev-parse --short=12 HEAD)"
export PUBLISH_IMAGE="harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend:${PUBLISH_TAG}"
docker image inspect "$LOCAL_IMAGE" --format '{{.Id}}'
docker tag "$LOCAL_IMAGE" "$PUBLISH_IMAGE"
docker login harbor.baai.ac.cn
docker push "$PUBLISH_IMAGE"
```

The commands above describe future updates. Harbor is the repository's
current convention; the workflow accepts another
registry if every selected runner can pull it. On each `flagcicd-910c`
runner, pull the published reference using the runner account and record its
registry digest:

```bash
docker pull "$PUBLISH_IMAGE"
docker image inspect "$PUBLISH_IMAGE" --format '{{json .RepoDigests}}'
```

Set `ci_image` to that verified `repository:tag@sha256:...` reference, then run the
shared Ascend workflow with the checked-out plugin revision. Keep the driver
mounts, model mounts and allocated CI device IDs aligned with the selected
runner.

The [Ascend matrix](../../tests/platforms/ascend.yaml) explicitly selects
`VLLM_USE_V2_MODEL_RUNNER=1` and enables FlagGems with the
[Ascend dispatch policy](../../vllm_fl/dispatch/config/ascend.yaml).
The Ascend benchmark uses its own
[`smoke_ascend.yaml`](../../tests/benchmarks/configs/smoke_ascend.yaml) so the
0.6B dummy-serving smoke also passes explicit eager, disabled chunked
prefill, disabled async scheduling and disabled prefix caching options. Both
the Ascend model matrix and benchmark explicitly set `max_num_seqs: 4` to
match the configured warmup capacity. The PR and nightly model matrices
select eight cases each: inference and serving for both Qwen3.6 models in
eager and Ascend V2 decode-graph modes. Inspect the selected cases from the
prepared image before starting them:

```bash
python tests/run.py --platform ascend --device 910c --scope e2e --stage pr --dry-run
python tests/run.py --platform ascend --device 910c --scope benchmark --stage pr --dry-run
```

Platform setup also installs FlagGems-vllm from the fixed source commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19` (upstream tag `v0.2.0`). Its
installed package metadata reports `0.1.0`; setup validates the source commit
in `direct_url.json`, rather than relying on that metadata version. The
prepared dependency image supplies its build dependencies, but each fresh CI
container still needs Git access to the pinned repository unless that exact
source installation is already present.

## Check the prepared image

First run the metadata check inside the new image:

```bash
python /opt/ascend-stack/runtime_constraints.py verify /opt/ascend-stack \
  --vllm-version 0.32.0
```

On a host with a compatible driver and assigned idle NPUs, mount the host
Ascend driver and expose only the assigned devices. Follow the container
runtime or explicit-device launch method on the image's release page. When
using the Ascend runtime, pass both `ASCEND_VISIBLE_DEVICES` and
`ASCEND_RT_VISIBLE_DEVICES` for the assigned physical NPU IDs at container
launch. Then run inside the container:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export VLLM_USE_V2_MODEL_RUNNER=1
export USE_C_EXTENSION=0
python /opt/ascend-stack/runtime_constraints.py verify /opt/ascend-stack \
  --vllm-version 0.32.0 --check-imports
```

`--check-imports` initializes `torch_npu`, checks NPU availability and the
Ascend FlagGems backend, and reports the selected Triton module. It is not a
Docker build check because the build normally has no attached host driver.
Do not change the host driver as part of an application container build.

## Inference smoke tests

Ascend uses the upstream vLLM 0.32 V2 runner with either eager execution or
native `torch_npu.npu.NPUGraph` uniform decode capture. Select V2 with
`VLLM_USE_V2_MODEL_RUNNER=1`. The supported graph mode is
`FULL_DECODE_ONLY`, with compilation `backend="eager"` and `mode="NONE"`.
Here `eager` names the compilation backend: model decode still uses native
NPU graph replay. The V2 path does not invoke FX or `torch.compile`.

The graph manager captures request capacities 1, 2 and 4. It updates native
attention task parameters with each step's current sequence lengths and KV
metadata before replay; padded tail requests do not write KV cache entries.
Prefill remains eager. Graph modes `PIECEWISE`, `FULL`, and
`FULL_AND_PIECEWISE`, speculative decoding, ubatching and breakable graphs
are rejected by this graph path. The legacy FL runner remains selectable
with `VLLM_USE_V2_MODEL_RUNNER=0` and retains its separate Ascend graph path.
Dense hybrid cache transfer requires a connector that understands the
Ascend cache planes and is not enabled by this recipe.

The current GDN prefill uses reference operations. A previous Qwen3.6-27B
TP2 smoke at `max_num_seqs=4` took 321 seconds for warmup and 116 seconds
for 48 generated tokens. These are recorded functional timings, without a
performance SLO. V2 warms up for the configured request capacity, so the
usual capacity of 256 adds substantial startup work. These examples and
Ascend CI use 4; they retain warmup and leave global scheduling defaults
unchanged.

Provision model weights before testing. Inside the prepared container, use
one of the following server commands. Both keep prefill, asynchronous
scheduling and prefix caching aligned with the Ascend matrix:

```bash
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export VLLM_USE_V2_MODEL_RUNNER=1
export USE_C_EXTENSION=0
export GLOO_SOCKET_IFNAME=lo
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Eager model execution.
vllm serve /models/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --host 127.0.0.1 --port 7023 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 2048 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.3 --max-num-seqs 4 \
  --enforce-eager --no-enable-chunked-prefill \
  --no-async-scheduling --no-enable-prefix-caching
```

```bash
# Native V2 uniform decode graphs; run after stopping the eager server.
vllm serve /models/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --host 127.0.0.1 --port 7023 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 2048 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.3 --max-num-seqs 4 \
  --compilation-config '{"mode":"NONE","backend":"eager","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}' \
  --no-enable-chunked-prefill --no-async-scheduling \
  --no-enable-prefix-caching
```

Wait for `/v1/models` to return HTTP 200, then send a completion:

```bash
curl --fail http://127.0.0.1:7023/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","prompt":"The capital of France is","temperature":0,"max_tokens":16}'
```

Check the actual V2 runner type, native `NPUGraph` objects and increasing
replay counters in addition to generated text. Capture sizes alone do not
establish that replay ran. Use an interface reachable by every rank in
place of `GLOO_SOCKET_IFNAME=lo` for multi-host jobs.

V2 sampling retains the upstream request state and probability transforms.
Temperature and penalty kernels read an immutable logits source, use bounded
1024-entry tiles and write the original output view. This preserves a single
application of each transform even when a backend repeats a tile. Explicit
masks preserve Murmur3 logical right shifts on Ascend; seed/position mixing,
Gumbel noise and the full 32-bit source entropy remain the upstream algorithm.
Independent numerical oracles verify the transforms, hash values and sampled
choices; final-wheel model and API checks also verify seeded repeatability.

For reproducible offline seed comparisons, set
`VLLM_ENABLE_V1_MULTIPROCESSING=0` before creating `LLM`. Upstream 0.32
otherwise permits request arrival timing to change scheduling; a seed alone
does not guarantee identical outputs across arbitrary batches. Keep prompt
order, parameters, hardware and vLLM version fixed. This setting is an
offline test precondition, not an online-serving reproducibility guarantee.
The upstream batch-invariance support list does not include Ascend.

## Complete CI gate

Use the immutable dependency image above, a fresh checkout of the revision
under test, assigned idle NPUs and the same model/driver mounts as shared
CI. Run long remote commands in `tmux`. Record the exact checkout commit,
ordinary wheel filename/SHA256, dependency image digest, visible physical
NPU IDs, environment, command, exit status and JSON/JUnit/log paths for
every gate. Preserve test failures and skip reasons in those reports.

The lint commands follow [the shared lint workflow](../../.github/workflows/_lint.yml).
Run them from a checkout whose base revision is available; set `BASE_REF`
to the PR base used by that workflow:

```bash
export BASE_REF=7a1754e
mapfile -t PY_FILES < <(git diff --name-only --diff-filter=ACMRT "$BASE_REF" -- \
  '*.py' '*.pyi' ':(exclude)vllm_fl/**')
if ((${#PY_FILES[@]})); then
  ruff check --output-format=github "${PY_FILES[@]}"
  ruff format --check "${PY_FILES[@]}"
fi
mapfile -t SH_FILES < <(git diff --name-only --diff-filter=ACMRT "$BASE_REF" -- '*.sh')
if ((${#SH_FILES[@]})); then shellcheck "${SH_FILES[@]}"; fi
typos .
```

The generic [wheel build job](../../.github/workflows/_build_wheel.yml)
uses a clean Python 3.12 checkout and verifies package structure:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/*.whl
python - <<'PY'
from importlib.metadata import files, metadata
meta = metadata('vllm-plugin-fl')
assert any(str(f).startswith('vllm_fl/') for f in files('vllm-plugin-fl'))
print(meta['Name'], meta['Version'])
PY
```

On the prepared Ascend image, build the hardware-validation wheel without
build isolation or dependency replacement, from a fresh checkout without
stale `build/lib` contents:

```bash
python -m pip wheel --no-build-isolation --no-deps --wheel-dir /tmp/ascend-gate-wheels .
sha256sum /tmp/ascend-gate-wheels/vllm_plugin_fl-*.whl
python -m pip install --no-deps --force-reinstall /tmp/ascend-gate-wheels/vllm_plugin_fl-*.whl
python .github/scripts/install_flaggems_vllm.py
python docker/ascend/runtime_constraints.py verify /opt/ascend-stack \
  --vllm-version 0.32.0 --check-imports
```

Shared CI's Ascend setup installs its checkout in editable mode. The final
ordinary-wheel acceptance must separately verify the installed package
outside the checkout, confirm that it is non-editable, and compare all
packaged source/configuration file hashes with the selected checkout.
Use one immutable source snapshot for tests, with no production-source
overlays after wheel verification. Record installed module paths so a
checkout import cannot conceal a stale installed wheel.

Run the complete selected scopes, without `-k`, `--maxfail`, or ad hoc
exclusions:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
python tests/run.py --platform ascend --device 910c --scope unit \
  --output-dir /tmp/ascend-full-gate/unit
python tests/run.py --platform ascend --device 910c --scope functional \
  --output-dir /tmp/ascend-full-gate/functional
python .github/scripts/generate_matrix.py --platform ascend --ci-stage pr
python tests/run.py --platform ascend --device 910c --scope e2e --stage pr \
  --output-dir /tmp/ascend-full-gate/e2e
python tests/run.py --platform ascend --device 910c --scope benchmark --stage pr \
  --output-dir /tmp/ascend-full-gate/benchmark
```

`--scope unit` runs both normal tests (`not multi_gpu and not serial`,
up to 16 xdist workers) and a second serial pass (`multi_gpu or serial`).
Keep both JSON reports, `report-ascend.json` and
`report-ascend-serial.json`, plus the combined coverage and launcher
reports. A targeted subset does not replace either phase. Functional tests
use the platform's committed selection, which currently excludes
`ops/test_ops_correctness.py`; record this selection with the results.

For both PR and nightly, matrix discovery must select eight nonempty cases:
Qwen3.6-27B and Qwen3.6-35B-A3B, each eager and native decode graph, each
inference and serving. Effective settings are TP2,
`max_model_len=4096`, `max_num_batched_tokens=4096`, and `max_num_seqs=4`.
Graph options are scoped to the Ascend graph case files, leaving eager
cases eager. Both serving models retain the original text assertion
`Paris` and image assertions `Hello VLM` and `blue rectangle`. The benchmark
uses the dedicated eager `smoke_ascend.yaml` with five requests.

## Validation record

Hardware checks used Ascend 910C 64 GB with host driver `25.5.0`. The base
image's release page lists `26.0.rc1`; retain that distinction when applying
these results to another host. The host driver was retained.

| Artifact | Identity / status |
| --- | --- |
| Published dependency image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend:vllm0.32.0-a3-ci-20261104-v2-0b420101` |
| Immutable CI image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl-ascend:vllm0.32.0-a3-ci-20261104-v2-0b420101@sha256:eadc6f0fa34d2b8291db76c2ff24f4dddfadfb449b9f5c282147ef7c229a5ad5` |
| Docker config/image ID | `sha256:0b420101520066407a2e208d3fc3fb7f406fa4afe84c38eb2cf8e86bef97c3d2` |
| Published-image verification | Manifest/config match; digest pull succeeded and returned the same `linux/arm64` build |
| vLLM | Ordinary `0.32.0+empty` wheel |
| Validated implementation commit | `3111d142c36e5f620740cf36b67a05bcd11876ff` |
| Final ordinary plugin wheel | `vllm_plugin_fl-0.5.0.dev0+ascend032v2graphg4-py3-none-any.whl` |
| Plugin wheel SHA256 | `e403e1d6df5b187ec2a6d11cea1bc50887eef4a07ca9551ae16856c2f0f1a1d0` |
| Installed contents | All 288 source/configuration files matched the frozen g4 manifest and wheel ZIP; ordinary package imported from `/flagos/lib/python3.11/site-packages/vllm_fl`. Final-checkout formatting equivalence is recorded below. |
| Generated `_version.py` SHA256 | `b62784e51b25645d5bf50c7c6c16926208aea8f82143087280387adcf356e3a2`, matched between wheel ZIP and installed package |

The plugin wheel is installed separately from the dependency image. Its
version, non-editable installation, complete packaged source set and hashes
were checked before the final-wheel API validation.

| Complete final-wheel gate | Result and report identity |
| --- | --- |
| Ruff check / format | Passed: Ruff `0.15.8`, all 31 changed Python files passed check and format |
| ShellCheck / Typos | Passed with ShellCheck `0.11.0` and Typos `1.50.2` |
| Clean generic wheel build | Passed: CPython `3.12.13` on Linux/aarch64, default build isolation, declared dependencies installed, package structure verified and `pip check` exit 0 |
| Ascend ordinary wheel / native integrity | Passed: wheel/source/installed hashes, direct-Python native/compiler checks, NPU availability and NPU-to-CPU copy |
| Unit normal phase | Passed: 2286 passed, 193 skipped; no failures |
| Unit serial/multi-GPU phase | Passed: 7 passed; no failures |
| Functional selection | Passed: 26 passed, 58 GCU-specific skips; successful launcher exit |
| V2 eager/graph 0.6B probe | Passed: all five groups matched eager token IDs and text with deterministic offline scheduling; both modes' seeded repeats matched |
| HTTP API and sampling | Passed all 5 groups: models, completion, non-default sampling, chat and SSE; original two-prompt batch repeats and added single-prompt seed repeats matched; exit 0 in 269.35 seconds |
| Original CI benchmark smoke | Passed: 1 case, 5 requests, 0 failed; 105.42 seconds, exit 0 |
| Ascend CI setup replay | Passed: original device check and setup both exit 0; genuine pinned VCS metadata, native/compiler integrity and source preservation verified with the offline Git transport described below |

The generic build ran on CPU Linux/aarch64 with CPython 3.12.13; this
architecture differs from the shared workflow's x86_64 `ubuntu-latest`
host. It produced a pure `py3-none-any` wheel with SHA256
`3a8d46af7c7a3ffa92e48005fbd214b892001a9f638fd658b168ed4209905117`.
Default build isolation and declared build/runtime dependencies were
retained. Dependency transport used a cache of real published-hash-verified
wheels through `PIP_NO_INDEX` and `PIP_FIND_LINKS`, including the official
CPU PyTorch build dependency. The clean 681-file source snapshot was
unchanged, installation included declared dependencies, and `pip check`
passed. Reproducing this build requires access to those dependencies through
an index or the verified cache.

The full normal and serial unit phases include the new Ascend NPU
regressions, which actually ran and passed. Targeted source-overlay results
are separate evidence and are not added to these final-wheel totals. The
functional selection retains its committed exclusion of
`ops/test_ops_correctness.py`; its 58 skips require the GCU backend.

The offline probe used `VLLM_ENABLE_V1_MULTIPROCESSING=0`, native
`NPUGraph` capture sizes 1/2/4 and batches 1/2/4. Graph manager counters
reported 98 total replays: 58 at size 4, 20 at size 2 and 20 at size 1.
The HTTP API run separately proved request-period replay: after health and
`/v1/models` returned 200, it recorded the server-log byte offset and
observed 133 subsequent replays. Its three startup warmup replays were
excluded from that count. API cleanup targeted only its own recorded
process identities and confirmed physical NPU 4 was idle with no process
remaining.

| Final TP2 matrix case | Result |
| --- | --- |
| Qwen3.6-27B eager inference | Passed |
| Qwen3.6-35B-A3B eager inference | Passed |
| Qwen3.6-27B graph inference | Passed: each of 2 ranks captured sizes 1/2/4 and performed 79 request replays |
| Qwen3.6-35B-A3B graph inference | Passed: each of 2 ranks captured sizes 1/2/4 and performed 99 request replays |
| Qwen3.6-27B eager serving, text and image | Passed |
| Qwen3.6-35B-A3B eager serving, text and image | Passed |
| Qwen3.6-27B graph serving, text and image | Passed: both ranks captured sizes 1/2/4 and each performed 821 post-ready request replays |
| Qwen3.6-35B-A3B graph serving, text and image | Passed: both ranks captured sizes 1/2/4 and each performed 607 post-ready request replays |

All 8 selected E2E cases passed, comprising 12 pytest items. The serving
cases retained the original `Paris` text assertion and `Hello VLM` plus
`blue rectangle` image assertions. The corrected discovery command above
was also run on CPU and selected two nonempty jobs with four cases each:
four eager and four graph cases across inference and serving.

The frozen g4 build snapshot matched all 288 production source/configuration
files in the runtime wheel and installed package byte for byte. Compared
with that snapshot, implementation commit
`3111d142c36e5f620740cf36b67a05bcd11876ff` has 286 of 288 files byte-identical.
The other two files, `ascend_v2_gumbel.py` and
`ascend_v2_temperature_kernel.py`, only remove a blank line at EOF; their
ASTs and every function's source bytes remain identical. Three test files
also received formatting-only changes with identical full ASTs. An
independent real-NPU recheck passed 15 cases and 9 subtests.

A separate real-weight Qwen3-0.6B throughput measurement used 16 requests,
1024 prompt tokens and 128 output tokens per request, `max_num_seqs=4`, and
three timed runs per mode:

| Mode | Median output tokens / second |
| --- | ---: |
| V2 eager | 133.365 |
| V2 native decode graph | 452.248 |

The graph/eager ratio was 3.391x for this workload. These rates apply to that
model, hardware and request shape. The CI benchmark smoke above exercises a
separate five-request scenario.

The setup gate used an offline mirror of genuine Git objects fetched from
the official FlagGems-vllm repository, retaining its complete Git history.
The pinned commit was `b5c2c2249319277ccda4d41aa0f65164bd9b2a19`, with tree
`d8afa0e82429a0afd620e8c3005e501bfc52b802`; transport used a verified Git
bundle with SHA256
`f9d731aa79cefdd972b0e821f6cae08938307fe1dfa75821159171baacd295ff`.
A process-scoped Git URL rewrite selected that mirror while the original
check/setup scripts and pip Git URL remained unchanged. Installed VCS
metadata retained the official origin and exact commit. Native distributions,
compiler ownership/hashes, all 288 plugin files and all 681 checkout snapshot
files were unchanged across setup. This verifies setup with that offline Git
transport; other hosts need access to the pinned source through their
configured Git transport.

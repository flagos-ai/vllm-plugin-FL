# Ascend 910C: vLLM 0.32.0 empty

This recipe targets the `0.5.0-dev` plugin branch and installs upstream
`v0.32.0` as a non-editable `0.32.0+empty` wheel. Ascend selects the upstream
V2 model runner by default; the primary 0.32 checks use
`VLLM_USE_V2_MODEL_RUNNER=1`. An explicit value of `0` retains the legacy FL
runner for compatibility. The recipe reuses the Ascend runtime
from the latest 910C application image listed on the
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

| Component | Installed version in the selected base |
| --- | --- |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0` |
| FlagTree | `0.7.0+ascend3.5` |
| FlagGems | `5.4.0` |
| NumPy | `1.26.4` |
| compressed-tensors | `0.17.0+flagos` |
| OpenCV | `5.0.0.93+flagos` |
| xgrammar | `0.2.3+flagos` |

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
  --image-name harbor.baai.ac.cn/plugin/vllm-plugin-fl
```

The generated tag is `ascend-vllm0.32.0-a3-ci`. The build also accepts
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

The validated dependency image has been published as:

```text
harbor.baai.ac.cn/plugin/vllm-plugin-fl:ascend-vllm0.32.0-a3-ci-20261104-v2-0b420101
```

[The Ascend CI configuration](../../.github/configs/ascend.yml) now selects
this immutable reference for `ci_image`:

```yaml
ci_image: harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:eadc6f0fa34d2b8291db76c2ff24f4dddfadfb449b9f5c282147ef7c229a5ad5
```

Registry manifest inspection confirmed that its config ID matches the
validated local image recorded below. Pulling by this digest also succeeded;
the pulled image is `linux/arm64` and its ID matches the validated build.

The platform workflow reads `ci_image` and passes it directly to the
functional, unit, E2E and benchmark job containers. GitHub Actions starts
that container before checkout and platform setup. The repository's build
workflow publishes a Python wheel artifact; `docker/build.sh` only builds a
local Docker image. Neither publishes the dependency image automatically.

For future updates, build and validate a new dependency image, then publish
it under a new tag with the following procedure:

```bash
export LOCAL_IMAGE=codex/vllm-plugin-fl:ascend-vllm0.32.0-a3-ci
export PUBLISH_TAG="ascend-vllm0.32.0-a3-ci-$(date -u +%Y%m%d)-$(git rev-parse --short=12 HEAD)"
export PUBLISH_IMAGE="harbor.baai.ac.cn/plugin/vllm-plugin-fl:${PUBLISH_TAG}"
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

Set `ci_image` to that verified `repository@sha256:...` reference, then run the
shared Ascend workflow with the checked-out plugin revision. Keep the driver
mounts, model mounts and allocated CI device IDs aligned with the selected
runner.

The [Ascend matrix](../../tests/platforms/ascend.yaml) explicitly selects
`VLLM_USE_V2_MODEL_RUNNER=1`. Its existing `USE_FLAGGEMS=1` and
`VLLM_FL_FLAGOS_WHITELIST=zeros` settings are retained. Broadening that CI
operator coverage requires its own validation; the separate full-policy
operator checks below exercise a wider set of registrations. The Ascend
benchmark uses its own
[`smoke_ascend.yaml`](../../tests/benchmarks/configs/smoke_ascend.yaml) so the
0.6B dummy-serving smoke also passes explicit eager, disabled chunked
prefill, disabled async scheduling and disabled prefix caching options. Both
the Ascend model matrix and benchmark explicitly set `max_num_seqs: 4` to
match the validated warmup capacity. The r4 benchmark passed on physical
NPU 4 with Qwen3-0.6B dummy weights, validating the benchmark entrypoint and
basic plugin integration. Inspect the selected cases from the prepared image
before starting them:

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

## Inference smoke test

The Ascend adapter is a subclass of the upstream vLLM 0.32 V2 GPU model
runner. V2 currently requires `--enforce-eager` (`cudagraph_mode=NONE`);
requesting graph capture is rejected. The legacy FL runner remains available
with `VLLM_USE_V2_MODEL_RUNNER=0` for its separate Ascend ACL graph path.
Dense hybrid cache transfer also requires a connector that understands the
Ascend cache planes and is not enabled by this V2 recipe.

The current GDN prefill path uses reference operations. A recorded
Qwen3.6-27B TP2 smoke with `max_num_seqs=4` took 321 seconds for warmup and
116 seconds to generate 48 tokens. These are observed functional-test timings;
no performance SLO is established. V2 warms up for the configured request
capacity, so the default capacity of 256 would substantially increase startup
work. The Ascend examples and CI configurations use 4 to match the validated
capacity. This limit is scoped to those configurations and does not bypass
warmup or change user scheduling defaults globally.

Offline Qwen3-0.6B TP1 eager generation has passed on the V2 runner. The clean
r4 plugin wheel passed models, completion, chat, streaming HTTP and non-default
sampling checks, including repeatable seeded output and a confirmed V2 runner
banner. The commands below describe that serving configuration. Provision
model weights before testing:

```bash
export GLOO_SOCKET_IFNAME=lo
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
vllm serve /models/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 127.0.0.1 --port 7023 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 2048 --gpu-memory-utilization 0.3 \
  --max-num-seqs 4 \
  --enforce-eager --no-enable-chunked-prefill \
  --no-async-scheduling --no-enable-prefix-caching
```

Wait for `/v1/models` to return HTTP 200, then send a completion request:

```bash
curl --fail http://127.0.0.1:7023/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","prompt":"The capital of France is","temperature":0,"max_tokens":16}'
```

Check for non-empty generated text and server-side errors. Use an interface
reachable by all ranks instead of `GLOO_SOCKET_IFNAME=lo` for multi-host jobs.
The [Ascend test matrix](../../tests/platforms/ascend.yaml) covers
Qwen3.6-27B and Qwen3.6-35B-A3B TP2 eager inference and serving. All four
configured cases passed, including text and generated-image assertions for
both serving models. The validation record below identifies the tested
settings and artifacts.

## Validation record

Validation used Ascend 910C 64 GB hardware with host driver `25.5.0`.
The selected image's release page lists driver `26.0.rc1`; the results below
record the observed `25.5.0` environment. The host driver was retained.

| Artifact | Verified identity |
| --- | --- |
| Published dependency image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:ascend-vllm0.32.0-a3-ci-20261104-v2-0b420101` |
| Immutable CI reference | `harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:eadc6f0fa34d2b8291db76c2ff24f4dddfadfb449b9f5c282147ef7c229a5ad5` |
| Local dependency image | `codex/vllm-plugin-fl:ascend-vllm0.32.0-a3-ci` |
| Docker image ID | `sha256:0b420101520066407a2e208d3fc3fb7f406fa4afe84c38eb2cf8e86bef97c3d2` |
| vLLM | Ordinary `0.32.0+empty` wheel |
| Plugin wheel | `vllm_plugin_fl-0.5.0.dev0+ascend032v2r4-py3-none-any.whl` |
| Plugin wheel SHA256 | `357177f02ee575b881d66fdc43fe9ebc578c544cc7300599da8279abbed2511c` |
| Installed plugin contents | All 285 source/configuration files matched the checkout's SHA256 hashes |

The image build and registry publication completed successfully. Registry
manifest inspection confirmed the Docker config ID shown above. The plugin
wheel is installed separately from the dependency image.

| Runtime and functional validation | Result |
| --- | --- |
| Direct Python native stack checks | Passed: package/compiler integrity, imports, NPU availability and four-element NPU-to-CPU copy; no Bash initialization step |
| V2 runner/device integration | Passed: real V2 runner selection, NPU streams/events and NonUVA buffers |
| Qwen3-0.6B TP1 API and sampling | Passed all 5 groups: models, completion, chat, streaming, non-default temperature/top-k/top-p/three penalties; repeated seeded requests matched and the real V2 banner was confirmed |
| Core regressions | 114 passed, 0 failed, 0 skipped in 37.18 seconds; direct Python, exit 0 |
| CI benchmark smoke | Passed 1 case with Qwen3-0.6B TP1 dummy weights: 5 requests succeeded, 0 failed; 257.1 seconds including startup, exit 0 |

The core suite covers attention, Mamba copies, cache block planes, V2 sampling,
native padding, empty buffers and padded MoE routes. Model execution uses V2
eager mode with `max_num_seqs=4`; the reference GDN warmup and generation
timings described above remain the current performance boundary. V2 graph
execution is not supported by this adapter.

The formal model matrix uses TP2, `max_model_len=4096`, eager mode and
`max_num_seqs=4`. All four cases passed in 966.51 seconds with exit 0:

| Matrix case | Result |
| --- | --- |
| Qwen3.6-27B inference | Passed |
| Qwen3.6-35B-A3B inference | Passed |
| Qwen3.6-27B serving (text and image) | Passed |
| Qwen3.6-35B-A3B serving (text and image) | Passed |

Both serving cases required `Paris` for the text response and `Hello VLM`
plus `blue rectangle` for the generated-image response.

A supplementary full-policy Qwen3.6-27B smoke returned `Paris` with the real
V2 runner and exit 0. Its native guards for `index_fill_`, `pad` and
`constant_pad_nd` are now part of the final default Ascend policy. This is
supporting evidence for the equivalent operator policy from a separate
installation; the validation results for the final wheel are listed above.

Use the immutable dependency image reference together with the validated
plugin revision when running shared CI on this runtime.

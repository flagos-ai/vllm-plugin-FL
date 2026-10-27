# Enflame S60 recipe for vLLM 0.28

The recipe keeps the vendor torch 2.11.0+cpu / torch_gcu
2.11.0+3.8.20260713 stack and installs the official vLLM 0.28.0 empty-device
build. The FL plugin registers the GCU platform. The vendor base is:

```text
harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-enflame-gems_5.4.0.dev0-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.8-torch_gcu_2.11.0_3.8.20260713-pcp_tops3.8.20260714-gpu_s60-arc_amd64-driver_1.9.10:202608141853
```

The installed packages are pinned to FlagGems 5.4.0, FlagTree
0.7.0+enflame3.6 and triton-gcu 3.6.0+1.0.20260722. FlagTree's Enflame wheel
provides the `triton` Python namespace; the generic `+triton3.6` release is a
different build. Wheel hashes are checked against the official
[Enflame package index](https://resource.flagos.net/repository/flagos-pypi-enflame/simple).
The current official [S60/tops1.10.6 configuration](https://github.com/flagos-ai/build-infra/blob/bbc9ae28e946a59c94e2af63d2d8aae04ff846ec/configs.yaml)
and [Enflame installation guide](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-enflame)
select this FlagTree version. FlagGems v5.4.0's own older backend matrix
selects 0.6.1+enflame3.6; the newer combination is validated below.

Enflame uses `ModelRunnerFL` by default for eager and graph execution.
vLLM 0.28 can otherwise select its upstream V2 runner automatically for
ordinary dense models, reaching sampler kernels that the current GCU stack
does not support. Explicit `VLLM_USE_V2_MODEL_RUNNER=1` requests fail during
configuration; V2-only features are unavailable on this backend.

Hybrid models with speculative decoding and `mamba_cache_mode="align"`
are rejected during configuration. vLLM 0.28 uses fused speculative Mamba
pre/post-copy kernels for that combination; their indirect state and block-table
pointers are not yet translated to GCU device addresses. Disable speculative
decoding for hybrid align-mode serving. Other speculative/cache combinations
still require their own validation; accepting a configuration is not evidence
of device support.

The 2026-10-13 dependency image was validated on S60 with host driver 1.10.6.
The compiler-file guard and 12 cold-cache FlagGems normal-operator cases passed.
The original Qwen3.6-27B inference and Qwen3.6-35B-A3B serving cases passed in
both eager and graph modes at plugin commit
`2cd00248d3992088b2432201629ab295d4d94034`. These image checks are separate
from each candidate commit's original adaptation Gate. The old vLLM 0.20.2
vendor wrapper remains available as `Dockerfile.v0.20.2`.

The local image ID is
`sha256:4feef9aed222a8cffd9fab610288053c3f2ee5acdfc3540786e47d7daea48f0e`.
The canonical dependency image name follows the other vLLM 0.28 backends:
`harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-enflame-ci`.
This tag was published on 2026-10-20 and aliases the unchanged image originally
published on 2026-10-13 as
`harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-enflame-ci-2cd00248d399-compiler`.
An anonymous manifest lookup confirmed that both names identify the same content.
Its registry manifest digest is
`sha256:f21e53101ffcf00fa5b70b39913bd401be93c0d1834c0ae8d07684bf0690bc88`.
Authenticated and anonymous pulls by this digest both resolved to the local
image ID above. CI uses the canonical tag with this digest and installs the tested plugin checkout
with `--no-deps`; the dependency image does not bake in that checkout.

See the [Enflame usage guide](../../docs/usage/enflame.md) for container setup,
model serving, and validation commands.

## Build a local candidate

Run long builds in a uniquely named tmux session and retain the build log.
From a private checkout, the underlying build command is:

```bash
docker/build.sh --platform enflame --target ci \
  --image-name local/vllm-plugin-fl --image-tag v0.28.0-enflame-candidate
```

The example tag is a local build destination, not a published image.
The default generated tag includes the selected stage, such as
`v0.28.0-enflame-ci` or `v0.28.0-enflame-dev`. This command does not push.
`.github/configs/enflame.yml` references the published vLLM 0.28 dependency
image by digest. Enflame is enabled in `platforms.yml`, following main's
existing `enflame-s60` runner label, devices `2,3` and `/data:/data` mount.
The reference is main commit `6182281a42ce857fa52ef73ec1abafca941cdf3a`.
Publishing the image and selecting this configuration do not establish
that hardware CI has passed; record the actual workflow results separately.
The S60 e2e jobs allow 120 minutes for cold-cache compilation, graph capture
and both original model cases; the measured 35B serving run took 3607 seconds.

`VLLM_VERSION` and `VLLM_SOURCE_COMMIT` select the source version and exact
commit; the default official v0.28.0 commit is
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. Pass matching values with
`--build-arg` when changing the source version, and use `VLLM_SOURCE_URL`
for a verified mirror when necessary. Wheel versions, hashes and the
official backend index are also Docker build arguments. Change a version
together with its SHA256 pin.

The build installs the selected source's `requirements/common.txt` with
vendor torch/compiler constraints before installing vLLM with
`VLLM_TARGET_DEVICE=empty --no-build-isolation --no-deps`. Installing CPU
requirements or `.[test]` can replace vendor torch. The metadata guard
checks the exact requested vLLM base version, empty build and vendor stack;
it does not initialize hardware during Docker build.

`xgrammar` can install upstream Triton while resolving the common
requirements, overwriting FlagTree's `triton` namespace while leaving its
package metadata intact. The recipe constrains that dependency to 3.6.0,
then removes upstream Triton and force-reinstalls the hash-pinned Enflame
FlagTree wheel. `verify_compiler.py` checks the active module origin and
FlagTree RECORD hashes for its shipped `triton` files during the build and
CI setup, without initializing a GPU. Runtime and original Gate validation
are still required after this source-file check.

The build explicitly upgrades the OpenAI Python SDK to `OPENAI_VERSION=3.28.0`
and pandas to `PANDAS_VERSION=3.0.6` after common requirements, using the same
vendor constraints. The vendor bundle's older SDK satisfies vLLM's
`openai>=2.0` requirement but lacks `NamespaceTool`, which caused the first
task service attempt to fail while importing `vllm.tool_parsers.utils`.
The next CLI check exposed the bundle's older pandas binary failing against
NumPy 2 with `numpy.dtype size changed`. After these private environment
upgrades, `vllm serve --help` exits successfully and vendor torch remains
unchanged. Model serving and Gate acceptance remain to be verified.

FlagGems 5.4's Python wheel has no native C++ extension bundled. The optional
`flag-gems-cpp-gcu` package must match the Python package's version and
source commit; source builds require the TOPS SDK and a C++20 toolchain.
Install it only when a required operator path needs it, then test that path
separately. Python operator/device smoke is independent of C++ coverage.

The runtime `index_fill` adapter validates the supported FlagGems version and
private launcher/kernel API. Exact release-byte verification runs separately in
`docker/enflame/verify_index_fill.py` during CI setup. Setup executes that script
from the tested checkout, so this separation does not require a new dependency
image.

## Temporary vendor compatibility adapters

The adapters in
[`inductor.py`](../../vllm_fl/dispatch/backends/vendor/gcu/impl/inductor.py)
apply to the fixed vendor stack above. Matching a private API signature alone
does not establish support for another TorchGCU or FlagTree release. Vendor
integration follow-ups are tracked in the
[PR #582 review](https://github.com/flagos-ai/vllm-plugin-FL/pull/582#pullrequestreview-5544802336);
no separate upstream issue or PR has been filed by this adaptation.

| Adapter | Upstream removal condition |
|---|---|
| Persistent-reduction signature bridge | TorchGCU natively accepts `size_hints`, `reduction_hint`, `inductor_meta` and `triton_meta`, preserves that interface through initial/repeated lazy registration in both GCU module namespaces, and defines `native_matmul` behavior. |
| Integer compile-option forwarding | TorchGCU/Inductor forwards `enable_i64`/`ENABLE_I64` at `_create_compile_options`, preserving explicit choices, and FlagTree correctly lowers integer and pointer `i64`/`u64` signatures. |
| Register-resource exception normalization | The vendor compiler raises compatible `OutOfResources` for register exhaustion, so the stock autotuner skips the exhausted tile and completes a valid tile while propagating unrelated failures. |

Before removing an adapter, convert the corresponding
[`test_gcu_inductor.py`](../../tests/unit_tests/dispatch/test_gcu_inductor.py)
cases into native-provider contract tests and run the affected cold-cache
compilation paths on S60 with the adapter disabled. Include initial and repeated
registration, dynamic integer/pointer signatures, and exhausted plus valid
autotune configurations as applicable; metadata or mock-only checks are
insufficient evidence for removal.

The cache writer keeps per-call layout, stride and address-range checks.
Attention creates new cache views, and hybrid allocation can change their
layout; sharing and reinitialization also change the binding lifecycle.
Moving these checks to initialization requires a binding contract that covers
those paths and direct wrapper calls. An object-identity cache or a check before
the final layout is established would weaken the safety boundary.

## Validate on an S60 host

Select device IDs after checking memory and process ownership. Model YAMLs
use `/data/models/Qwen`; mount the actual host model directory there
read-only. Keep the container's default private PID namespace because the
test runner performs process cleanup. Use a unique task container and
tmux sessions for setup, service and tests.

On S60, Qwen3.6-35B-A3B was validated in TP2 graph mode with
`--gpu-memory-utilization 0.94`, `--max-model-len 32768`,
`--max-num-seqs 256` and FP32 Mamba state. All five original Gate scenarios
passed (26 requests and 260 checks), with their inputs and checks unchanged.
Enflame CI shares one device-level memory budget across its four configured
TP2 eager/graph cases, so the `0.94` override applies to all four while
retaining their existing model settings and quality thresholds.

Inside that task container, from the private plugin checkout:

```bash
bash .github/scripts/enflame/check.sh
bash .github/scripts/enflame/setup.sh
python tests/run.py --platform enflame --scope unit
python tests/run.py --platform enflame --scope functional
python tests/run.py --platform enflame --scope e2e --task inference \
  --model qwen3_6 --case 27b_tp2_eager
```

The container must receive `VLLM_PLUGINS=fl`, the verified
`TOPS_VISIBLE_DEVICES` pair and the runtime environment variables set by
the Dockerfile. These suites supplement the original cases in
[`tools/adaptation-gate-cases`](../../tools/adaptation-gate-cases/README.md).
Run the configured models in eager and graph modes and retain each raw
result JSON; evaluate `summary.passed` and every request's checks before
reporting Gate acceptance.

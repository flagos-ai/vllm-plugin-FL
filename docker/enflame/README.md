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
selects 0.6.1+enflame3.6; the newer combination requires runtime validation.

The task-private equivalent installation has passed torch/FlagGems device
addition and FL platform smoke checks on S60 with host driver 1.10.6.
A task-private build of this recipe completed on 2026-10-12. Its metadata
smoke container exited successfully with the pinned vLLM, torch, FlagGems,
FlagTree and Triton-GCU versions, OpenAI 3.28.0 and pandas 3.0.6. This checks
the dependency image; it does not establish GPU serving, the final plugin
commit's CI results or Gate acceptance. The old vLLM 0.20.2 vendor wrapper
remains available as `Dockerfile.v0.20.2`.

The local image ID is
`sha256:dfabdef337c4115d019534604c0df01cc2ef7b68c7062bdb5436bd1c5f9d79ab`.
The attempted Harbor push was rejected as unauthorized. No registry
manifest digest is available, and this local ID is not a published digest.

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
`.github/configs/enflame.yml` still references the older published
`v0.24.0-enflame-ci` image, which does not match this branch's vLLM 0.28
target. Enflame remains disabled in `platforms.yml`; update its image only
after a new image is built, validated and published.

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

## Validate on an S60 host

Select device IDs after checking memory and process ownership. Model YAMLs
use `/data/models/Qwen`; mount the actual host model directory there
read-only. Keep the container's default private PID namespace because the
test runner performs process cleanup. Use a unique task container and
tmux sessions for setup, service and tests.

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

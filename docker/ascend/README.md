# Ascend 910C runtime image

The [Dockerfile](Dockerfile) builds vLLM 0.32.0 with `VLLM_TARGET_DEVICE=empty`
on the public `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` base. It retains the
base's CANN 9.0.0 and Torch 2.10.0+cpu / torch-npu 2.10.0 pair and replaces
its inherited vLLM installation. No NVIDIA vLLM extension is required.

FlagTree 0.6.2a1+ascend3.5 supplies Triton 3.5.1. FlagGems is pinned to official
v5.4.0 (`6ed2f39071ef85db26b7e64f0652a162f434959c`), with NumPy 2.2.6,
PyYAML 6.0.1 and cann-shmem 1.6.0. See the Dockerfile and
[runtime constraints](runtime-constraints.txt) for the dependency definitions.
Standalone `triton` and `triton-ascend` packages overwrite FlagTree's Python
module and must be removed before installing it.

## Build

From the repository root on an aarch64 build host:

```bash
docker/build.sh --platform ascend --target ci \
  --image-name vllm-plugin-fl --image-tag ascend-vllm0.32.0
```

Build isolation remains disabled for vLLM and the plugin, preserving the vendor
Torch pair. FlagGems is installed without hardware extras, which specify a
different dependency stack. The `dev` stage adds development tools; `ci` also
installs the pinned pytest plugins. The Dockerfile does not install the plugin
checkout or FlagGems-vllm. Follow the [Ascend deployment guide](../../docs/usage/ascend.md)
to install both and start eager or native decode graph serving.

## Shared CI

Publish the built image to a team-accessible registry before selecting it in
[the Ascend CI configuration](../../.github/configs/ascend.yml). Update `ci_image`
and enable Ascend in [the platform registry](../../.github/configs/platforms.yml)
together. The inherited vLLM 0.28 image is incompatible with this branch.
[The setup script](../../.github/scripts/ascend/setup.sh) checks runtime versions
and fails on a mismatch instead of testing a different vLLM version.

CI overlays the submitted checkout with an editable plugin install. Deployment
images should instead install a wheel from the selected immutable revision.
Model weights must be provisioned before E2E jobs: mount their parent directory
at `/data/models/Qwen` for the paths declared by
[the Ascend test matrix](../../tests/platforms/ascend.yaml).

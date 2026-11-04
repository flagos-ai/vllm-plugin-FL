# Ascend 910C: vLLM 0.28.0 CI image

`Dockerfile.v0.28.0` builds the validated Ascend runtime on FlagOS runtime 2.3.0.
`Dockerfile` contains the same recipe for the default Ascend build path.

Published tag: `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-ascend-ci`.
Runner label: `flagcicd-ascend-910c-k8s-vllm028`.

Published manifest: `sha256:a388a2d44ee7002b2c4b27af59f1421adbf681b32679e0b6ab5dca9edc676412`.
Published config: `sha256:a55eee4498d16b45ee6263c23b5308fb5ac24f64367f211c5c7cb9a128e4b5ff`.
The published image has sanitized history and identical runtime config and
filesystem layers to the preinstalled CI image validated above.

## Fixed stack

| Component | Version or source |
| --- | --- |
| Base | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c@sha256:6a4692bfa392be9cc97bcf5302362b9393d9867a2b59d23c873fcde4c08cca04` |
| CANN / Python | 9.0.0 / 3.11.15, aarch64 |
| torch / torch-npu | 2.10.0+cpu / 2.10.0 |
| vLLM | 0.28.0+empty, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| FlagTree / Triton | 0.6.2a1+ascend3.5 / 3.5.1; FlagTree supplies Triton |
| FlagGems | commit `3b406c36212744b98b9720bf6d0a5387c09fe96b`, package 5.0.2 |
| FL wheel / dispatch policy | commit `9a7eaf1191999422682bf5c6d1cf7b3f88d253b5` |
| cann-shmem / NumPy | 1.6.0 / 1.26.4 |

The base's CANN/PyTorch stack is retained. FlagTree and FlagGems are replaced by
these fixed versions. The FlagCX Python/Torch plugin is uninstalled; the tested
inference route uses HCCL. FlagGems uses Python/Triton operators with
`USE_C_EXTENSION=0`. The image includes a non-editable FL wheel; shared CI then
overlays the checked-out branch through its existing setup script.

The base normally prepends `/opt/flagtree` (0.7.0) to Python's search path.
`scripts/runtime-env.sh` replaces the dynamic compiler selection profile and
keeps the fixed `/flagos` installation active, including in non-login shells.
No package installation is performed by the image entrypoint.

## Build

Run on a native ARM64 Docker host from the repository root:

```bash
bash docker/build.sh --platform ascend --target ci \
  --image-name vllm-plugin-fl-ascend --image-tag 0.28.0-ci-build
```

The helper selects `Dockerfile.v0.28.0` and prepares immutable source checkouts
under `docker/ascend/sources/`. These checkouts are ignored by Git and included
in the Docker build context, so wheel versions and source manifests remain
traceable. Rebuilds reuse existing pinned checkouts.

For a direct Docker build, prepare the same context first:

```bash
bash docker/ascend/scripts/prepare-v0.28.0.sh
docker build --target ci -f docker/ascend/Dockerfile.v0.28.0 \
  -t vllm-plugin-fl-ascend:0.28.0-ci-build docker/ascend
```

Targets: `release` installs the inference stack; `ci` adds test/build tools;
`dev` aliases the prepared CI environment. This recipe validates the versions
above and rejects incompatible version build arguments.

Proxy values come from the caller's environment. If required, add
`--build-arg http_proxy --build-arg https_proxy --build-arg HTTP_PROXY
--build-arg HTTPS_PROXY` to the helper invocation. Do not put authentication
values in source files.

## CI tools

The Dockerfile embeds the dependency declarations and pins pytest 8.3.2, coverage 7.16.1, pytest-cov 7.1.0,
pytest-json-report 1.5.0, pytest-metadata 3.1.1, pytest-timeout 2.4.0,
pytest-xdist 3.6.1, decorator 5.1.1, build 1.2.2.post1 and scikit-build-core 0.11.0.
The transitive tool dependencies are also pinned in the Dockerfile.

FlagGems is built with scikit-build-core 0.12.2 using `wheel.cmake=false` because
its C++ extension is disabled. Only after that wheel is installed does the CI
stage select 0.11.0, as required by the FL plugin's build metadata.

The shared setup in this branch also installs its pinned FlagGems-vllm package
through `.github/scripts/install_flaggems_vllm.py`. That existing behavior is
preserved; it is separate from the fixed FlagGems package in this image.

## Publish

The selected base contains authenticated proxy URLs in its inherited image
history. Before publishing, scrub the metadata using the included save/load
helper. The helper asserts identical filesystem layers and runtime config,
retains the archive and writes a report. It does not execute deletion commands.

```bash
python3 docker/ascend/scripts/sanitize-image-history.py \
  --source vllm-plugin-fl-ascend:0.28.0-ci-build \
  --target harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-ascend-ci \
  --archive /path/with/free-space/ascend-ci-clean.tar \
  --report /path/with/free-space/ascend-ci-publication.json
docker login harbor.baai.ac.cn
docker push harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-ascend-ci
```

If the host Docker daemon cannot push because of its overlay mount issue, a
registry client can push the same sanitized Docker archive directly:
`crane push /path/with/free-space/ascend-ci-clean.tar
harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-ascend-ci`.

History sanitization changes the image config ID and manifest digest, not the
filesystem layers or startup configuration.

## Validation and limits

The fixed runtime above, with the listed CI tools, passed the pinned source's
complete Ascend 910C matrix on devices 14 and 15: 798 unit tests passed with
15 skips, 16 functional tests passed, Qwen3.6-27B and Qwen3.6-35B-A3B passed TP2
eager offline inference and text/image serving, and benchmark smoke completed
five requests with zero failures. The image with tools preinstalled then passed
runtime/NPU checks, official setup and wheel construction without adding tools;
164 fixed package versions matched the full test environment.

Those results apply to FL commit `9a7eaf1191999422682bf5c6d1cf7b3f88d253b5`.
They are not a claim that the later `ci/thead-0.28` branch and its separate
FlagGems-vllm setup have already passed a new full run.

The embedded runtime constraints protect the CPU torch/NPU adapter and pinned
runtime. Embedded reference constraints select common dependency versions from
the earlier validated image. OpenCV is explicitly limited to 4.11 for NumPy 1.26.4
compatibility; the vLLM dependency metadata is patched accordingly. CANN profiler
tools retain inherited optional-tool dependency gaps and OpenTelemetry conflicts;
`pip check` is not globally clean. This validation does not cover profiling tools,
model graph mode, quantization, or performance tuning.

Model weights stay outside the image. The existing Ascend workflow volume mounts
and device selection remain in `.github/configs/ascend.yml`; provision those paths
and devices on the named runner before running CI.

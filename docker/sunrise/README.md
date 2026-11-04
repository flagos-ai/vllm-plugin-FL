# Sunrise: vLLM 0.32.0 empty

This recipe targets `0.5.0-dev` and uses **Model Runner V2** for primary
acceptance (`VLLM_USE_V2_MODEL_RUNNER=1`). It retains the Sunrise/PTPU vendor stack from
[the latest published Sunrise application image](https://flagos-ai.github.io/release-info/application/vllm0.24.0-sunrise-tangrt1.2.0/)
as checked on 2026-11-04:

```text
harbor.baai.ac.cn/flagos-app/vllm0.24.0-sunrise-tangrt1.2.0:2.3.0-0.3.0
```

The image was published on 2026-10-27. It supplies TangRT 1.2.0,
Python 3.10, vendor PyTorch `2.11.0+cpu`, `torch_ptpu` `0.2.3+torch2.11`,
FlagTree `0.7.0+sunrise3.6` and FlagCX `0.14.0+tangrt1.2.0`.
The host must provide the supported Sunrise devices and driver; the release
page lists SR-SUN-S2-X1 and host driver 0.24.0.

Official vLLM 0.32.0 requires Python >=3.11. The available Sunrise stack is
cp310, so this recipe applies an explicit **third-party Python 3.10
compatibility build** to the downloaded upstream source. This is recorded in
`/opt/sunrise-stack/python310-compat.json`; it is not an official upstream
Python 3.10 support claim. The vLLM wheel is built with
`VLLM_TARGET_DEVICE=empty`, without CUDA extensions, and installed as an
ordinary `0.32.0+empty` wheel. The source archive, compatibility receipt and
wheel remain in the image for inspection.

`SUNRISE_PYTHON` can select a different existing interpreter, but it must have
matching vendor torch/torch_ptpu/compiler packages. A stock Python upgrade
cannot reuse the cp310 binary wheels. Python >=3.11 skips the cp310 patch.

## Build

Run from the repository root; all stages install this checkout of the plugin:

```bash
./docker/build.sh --platform sunrise --target ci
# Or invoke Docker directly:
docker build -f docker/sunrise/Dockerfile --target ci \
  -t vllm-plugin-fl:sunrise-0.32.0-empty .
```

Use tmux for a remote build, with a log outside the container:

```bash
tmux new-session -d -s sunrise032-build \
  'docker build -f docker/sunrise/Dockerfile --target ci -t vllm-plugin-fl:sunrise-0.32.0-empty . > /tmp/sunrise032-build.log 2>&1'
tmux capture-pane -t sunrise032-build -p
```

Arguments include `SUNRISE_BASE_IMAGE`, `SUNRISE_PYTHON`, `VLLM_VERSION`,
`VLLM_SOURCE_URL`, `VLLM_SOURCE_SHA256`, `PLUGIN_VERSION`, `INDEX_URL` and
`EXTRA_INDEX_URL`. The default source SHA256 applies to the official v0.32.0
archive. A different archive requires its independently verified checksum.
The `dev` stage adds development tools, `ci` adds test dependencies, and
`release` retains the inference runtime.
For git-only network configuration, set `SUNRISE_GIT_CONFIG` to a local git
config file when invoking `docker/build.sh`; it is mounted as an optional
BuildKit secret for the provider clone and is not copied into image layers.
If GitHub Git access is unavailable, set `SUNRISE_FLAGGEMS_VLLM_BUNDLE_DIR`
to a directory containing only the public `flaggems-vllm.bundle`, whose `HEAD`
is the pinned provider commit. This named build context is bound read-only
during provider installation; the default empty stage uses online Git instead.
Git still checks out the exact commit and the shared installer verifies the
original source identity. Only the private git configuration uses a secret;
the public source bundle is not subject to BuildKit secret size limits.

During image construction, `TORCH_DEVICE_BACKEND_AUTOLOAD=0` is scoped to
build processes so torch package metadata can be read without host devices.
It is not set in the runtime image environment.

The build snapshots native package versions before installing Python
dependencies. Resolver overrides preserve vendor torch, torch_ptpu,
FlagTree/Triton, FlagGems, FlagCX and other native packages. A subsequent
comparison rejects replacements or newly introduced CUDA packages. When
FlagTree owns the Triton import namespace without a `triton` distribution,
transitive PyPI Triton is excluded. Build and plugin installations use
`--no-build-isolation --no-deps`. The build reuses the published image's `/flagos` environment inside Docker's
isolated layers. A nested venv would lose the parent venv's vendor packages.
Compiler/runtime profiles are sourced for build layers and the container
entrypoint, then the selected interpreter takes precedence on `PATH`.
The pinned shared installer adds FlagGems-vllm `v0.2.0` commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19` without replacing native packages.

## Verify on Sunrise hardware

Select available devices and mount the actual model directory. This follows
the release page's privileged/device configuration; use the device policy
required by the host installation.

```bash
docker run --rm -it --privileged --ipc=host -v /dev:/dev \
  -e TANG_VISIBLE_DEVICES=6,7 -v /data/models:/data/models:ro \
  vllm-plugin-fl:sunrise-0.32.0-empty bash
# Inside the container:
python /opt/sunrise-build/verify_runtime.py --min-devices 2
tmux new-session -d -s sunrise032-smoke \
  'python /opt/sunrise-build/verify_runtime.py --model /data/models/Qwen3-0.6B --tensor-parallel-size 2 > /tmp/sunrise032-smoke.log 2>&1'
```

Replace the model path with a real local model. Remote inference and server
commands must also run inside tmux. The verifier requires Runner V2 and checks the constructed engine config
before inference. Save the server/worker log identifying Runner V2 as part of
acceptance; a V1 smoke result does not satisfy the primary gate. It also checks
stack versions, the
empty wheel, PTPU registration, visible devices and arithmetic on each
selected device, then optionally performs eager offline inference. Its text
checks are smoke checks; model quality and graph acceptance require the
repository's full functional/end-to-end suites and request-time replay
checks. An image build or successful import alone does not establish hardware
inference correctness.

Example eager service, using a conservative explicit per-worker KV budget:

```bash
tmux new-session -d -s sunrise032-serve \
  'vllm serve /data/models/Qwen3-0.6B --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 --enforce-eager --disable-custom-all-reduce --max-model-len 2048 --max-num-seqs 8 --kv-cache-memory-bytes 1073741824 > /tmp/sunrise032-serve.log 2>&1'
```

Adjust the model, context and KV budget to the available memory. Older PTPU
memory counters can describe whole-device usage instead of per-process
allocation peaks, so an explicit conservative `--kv-cache-memory-bytes`
avoids relying on those counters for sizing. Do not enable the historical
zero-output GDN benchmark workaround during correctness validation.

Actual acceptance results are reported separately from this reproducible
recipe. The recipe does not claim that the full model/graph matrix has passed.

# T-Head PPU CI image

## FlagOS Runtime 2.3.0 candidate

From a checkout containing the full commit in `plugin-source.ref`, run:

```bash
bash docker/thead/build-ci.sh vllm-plugin-fl-thead:0.32.0-runtime-2.3.0
```

`Dockerfile.v0.32.0` uses the official Runtime 2.3.0 image by manifest
digest. The base config digest is
`sha256:e7b8b0ac106ddabc2cf13844340e69f037807e29f30690fd6ff4794e9aa2af40`.
`requirements-ci.txt` records exact wheel URLs and SHA256 hashes for additions
to the base stack. The build archives the pinned plugin commit and builds its
wheel without resolving dependencies. Find the plugin source and wheel hash,
pip plan and install reports, before/after vendor payload hashes, and installed
package versions under `/opt/thead-ci` in the built image.

FlagTree 0.7.0+ppu3.6 provides the default Triton from `/opt/flagtree`
through `PYTHONPATH`. The vendor Triton remains in `/opt/triton`. The T-Head
FlagCX wheel supplies the graph-capable TP collective at `FLAGCX_PATH`.
The pinned plain FlagGems wheel supplies the version verified by the
multimodal A/B probe. Its wheel is installed from the lock without extras.

Run the stack checker with the normal interpreter from a clean container on four
reserved ZW810E cards:

```bash
python /opt/thead-ci/runtime_stack.py check
python tests/run.py --platform thead --device zw810e --scope unit
python tests/run.py --platform thead --device zw810e --scope functional
python tests/run.py --platform thead --device zw810e --scope e2e --stage pr
FL_CONCURRENT_MAX_TOKENS=256 python tests/run.py --platform thead --device zw810e --scope e2e --stage nightly --task concurrent
```

Map the four assigned `/dev/alixpu_ppuN` nodes plus `/dev/alixpu` and
`/dev/alixpu_ctl`, and mount the model directory at `/data/models`.
Use the visible ordinals reported inside that container.
The current PR and nightly benchmark matrices are disabled. For manual CLI
validation, use the throughput, latency and serve cases defined in
`tests/benchmarks/configs/smoke.yaml`.

The published image in `.github/configs/thead.yml` changes after this
candidate is accepted and the published digest is verified.

## Existing CI image

Builds `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-thead-ci`, the image
consumed by `.github/configs/thead.yml`.

The image carries the fixed PPU stack only. The plugin and `flaggems_vllm` are
installed at test time by `.github/scripts/thead/setup.sh`, which requires an
ordinary venv (`THEAD_BASE_PYTHON=/opt/thead/venv/bin/python`) and rejects
editable distributions.

## Contents

| Component | Version |
|---|---|
| Base | `harbor.baai.ac.cn/flagos-app/sglang0.5.12-thead-ppu2.1.0:2.2.0-0.2.0@sha256:27de587e…` (FlagOS 2.2, Ubuntu 24.04.2) |
| Python | 3.12.3 (`/opt/thead/venv`, created from the base `/usr/local` install) |
| torch | vendor 2.10.0 from the base image |
| FlagTree / Triton | `0.7.0+ppu3.6` (the only Triton 3.6 provider) |
| FlagGems | `5.5.0.dev0+g317b76320` |
| vLLM | `0.32.0+empty` (no `vllm._C`, no `_C_stable_libtorch`) |
| FlagCX | `/sgl-workspace/FlagCX` from the base image |

## Build

The three stack wheels are not on PyPI. Put them in a directory and pass it as
the `thead_wheels` build context:

```
thead_wheels/
  vllm-0.32.0+empty-cp312-cp312-linux_x86_64.whl
  flagtree-0.7.0+ppu3.6-cp312-cp312-linux_x86_64.whl
  flag_gems-5.5.0.dev0+g317b76320-py3-none-any.whl
```

```bash
cd docker
THEAD_WHEELS_DIR=/path/to/thead_wheels ./build.sh --platform thead --target ci
```

Without `THEAD_WHEELS_DIR`, FlagTree and FlagGems are resolved from
`--index-url`, but the `vllm-0.32.0+empty` wheel is still required: the build
stops with an error instead of falling back to a source build.

The build fails early if any of these do not hold, so a broken stack never
reaches CI:

- `torch.__version__ == 2.10.0`
- `importlib.metadata.version("vllm") == "0.32.0+empty"`
- `flagtree == 0.7.0+ppu3.6`, `triton.__version__ == 3.6.0`
- `flag-gems == 5.5.0.dev0+g317b76320`
- `vllm_fl` is **not** importable (the plugin must not be baked in)

## Publish

```bash
docker tag <built-image> harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-thead-ci
docker push harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-thead-ci
```

Then pin the resulting digest in `.github/configs/thead.yml`:

```yaml
ci_image: harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-thead-ci@sha256:<digest>
```

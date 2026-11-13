# T-Head PPU CI image

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

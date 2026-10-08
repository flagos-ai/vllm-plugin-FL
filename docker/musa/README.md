# MUSA CI image (vLLM 0.24)

The MUSA CI image for the vLLM 0.24 line wraps the FlagOS-packaged Moore
Threads stack:

```text
harbor.baai.ac.cn/flagrelease-public/flagrelease_mthreads-gmi_vllm024plugin_base:08281629
```

The base provides MUSA 4.3.5, torch 2.9.0 + torch_musa 2.9.0, vLLM 0.24.0
(empty-device build), FlagGems 5.3.2.post1.dev22 (`b1f939eb5`), FlagTree
`b97f8214` with the MUSA graph-capture fixes. `Dockerfile` replaces any plugin
from the base with the repository's `vllm-plugin-fl` package, adds build
tooling and CI test dependencies, and fails the build if the base does not
carry the expected vendor stack. The resulting dev, CI, and release images
work without mounting a source checkout. The v0.20.2 image that wrapped Moore
Threads' own registry image directly is kept as `Dockerfile.v0.20.2`.

## Build and push

From the repository root:

```bash
docker/build.sh --platform musa --target ci
# -> harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-musa-ci

docker push harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-musa-ci
```

`.github/configs/musa.yml` references `v0.24.0-musa-ci`, and the MUSA platform
is enabled in `.github/configs/platforms.yml`.

## Validate on a MUSA host before enabling CI

```bash
# 1. Hardware check (mthreads-gmi must be on PATH inside the container).
bash .github/scripts/musa/check.sh

# 2. Verify the preinstalled plugin and stack imports. In CI, setup.sh also
#    installs the checkout editable so pull-request changes override it.
GEMS_VENDOR=mthreads VLLM_PLUGINS=fl MTHREADS_VISIBLE_DEVICES=all \
  bash .github/scripts/musa/setup.sh

# 3. Run the same suites CI would run.
python tests/run.py --platform musa --scope unit
python tests/run.py --platform musa --scope functional
python tests/run.py --platform musa --scope e2e
```

E2E cases (`tests/platforms/musa.yaml`, device `s5000`) expect model files
under `/data/models/` (e.g. the Qwen3.6 cases used by the 0.24 line).

## Switching the vendor stack

When Moore Threads publishes a refreshed stack, rebuild without touching the
Dockerfile:

```bash
MUSA_BASE_IMAGE=harbor.baai.ac.cn/<project>/<new-stack-image>:<tag> \
  docker/build.sh --platform musa --target ci --image-tag v0.24.0-musa-ci
```

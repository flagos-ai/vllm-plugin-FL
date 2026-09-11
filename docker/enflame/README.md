# Enflame CI image (vLLM 0.24)

The vLLM 0.24 image wraps the vendor-delivered torch_gcu 2.11.0 bundle for
S60 (host driver 1.9.10, validated on-machine):

```text
harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-enflame-gems_5.4.0.dev0-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.8-torch_gcu_2.11.0_3.8.20260713-pcp_tops3.8.20260714-gpu_s60-arc_amd64-driver_1.9.10:202608141853
```

The base provides torch 2.11.0 + torch_gcu 2.11.0, triton_gcu 3.6.0,
FlagGems 5.4.0.dev0, and the S60 runtime; `Dockerfile` replaces its vLLM
with the official 0.24.0 empty-device build (vLLM 0.24 needs
`setuptools>=77` and `setuptools-rust`, both added here) and installs the CI
test dependencies. The v0.20.2 thin wrapper around the old vendor image is
kept as `Dockerfile.v0.20.2`.

Notes validated on `vm-dhrj-gd-zone7-d-s60-48g-1-5` (S60 x8, driver
1.9.10): the image's topstx 1.9.29 userspace runs fine against the 1.9.10
host driver, and the torch_gcu 2.11 stack imports with vLLM 0.24.0+empty.

## Build and push

```bash
docker/build.sh --platform enflame --target ci
# -> harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-enflame-ci

docker push harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-enflame-ci
```

`.github/configs/enflame.yml` references `v0.24.0-enflame-ci`; the platform
stays `enabled: false` in `platforms.yml` until e2e validation passes on a
runner.

## Validate on an S60 host before enabling CI

```bash
# 1. Hardware check (falls back to the torch_gcu probe when efsmi is not
#    in the container — the torch_gcu 2.11 bundle does not ship it).
bash .github/scripts/enflame/check.sh

# 2. Install the checked-out plugin and verify the stack imports.
TOPS_VISIBLE_DEVICES=2,3 bash .github/scripts/enflame/setup.sh

# 3. Run the same suites CI would run.
python tests/run.py --platform enflame --scope unit
python tests/run.py --platform enflame --scope functional
python tests/run.py --platform enflame --scope e2e --task inference \
  --model qwen3_6 --case 27b_tp2_eager
```

Container pattern for on-host runs (privileged, devices via
`TOPS_VISIBLE_DEVICES`, models under `/data`):

```bash
docker run --privileged --ipc=host --network host \
  -v /dev:/dev -v /data:/data -v "$PWD":"$PWD" -w "$PWD" \
  -e TOPS_VISIBLE_DEVICES=2,3 \
  harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-enflame-ci \
  bash -lc '<commands above>'
```

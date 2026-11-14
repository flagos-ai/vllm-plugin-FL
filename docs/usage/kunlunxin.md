# Kunlunxin P800 on `0.5.0-dev`

## Runtime requirements

| Component | Requirement |
|-----------|-------------|
| Accelerator | Kunlunxin P800 OAM (x86_64) |
| Host driver | `5.0.21.47` |
| Container XPU runtime (XRE) | `5.37.1` |
| Python | 3.10 with the FlagOS wheel and matching Kunlunxin runtime packages |
| vLLM | `0.32.0+flagos` from the Kunlunxin package index |
| PyTorch | `2.9.0+cu129` from the application image |
| FlagTree and Triton | FlagTree `0.7.0+xpu3.6`, Triton `3.6.0` |
| FlagGems | `5.4.0` |
| FlagGems-vLLM | Source tag `v0.2.0` at [`b5c2c224`](https://github.com/flagos-ai/FlagGems-vllm/commit/b5c2c2249319277ccda4d41aa0f65164bd9b2a19) |
| FlagCX | `0.14.1+xre5.37.1` for tensor-parallel inference |

The FlagOS wheel contains the vLLM Python runtime. The FlagOS P800 application
image below provides PyTorch, Triton, the driver libraries and the FlagCX
adaptor. The image's Python is in `/flagos/bin`. See the
[FlagOS application release](https://flagos-ai.github.io/release-info/application/vllm0.24.0-kunlunxin-xre5.37.1/).

```text
harbor.baai.ac.cn/flagos-app/vllm0.24.0-kunlunxin-xre5.37.1:2.3.0-0.3.0@sha256:db1cceff13d438142abfd695500ac0820ebfe6ae13d909017a2ce073ac607b17
```

The P800 image build applies the [Python 3.10 compatibility patch](../../docker/kunlunxin/python310_compat.py)
to the pinned FlagOS wheel. A source build of
[upstream vLLM 0.32.0](https://github.com/vllm-project/vllm/blob/v0.32.0/pyproject.toml)
in empty mode is also accepted by CI when the vendor runtime uses Python
3.11–3.13.

## Install

From the repository root, build the CI image with the Python 3.10 compatibility
patch applied.

```sh
bash docker/build.sh --platform kunlunxin --target ci \
  --image-name vllm-plugin-fl --image-tag kunlunxin-0.32.0-ci
```

The P800 CI jobs start from the published application image and run the
same runtime installer during setup. The derived CI image can also be
built locally for reproducing those jobs.

## Start a service

Select four available cards and mount the model directory read-only.

```sh
docker run --rm -it --ipc=host \
  --device /dev/xpuctrl --device /dev/xpu0 --device /dev/xpu1 \
  --device /dev/xpu2 --device /dev/xpu3 \
  -p 127.0.0.1:8000:8000 -v /data/models:/data/models:ro \
  --entrypoint /bin/bash vllm-plugin-fl:kunlunxin-0.32.0-ci
```

In the container, load the vendor library paths. Then clear the base image's
operator whitelist and the Bash hook that would restore it in child shells.

```sh
source /etc/bash_env.sh
unset VLLM_FL_FLAGOS_WHITELIST BASH_ENV
export GEMS_VENDOR=kunlunxin
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
PORT=8000
vllm serve /data/models/Qwen/Qwen3.6-27B --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 256 \
  --block-size 128 --gpu-memory-utilization 0.7 --enforce-eager
```

Send a request from the host after startup.

```sh
curl --fail --silent --show-error http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

The [Kunlunxin dispatch configuration](../../vllm_fl/dispatch/config/kunlunxin.yaml)
defines operator routing. The launch above uses the [P800 CI profile](../../tests/platforms/kunlunxin.yaml).
Tensor parallelism uses the FlagCX package installed in the application image.
Set `FLAGCX_PATH` only when supplying a FlagCX source build.

## Validation

Run the [P800 device check](../../.github/scripts/kunlunxin/check.sh) first.
The CI setup script checks Python and vLLM package metadata before installing
the checkout. It then checks plugin selection and availability of four P800 cards.

The Kunlunxin CI matrix runs offline inference and OpenAI-compatible serving
for Qwen3.6-27B and Qwen3.6-35B-A3B with BF16 and FlagCX TP4 in both eager
and graph modes. The [adaptation gate](../../tools/adaptation-gate-cases/README.md)
exercises 26 text, image and mixed requests for each model and mode.

The 27B TP4 inference cases also verify that Mamba `align` produces identical
greedy tokens across a block boundary with a cold cache and a reused prefix.
Run the separate hardware regression to compare against prefix caching
disabled as well:

```sh
FL_TEST_PLATFORM=kunlunxin FL_TEST_DEVICE=p800 VLLM_USE_V2_MODEL_RUNNER=1 \
  python -m pytest -q -s tests/e2e_tests/inference/test_kunlunxin_mamba_align.py
```

Speculative decoding with Mamba `align` is unsupported on P800. For
speculative decoding with these hybrid models, disable prefix caching.

Graph mode uses native full-decode graph capture with batch sizes up to
eight. Prefill and larger batches run eagerly. The platform disables
torch.compile and async scheduling. Pass `--enforce-eager` to disable
decode capture.

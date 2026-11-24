# Kunlunxin P800 on `0.5.0-dev`

## Runtime requirements

| Component | Requirement |
|-----------|-------------|
| Accelerator | Kunlunxin P800 OAM (x86_64) |
| Host driver | `5.0.21.47` |
| Container XPU runtime (XRE) | `5.37.1` |
| Python | 3.10 with the FlagOS wheel and matching Kunlunxin runtime packages |
| vLLM | `0.32.0+flagos` wheel for Python 3.10 and XRE 5.37.1 |
| PyTorch | `2.9.0+cu129` from the official FlagOS Runtime |
| FlagTree and Triton | FlagTree `0.7.0+xpu3.6`, Triton `3.6.0` |
| FlagGems | `5.5.0.dev20261124` |
| FlagGems-vLLM | [`8ae5c22`](https://github.com/flagos-ai/FlagGems-vllm/commit/8ae5c22f5af6d650216f881c329c4dff20c182e1) based on `v0.2.0`, with Kunlunxin Runner V2 sampling penalties and grammar masking |
| FlagCX | `0.14.1+xre5.37.1` for tensor-parallel inference |

The prebuilt runtime includes the versions in the table and the dependencies
needed to install the plugin from your checkout. The image's Python is in
`/flagos/bin`. Its base is the
[FlagOS Runtime 2.3.0](https://flagos-ai.github.io/release-info/zh-cn/runtime/kunlunxin-xre5.37.1/)
manifest
`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1:2.3.0@sha256:cc0deb9c09cd49977e754892e85f4414835d142379949cc2d66261f63b47959d`.

## Pull the runtime image

```sh
KUNLUNXIN_IMAGE=<validated CI image reference>
docker pull "$KUNLUNXIN_IMAGE"
```

Build the CI image with `bash docker/build.sh --platform kunlunxin --target ci`.
The build installs the vendor-provided `vllm==0.32.0+flagos` wheel, which must
support Python 3.10 and XRE 5.37.1. To use an approved wheel hosted outside
the default index, pass its URL and checksum as a pip requirement:

```bash
bash docker/build.sh --platform kunlunxin --target ci \
  --build-arg "KUNLUNXIN_VLLM_REQUIREMENT=vllm @ $WHEEL_URL#sha256=$WHEEL_SHA256"
```

The recipe builds `vllm_fl._C` against the Runtime's Torch. CI installs the
checkout with the image's existing dependencies. The Runtime's Bash profile
selects FlagTree for execution.

## Start a service

From the plugin checkout, archive the commit you want to run. Select four
available cards and mount the model directory read-only.

```sh
PLUGIN_ARCHIVE=$(mktemp --suffix=.tar.gz)
git archive --format=tar.gz -o "$PLUGIN_ARCHIVE" HEAD
docker run --rm -it --ipc=host \
  --device /dev/xpuctrl --device /dev/xpu0 --device /dev/xpu1 \
  --device /dev/xpu2 --device /dev/xpu3 \
  -p 127.0.0.1:8000:8000 -v /data/models:/data/models:ro \
  -v "$PLUGIN_ARCHIVE:/tmp/vllm-plugin-fl.tar.gz:ro" \
  --entrypoint /bin/bash "$KUNLUNXIN_IMAGE"
```

In the container's Bash session, install the plugin using the preinstalled
dependencies.

```sh
PLUGIN_SOURCE=$(mktemp -d)
tar -xzf /tmp/vllm-plugin-fl.tar.gz -C "$PLUGIN_SOURCE"
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.0.dev0 python -m pip install \
  --no-index --no-deps --no-build-isolation "$PLUGIN_SOURCE"
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

A clean container from the published digest served Qwen3.6-35B-A3B with BF16,
FlagCX TP4 and the default Runner V2 in eager mode. Installing the plugin
with the command above left the preinstalled package versions unchanged.
The chat request returned HTTP 200 with Paris as the answer.

Run the [P800 device check](../../.github/scripts/kunlunxin/check.sh) first.
The CI setup script installs the checkout and checks the vLLM version, plugin
selection, and availability of four P800 cards.

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

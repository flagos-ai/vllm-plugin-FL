# Kunlunxin P800 on `0.5.0-dev`

## Runtime requirements

| Component | Requirement |
|-----------|-------------|
| Accelerator | Kunlunxin P800 OAM (x86_64) |
| Python | 3.10 with the FlagOS wheel and matching Kunlunxin runtime packages |
| vLLM | `0.32.0+flagos` from the Kunlunxin package index |
| PyTorch | `2.9.0+cu129` from the application image |
| FlagTree | `0.7.0+xpu3.6` from the application image |
| FlagGems | `5.4.0` with FlagGems-vLLM `v0.2.0` |
| FlagCX | `0.14.1+xre5.37.1` for tensor-parallel inference |

The FlagOS wheel contains the vLLM Python runtime. The FlagOS P800 application
image below provides PyTorch, Triton, the driver libraries and the FlagCX
adaptor. The image's Python is in `/flagos/bin`. See the
[FlagOS application release](https://flagos-ai.github.io/release-info/application/vllm0.24.0-kunlunxin-xre5.37.1/).

Use the image's `vllm-serve` launcher for API serving. When invoking Python
directly, run `source /etc/bash_env.sh` first to load the vendor environment.

```text
harbor.baai.ac.cn/flagos-app/vllm0.24.0-kunlunxin-xre5.37.1:2.3.0-0.3.0
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
bash docker/build.sh --platform kunlunxin --target ci
```

The P800 CI jobs start from the published application image and run the
same runtime installer during setup. The derived CI image can also be
built locally for reproducing those jobs.

## Runtime configuration

```sh
export GEMS_VENDOR=kunlunxin
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

The [Kunlunxin dispatch configuration](../../vllm_fl/dispatch/config/kunlunxin.yaml)
enables FlagGems for SiLU, RMSNorm, and rotary embedding. Other ATen
operators keep the vendor Torch path.

The [P800 test profile](../../tests/platforms/kunlunxin.yaml) uses
`block_size=128`, `gpu_memory_utilization=0.7` and `max_num_seqs=256`.
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

Graph mode uses native full-decode graph capture with batch sizes up to
eight. Prefill and larger batches run eagerly. The platform disables
torch.compile and async scheduling. Pass `--enforce-eager` to disable
decode capture.

# MetaX Deployment Guide

Run an OpenAI-compatible service on MetaX C550 with the pinned image below. The
image contains an ordinary (non-editable) vLLM 0.28 empty wheel, FlagGems and
the MACA user-space stack; the plugin is the only thing installed on top. The
validated recipe uses four C550 cards (64 GiB each) for tensor parallelism.

## Deployment stack

| Component | Version or configuration |
|---|---|
| Image (Linux AMD64) | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-metax-ci@sha256:6c14e547566814afb5fd2e60341b8adcb442f526de797c636e257e0ab105d067` |
| Vendor base | MACA 3.7.0.38 / Ubuntu 22.04.3 / Python 3.12.11 |
| Kernel-mode driver | 3.9.25 on the validated C550 runner (3.8.1 also works) |
| vLLM | Distribution `0.28.0+empty` (no `vllm._C`); runtime `0.28.0` |
| PyTorch | `2.8.0+metax3.7.0.7` |
| Triton | `3.0.0+metax3.7.0.7` (no FlagTree on MetaX) |
| mcoplib | `0.4.5+maca3.7.0.37` |
| FlagGems | `5.3.5` |
| FlagGems-vllm | `0.1.0` from commit [b5c2c2249319](https://github.com/flagos-ai/FlagGems-vllm/commit/b5c2c2249319277ccda4d41aa0f65164bd9b2a19) |
| Model runner | `ModelRunnerFL` |
| Graph | `cudagraph_mode=PIECEWISE`, capture sizes `[1,2,4,8,16,32,64,128,256]`; `--enforce-eager` supported |
| Communication | Vendor MCCL, TP4, `--disable-custom-all-reduce` |

The empty vLLM wheel ships no CUDA extensions; do not copy `vllm._C` from
another installation. Keep the image's torch / Triton / mcoplib together — the
MACA kernels are built against `torch2.8` and `maca3.7.0.37`.

## 1. Check the host and select four idle cards

The host needs the MetaX driver, Docker, `/dev/dri` and `/dev/mxcd`. Check that
the four cards you plan to use are idle (a small driver reservation is normal):

```bash
mx-smi | head -20
mx-smi | grep -E "MiB|Processes" -A5
```

## 2. Pull the image and check the model directory

The validated models are `Qwen3.6-27B` and `Qwen3.6-35B-A3B` (about 67 GiB for
the MoE model). Each directory must contain `config.json`, the tokenizer files,
the chat template and all checkpoint shards.

```bash
export IMAGE=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-metax-ci@sha256:6c14e547566814afb5fd2e60341b8adcb442f526de797c636e257e0ab105d067
export MODEL_ROOT=/data/models
export MODEL_NAME=Qwen3.6-35B-A3B
export SERVED_MODEL=metax-qwen36
export CONTAINER=metax-qwen36-tp4
export PORT=8000

docker pull "$IMAGE"
ls "$MODEL_ROOT/Qwen/$MODEL_NAME"
```

## 3. Start the container

The image already provides vLLM, FlagGems and the MACA runtime. Install the
plugin from the checked-out source (plain editable install, no site-packages
hooks), then start a long-lived container:

```bash
docker run -d --name "$CONTAINER" \
  --hostname vllm-plugin-fl --ipc=host --shm-size=64g \
  --device /dev/dri:/dev/dri --device /dev/mxcd:/dev/mxcd \
  -e GEMS_VENDOR=metax -e VLLM_PLUGINS=fl \
  -e MACA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e FLAGGEMS_CACHE_DIR=/root/.flaggems \
  -v "$MODEL_ROOT":/data/models \
  -v "$PWD":/src \
  --entrypoint bash "$IMAGE" -lc 'sleep infinity'

docker exec "$CONTAINER" bash -lc '
  cd /src
  python -m pip install --no-build-isolation --no-deps -e .
  python .github/scripts/install_flaggems_vllm.py'
```

`FLAGGEMS_CACHE_DIR=/root/.flaggems` reuses the tuned kernel database baked
into the image. With a cold cache FlagGems re-tunes every Triton kernel at
runtime; the tuned-config keys are not stable across processes, so the selected
tiling (and therefore numeric behaviour) is not reproducible. Pin it for
serving and CI.

## 4. Start the server

```bash
docker exec "$CONTAINER" bash -lc '
  vllm serve /data/models/Qwen/Qwen3.6-35B-A3B \
    --served-model-name metax-qwen36 \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --disable-custom-all-reduce'
```

Notes:

- `--gpu-memory-utilization 0.85` is required for the 35B-A3B model at 8192
  context on a 64 GiB card; `0.95` asks for ~60.4 GiB of the 59.5 GiB that is
  actually free.
- Without graph flags the plugin selects `cudagraph_mode=PIECEWISE` through
  `PlatformFL.apply_config_platform_defaults`. Add `--enforce-eager` for the
  eager configuration; the explicit `NONE` mode is preserved.
- `--disable-custom-all-reduce` is part of the validated configuration.
- To use a different model, change the path (for example
  `/data/models/Qwen/Qwen3.6-27B`).

## 5. Smoke test

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"metax-qwen36","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":256,"temperature":0}'
```

## Tested configurations

| Model | TP | Mode | Result |
|---|---|---|---|
| Qwen3.6-27B | 4 | eager | serve + chat smoke: 2 passed |
| Qwen3.6-35B-A3B | 4 | eager | serve + chat smoke: 2 passed |
| Qwen3.6-27B | 4 | graph (PIECEWISE) | serve + chat smoke: 2 passed |
| Qwen3.6-35B-A3B | 4 | graph (PIECEWISE) | serve + chat smoke: 2 passed |

## Automated tests

```bash
# unit
python tests/run.py --platform metax --device c550 --scope unit

# one serving case (inside the container, from the source tree)
FL_TEST_PLATFORM=metax FL_TEST_DEVICE=c550 \
FL_TEST_MODEL=qwen3_6 FL_TEST_CASE=35b_a3b_tp4_eager \
python -m pytest tests/e2e_tests/serving/test_serving_smoke.py -v -s
```

## Known limitations

- `flaggems_vllm.moe_sum` has no working MACA kernel. The plugin keeps the
  `flaggems-vllm` call first and falls back to the validated
  `flag_gems.moe_sum` when the runtime vendor is MetaX; the failure is still
  raised on every other vendor.
- Only `PIECEWISE` graphs are enabled. Full cudagraphs (`FULL` /
  `FULL_AND_PIECEWISE`) and `use_inductor_graph_partition` are rejected by
  `torch 2.8.0+metax`.
- The C550 Triton backend has a 64 KiB shared-memory limit; the FLA
  `chunk_delta_h` autotuner is capped to `num_stages=1` by a vendor patch.
- vLLM's Triton top-k/top-p sampler does not compile on MACA (TTGIR SSA
  dominance). The plugin routes `apply_top_k_top_p` to the PyTorch
  implementation.
- The MetaX compatibility shims are installed from the plugin entry point
  (`vllm_fl.register`) and are gated on the MetaX runtime, not on
  `VLLM_PLUGINS`. `FL_BOOT_SHIMS_PATCH=0` disables the whole set; individual
  `FL_METAX_*_PATCH=0` switches disable one shim at a time.

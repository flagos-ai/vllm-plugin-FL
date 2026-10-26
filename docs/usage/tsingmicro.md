# Tsingmicro TXDA with vLLM 0.28.0

This backend targets TX8110/TXDA 81 devices with the vendor `torch_txda`, Triton,
FlagGems and FlagCX stack. The plugin base is `0.4.0-dev`; install a checkout
containing the TXDA 0.28 adaptation described here.

## Execution scope

- Use the FL/V1 model runner. Eager execution uses `--enforce-eager`.
  An experimental piecewise graph path is available with the limits below.
- Multi-device execution, including the TP8 examples below, requires a valid
  `FLAGCX_PATH`. Native TCCL remains the single-device fallback.
- Prefill and decode context parallel sizes must both be 1.
- Speculative decoding is unsupported because the fused speculative Mamba
  state-copy path is not available on TXDA. Omit `--speculative-config`.
- The TXDA SDPA backend uses the paged KV cache with reference attention.
  Attention, GDN/Mamba, collectives, native casts and matrix multiplication stay
  outside captured regions. MLA and sparse attention are unsupported.
- The smoke cases cover Qwen3.6-27B and Qwen3.6-35B-A3B at TP8, with a 1024-token
  context, at most 2 sequences and a 128 MiB KV-cache budget per worker. These
  bounds are smoke-test settings, not recommended production capacity limits.

## Reference environment

The starting image comes from the 2026-09-17 vLLM 0.24.0 inference record:

```text
harbor.baai.ac.cn/plugin/tsingmicro001-gems5.4.0-treenone-triton3.6.0-cx0.1.0-plugin0.3.0-vllm0.24.0-cp310-pt211-x64-v260610164501.01:202609171640
```

| Component | Reference image |
| --- | --- |
| Driver | V260610164501.01 |
| Python | 3.10 |
| PyTorch | 2.11.0+cpu with torch_txda |
| torch_txda | 0.1.0+20260807.e2f01788 |
| Triton | 3.6.0.post2026090316+git841474ad |
| FlagGems | 5.4.0.dev708+gd3a272b8a.d20260825 |
| FlagCX | 0.1.0, TXDA vendor build |
| Target vLLM | 0.28.0, empty build |

Keep the image's vendor PyTorch, `torch_txda`, Triton, FlagGems and FlagCX
packages when installing the target vLLM and plugin. The standard CUDA wheels do
not provide the TXDA runtime. `flaggems-vllm` is a separate shared operator
provider required by the current plugin; install it using the repository's
pinned installer below.

## Create an isolated container

Run on a TXDA host with the driver installed and model files already available
under `/models/Qwen`. Set `plugin_checkout` to the adapted plugin checkout.
Use a dedicated task directory and container name.

```bash
plugin_checkout=/path/to/vllm-plugin-FL
task_root=/root/tsingmicro-vllm-028
mkdir -p "$task_root/outputs"
docker run -d --name codex-tsingmicro-028 --network host --ipc=host --privileged \
  -v /dev:/dev -v /sys:/sys -v /lib/modules:/lib/modules \
  -v /models:/models:ro \
  -v "$plugin_checkout:/work/vllm-plugin-FL" \
  -v "$task_root:/work/task" \
  -w /work/vllm-plugin-FL \
  harbor.baai.ac.cn/plugin/tsingmicro001-gems5.4.0-treenone-triton3.6.0-cx0.1.0-plugin0.3.0-vllm0.24.0-cp310-pt211-x64-v260610164501.01:202609171640 \
  tail -f /dev/null
```

Put installation and inference commands in scripts under `$task_root`, then run
them through host `tmux` with `docker exec`. This keeps long jobs alive if SSH
disconnects and writes logs into the dedicated output directory.

## Install the target software

Save this as `$task_root/setup.sh` on the host:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /work/task
git clone --depth 1 --branch v0.28.0 https://github.com/vllm-project/vllm.git vllm-0.28.0
VLLM_TARGET_DEVICE=empty python -m pip install \
  --no-build-isolation --no-deps ./vllm-0.28.0
# Align the vLLM 0.28 Python requirements missing from this exact base image.
python -m pip install --no-deps \
  compressed-tensors==0.17.0 llguidance==1.7.6 fastapi==0.136.3
cd /work/vllm-plugin-FL
python -m pip install --no-build-isolation --no-deps -e .
python .github/scripts/install_flaggems_vllm.py
python -m pip show vllm vllm-plugin-fl torch torch-txda triton flag-gems flaggems-vllm
```

Launch it on the host:

```bash
tmux new-session -d -s txda-028-setup \
  "docker exec codex-tsingmicro-028 bash -lc 'bash /work/task/setup.sh' >'$task_root/outputs/setup.log' 2>&1"
tmux capture-pane -t txda-028-setup -p
tail -n 80 "$task_root/outputs/setup.log"
```

Use `bash -lc` for Python and inference commands in this image. Its login
profile supplies `TX8_DEPS_ROOT`, LLVM paths and dynamic-library paths required
by `torch_txda`; invoking Python with a plain `docker exec` does not load them.

The FlagGems-vllm installer pins release `v0.2.0` at commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`, installs that package with
`--no-deps --no-build-isolation`, and checks that the vendor runtime versions
remain unchanged. It installs missing Python support packages separately.

## Configure the runtime

Save the following as `$task_root/env.sh`. The FlagCX root shown here is the
working path in the reference image; update it for another vendor image. The
image's inherited `FLAGCX_PATH` may point at an obsolete directory.

```bash
export FLAGCX_PATH=/home/secure/260604163331/flagcx
test -f "$FLAGCX_PATH/plugin/interservice/flagcx_wrapper.py"
test -f "$FLAGCX_PATH/build/lib/libflagcx.so"
export LD_LIBRARY_PATH="$FLAGCX_PATH/build/lib:${LD_LIBRARY_PATH:-}"
export GEMS_VENDOR=tsingmicro
export VLLM_PLUGINS=fl
export VLLM_USE_V2_MODEL_RUNNER=0
export TXDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FL_BACKEND=tsingmicro
export FL_MODEL_BASE_PATH=/models
```

The plugin uses `TXDA_VISIBLE_DEVICES` for logical-to-physical device mapping.
Set the list to the allocated devices before starting the engine.

An explicit `--kv-cache-memory-bytes` takes priority over the legacy
`VLLM_TXDA_KV_CACHE_SIZE` environment variable. If neither is set, TXDA retains
its 20 GiB fallback budget and skips unsafe peak-memory profiling.

## Run the bounded smoke tests

The platform configuration is `tests/platforms/tsingmicro.yaml`. It selects both
TP8 eager model cases for offline inference and serving, including text and
locally generated image requests. To run one offline case, save this as
`$task_root/inference.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /work/task/env.sh
cd /work/vllm-plugin-FL
python tests/run.py --platform tsingmicro --device tx8110 \
  --scope e2e --task inference --model qwen3_6 --case 27b_tp8_eager \
  --output-dir /work/task/outputs
```

Run it from the host:

```bash
tmux new-session -d -s txda-028-inference \
  "docker exec codex-tsingmicro-028 bash -lc 'bash /work/task/inference.sh' >'$task_root/outputs/inference.log' 2>&1"
```

Use `--case 35b_a3b_tp8_eager` for the MoE model, or `--task serving` for the API
and image checks. Run one hardware test job at a time in the isolated container.
The serving test allows up to 180 startup polls at a 10-second interval.
Use `--dry-run` to inspect the selected cases without starting the models.

For a manual server, put this command after `source /work/task/env.sh` in a
script and launch it through the same host `tmux` pattern:

```bash
vllm serve /models/Qwen/Qwen3.6-27B \
  --served-model-name qwen --host 127.0.0.1 --port 8099 \
  --tensor-parallel-size 8 --enforce-eager --trust-remote-code \
  --disable-custom-all-reduce --max-model-len 1024 \
  --max-num-batched-tokens 1024 --max-num-seqs 2 \
  --kv-cache-memory-bytes 134217728
```

Then query the local endpoint:

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Adaptation acceptance gate

The acceptance standard is `tools/adaptation-gate-cases`, not the bounded
smoke cases above. It requires both Qwen3.6 models in both eager and graph
modes: five scenarios and 26 requests per model/mode, or 20 scenarios and
104 requests for the complete matrix. The cases include a detailed answer
with a 512-token budget, OCR, eight concurrent text requests, eight concurrent
image requests, and eight mixed requests. Do not reduce their output budgets
or change the quality assertions.

For each model, use the following eager server configuration in a host tmux
job. The gate README permits a custom serve command. Set `MODEL_PATH` to
`/models/Qwen/Qwen3.6-27B` or `/models/Qwen/Qwen3.6-35B-A3B`, and set `PORT`
to an unused port. The scheduler and KV-cache settings below accommodate
eight concurrent requests. Keep the selected device list in `env.sh`.

```bash
source /work/task/env.sh
export TXDA_LAUNCH_KERNEL_SYNC=0
export MODEL_PATH=/models/Qwen/Qwen3.6-27B
export PORT=18280
cd /work/vllm-plugin-FL/tools/adaptation-gate-cases
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 8 --enforce-eager --trust-remote-code \
  --disable-custom-all-reduce --max-model-len 4096 \
  --max-num-batched-tokens 512 --max-num-seqs 8 \
  --kv-cache-memory-bytes 1073741824 \
  --allowed-local-media-path "$PWD/images"
```

In another host tmux job with the same model and port, run the original test
script inside the same container so that local image paths resolve:

```bash
cd /work/vllm-plugin-FL/tools/adaptation-gate-cases
MODEL_PATH=/models/Qwen/Qwen3.6-27B PORT=18280 \
  REQUEST_TIMEOUT=7200 SERVICE_TIMEOUT=1800 \
  RESULTS_DIR=/work/task/outputs/adaptation-gate bash ./run_test.sh
```

`TXDA_LAUNCH_KERNEL_SYNC=0` disables the reference image's per-kernel debug
synchronization. The longer request timeout changes waiting time only. Keep
`MAX_TOKENS` unset so that the original 512/128-token request budgets apply.
Use LF line endings for the shell scripts on Linux; a Windows CRLF checkout
cannot be executed directly by Bash.

### Experimental piecewise graphs

For graph validation, replace `--enforce-eager` in the gate server command with:

```bash
--compilation-config '{"mode":3,"backend":"eager","cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
```

Here `backend="eager"` selects the compiler adapter that preserves FX operations;
TXDA device graphs still capture and replay supported regions. The plugin
selects the GGL launch path and disables per-kernel debug synchronization.
It installs partition rules in each spawned worker and warms the maximum token
batch before locking the MoE workspace.

The current vendor SDK does not reliably capture native operators or replay
multiple kernels in one graph. The plugin therefore captures audited add/mul
regions as sequences of single-kernel native graphs. Each sequence owns stable
input buffers, retained launch arguments and a private graph pool. Unsupported
regions, input layouts or changed scalar values execute eagerly. A model that
captures or reuses no live native graphs during startup is rejected explicitly.

This path has passed changing-input graph probes and full MoE model capture.
It has **not passed the model inference acceptance gate**: startup capture is
not evidence of correct inference-time replay. Keep the original request
budgets and assertions, and verify both response correctness and nonzero native
replay activity after API readiness before accepting a graph run.

## Validation status

Validated on TX8110 using the reference vendor packages and vLLM
`0.28.0+empty` built from upstream tag commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. The R6d graph candidate archive
has SHA256 `ab6df28db9181968848db71a7093a6197ccf62f112b13d9b2a190994fa81943e`.

- 284 regression tests passed locally and in the actual TXDA container.
- Both models passed TP8 eager offline text, image and seeded chunked-prefill
  smoke checks. Repeated seeded requests produced identical token IDs.
- Both models passed bounded OpenAI-compatible text/image serving with the
  default asynchronous scheduler. These checks do not establish full acceptance.
- The original long-text gate passed for each model, but the default eager
  configuration failed the eight-concurrent-request gate with a device error.
  Changing generic GEMM dispatch allowed eight requests to pass in a diagnostic
  run, but a subsequent long-text request failed. No such diagnostic exclusion
  is enabled by default.
- Isolated graph tests passed with changing FP32/BF16 tensor inputs, eager
  native casts/matmul around captured add/mul regions, and graph
  clear/recreation. Full MoE startup completed with native captures on all
  eight workers, but the first concurrent inference failed before any measured
  inference-time replay. Full graph acceptance remains unverified.

These are correctness checks, not throughput or production-capacity claims.
The vendor PyTorch, torch_txda, Triton, FlagGems and FlagCX versions were
preserved. No Tsingmicro workflow or runner configuration exists in upstream
`main` at `f901597765b3aedf4fe39d883bb9dfc32fa1f3a6`; this adaptation does not
add a hardware CI job.

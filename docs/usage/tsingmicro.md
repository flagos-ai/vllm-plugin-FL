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
- Generic FlagGems ATen replacement is limited to add/mul; four additional
  selectors retain the required MoE implementations. Normalization, rotary and
  SiLU use reference implementations. This conservative configuration can be
  slow and does not establish a throughput guarantee.
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
docker run -d --init --name codex-tsingmicro-028 --network host --ipc=host --privileged \
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
export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30
export TXDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FL_BACKEND=tsingmicro
export FL_MODEL_BASE_PATH=/models
```

The plugin uses `TXDA_VISIBLE_DEVICES` for logical-to-physical device mapping.
Set the list to the allocated devices before starting the engine.

Use `--shutdown-timeout 60` for manual servers and send SIGTERM to the API
server first. Allow at least 90 seconds for its shutdown before forcing owned
worker processes to exit. The vLLM default of zero immediately kills the engine
and workers, bypassing communicator cleanup. `--init` in the container example
also reaps any orphaned child processes.

On the reference host, some groups stalled during FlagCX warmup even with the
original 0.24 image. The tested MoE group used `NCCL_P2P_DISABLE=1`, selecting
socket transport; verify successful collectives on every allocated rank before
starting a model. This option is process-local and is not forced by the plugin.

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
  --shutdown-timeout 60 \
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
  --shutdown-timeout 60 \
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

This path passed all five original MoE graph scenarios and all 26 requests on
TP8, with native replay recorded on every worker during the formal requests.
The complete two-model, two-mode matrix is still pending. Keep the original
request budgets and assertions, and verify both response correctness and nonzero
native replay activity after API readiness before accepting a graph run.

TXDA communicator teardown explicitly disables further collectives and aborts
the underlying TCCL communicator, waiting at most five seconds. Abort failure or
timeout is logged. The vendor Abort API does not release all FlagCX host
bookkeeping, so this is not a guarantee of complete resource reclamation or safe
in-process reinitialization. Other backends retain their existing teardown.

## Validation status

Validated on TX8110 using the reference vendor packages and vLLM
`0.28.0+empty` built from upstream tag commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. The R6d graph candidate archive
has SHA256 `ab6df28db9181968848db71a7093a6197ccf62f112b13d9b2a190994fa81943e`.
The default-dispatch R7 snapshot has SHA256
`111f7c70f77a8b2bef6f2146f4682b3eeed5a75516a2e198a25b0cecdf2a0666`; its
production Python files are identical to R6d. Default registration was checked
against the equivalent R6d environment whitelist.

- The combined adaptation, communicator and shared compilation suite passed
  337 tests in the reference container; three Ascend hardware checks were
  skipped because this host has no NPU runtime. The CPU graph fixture isolates
  its offloader dependency so mocked device detection cannot leak into later
  tests. Earlier R7 default-dispatch checks also passed all 284 adaptation tests.
- Both models passed TP8 eager offline text, image and seeded chunked-prefill
  smoke checks. Repeated seeded requests produced identical token IDs.
- Both models passed bounded OpenAI-compatible text/image serving with the
  default asynchronous scheduler. These checks do not establish full acceptance.
- The original long-text gate passed for each model, but the default eager
  configuration failed the eight-concurrent-request gate with a device error.
  Changing generic GEMM dispatch allowed eight requests to pass in a diagnostic
  run, but a subsequent long-text request failed. The default now uses the
  narrower add/mul whitelist with the required MoE selectors. This combination
  passed all five original Dense eager scenarios and all 26 requests on TP8.
  Its long-text response completed normally at 309 tokens with all ten quality
  checks passing; the independent evidence summary reports PASS without
  integrity errors or verification gaps.
- Isolated graph tests passed with changing FP32/BF16 tensor inputs, eager
  native casts/matmul around captured add/mul regions, and graph
  clear/recreation. With the conservative whitelist, full MoE graph execution
  correctly answered both a capital question and a changed-input arithmetic
  question. Each of eight workers recorded 5,232 and 5,769 additional native
  kernel replays during those requests. These 12-token-budget supplemental
  requests are supplemental. The subsequent original MoE graph gate passed
  all five scenarios and all 26 requests on TP8 with P2P disabled; each worker
  recorded a sampled increase of 133,484 native kernel replays during the run.
  Dense graph and MoE eager acceptance remain pending.
- The TXDA-specific communicator teardown has 26 focused CPU regression cases;
  those and the three existing stream tests passed. They cover idempotence,
  initialization and synchronization failures, bounded waiting, library/handle
  lifetime, native ABI binding and unchanged behavior on other backends.
  On the SDK, two independent eight-rank process groups passed native copies,
  all-reduce, all-gather and explicit abort; all 16 aborts returned success in
  approximately 0.51 seconds, including when one peer delayed shutdown. An
  additional stalled group verified bounded abort on all eight ranks, but
  its communication test remained failed; this does not establish recovery of
  previously stalled devices.

These are correctness checks, not throughput or production-capacity claims.
The vendor PyTorch, torch_txda, Triton, FlagGems and FlagCX versions were
preserved. No Tsingmicro workflow or runner configuration exists in upstream
`main` at `2ed81caa5cdc43770b3272d917950c82e3db131d`; this adaptation does not
add a hardware CI job.

# Tsingmicro TXDA with vLLM 0.28.0

This backend targets TX8110/TXDA 81 devices with the vendor `torch_txda`, Triton,
FlagGems and FlagCX stack. The plugin base is `0.4.0-dev`; install a checkout
containing the TXDA 0.28 adaptation described here.

## Execution scope

- Use the FL/V1 model runner with `--enforce-eager` at every tensor-parallel
  size. Graph mode is temporarily unsupported on TXDA.
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
  MLA and sparse attention are unsupported.
- Validation covers Qwen3.6-27B and Qwen3.6-35B-A3B at TP8. The command
  examples below use bounded smoke-test settings, not recommended production
  capacity limits.

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

TXDA communicator teardown explicitly disables further collectives and aborts
the underlying TCCL communicator, waiting at most five seconds. Abort failure or
timeout is logged. The vendor Abort API does not release all FlagCX host
bookkeeping, so this is not a guarantee of complete resource reclamation or safe
in-process reinitialization. Other backends retain their existing teardown.

On the reference host, some groups stalled during FlagCX warmup even with the
original 0.24 image. The tested MoE group used `NCCL_P2P_DISABLE=1`, selecting
socket transport; verify successful collectives on every allocated rank before
starting a model. This option is process-local and is not forced by the plugin.

An explicit `--kv-cache-memory-bytes` takes priority over the legacy
`VLLM_TXDA_KV_CACHE_SIZE` environment variable. If neither is set, TXDA retains
its 20 GiB fallback budget and skips unsafe peak-memory profiling.

## Run a bounded smoke test

There is no Tsingmicro CI runner, so this adaptation adds no platform test
configuration. Validate manually against a server. Put the command below after
`source /work/task/env.sh` in a script and launch it through the same host
`tmux` pattern. Run one hardware job at a time in the isolated container.

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

Use the original cases in `tools/adaptation-gate-cases` for acceptance, in
addition to the bounded smoke check above. This eager-only delivery covers
Qwen3.6-27B and Qwen3.6-35B-A3B: five scenarios and 26 requests per model,
or 10 scenarios and 52 requests. Graph mode is outside the current scope.
The cases include a detailed answer with a 512-token budget, OCR, eight
concurrent text requests, eight concurrent image requests, and eight mixed
requests. Preserve the original budgets, image fixtures and quality assertions.

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

## Validation status

The following are historical eager results on TX8110 with the reference vendor
packages and vLLM `0.28.0+empty`, built from upstream tag commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. Each row is one complete original
five-scenario run with unchanged 512/128-token budgets, fixtures and assertions.
Both used TP8, the default asynchronous scheduler, 4096-token context, at most
eight sequences, a 512-token scheduling batch and 1 GiB of KV cache per worker.

| Model | Historical snapshot | Scenarios | Requests |
| --- | --- | --- | --- |
| Qwen3.6-27B | R6d eager | 5/5 PASS | 26/26 PASS |
| Qwen3.6-35B-A3B | R9 eager | 5/5 PASS | 26/26 PASS |

These historical runs total **10 scenarios and 52 requests**. They were not
rerun on the newly merged eager-only revision. The merged R12 candidate passed
458 CPU regression tests in the reference environment; 17 tests were skipped
under their original conditions (475 collected, pytest exit code 0). This CPU
result does not constitute a rerun of the historical hardware requests.

| Historical snapshot | Candidate archive SHA256 |
| --- | --- |
| R6d | `ab6df28db9181968848db71a7093a6197ccf62f112b13d9b2a190994fa81943e` |
| R9 | `d44944533f73613d3cce500dd2bfec41c13e2f17ba5a0360428db894a6093fb7` |

R6d selected add/mul for Dense; the R9 MoE run used the additional four MoE
selectors listed in the backend configuration. The conservative dispatch
selection follows device failures observed with broader generic FlagGems
replacement. The recorded passes establish correctness for those runs, without
a throughput or production-capacity guarantee.

The Dense eager receipt originally recorded Windows checkout hashes for five
Python files whose executed copies used LF. The corrected audit preserves the
original receipt and verifies the actual archived files: the Python bytes match
the baseline Git blobs, their ASTs and assertions are unchanged, and all image
fixtures are byte-identical. The corrected summary supports the result above.

The R9 MoE eager run completed normal worker shutdown with 16 successful native
Abort calls and no forced worker kill. Separate lifecycle checks also verified
bounded teardown, including a delayed peer. These results do not establish
recovery of stalled devices or complete SDK resource release.

Keep the reference vendor PyTorch, torch_txda, Triton, FlagGems and FlagCX
versions. This adaptation adds no Tsingmicro hardware CI job; skipped GitHub
workflows are not hardware test passes.

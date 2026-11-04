# Tsingmicro TXDA with vLLM 0.32.0

This backend targets TX8110/TXDA 81 devices with the vendor `torch_txda`,
FlagTree/Triton, FlagGems and FlagCX stack. Use the `0.5.0-dev` plugin base and
build upstream vLLM **0.32.0 in empty mode**. Preserve the vendor runtime when
installing vLLM and the plugin.

## Execution scope

- Use the upstream V2 model runner with `VLLM_USE_V2_MODEL_RUNNER=1` and
  `--enforce-eager`. V2 is the priority acceptance path for vLLM 0.32.
- Graph execution is unsupported by the current TXDA reference attention
  backend. Its metadata reads lengths and block IDs on the host and advertises
  `AttentionCGSupport.NEVER`. Removing the eager flag does not enable graphs.
- Multi-device execution requires a valid `FLAGCX_PATH`. Native TCCL remains
  the single-device fallback.
- Prefill and decode context parallel sizes must both be 1.
- Speculative decoding is unsupported. Omit `--speculative-config`.
- Attention uses paged SDPA; MLA, sparse attention, ALiBi and logits soft caps
  are unsupported. Normalization, rotary and SiLU use reference implementations.
- Generic FlagGems ATen replacement is limited to add/mul, with four extra
  selectors for MoE dispatch. This conservative configuration establishes no
  throughput guarantee.

The legacy FL/V1 Mamba copy callbacks are keyed by Mamba type in vLLM 0.32.
The retained TXDA native
tensor-copy fallback resolves the spec and callbacks for each layer, including
uniform cache groups containing different state specs. It copies only states
with upstream callbacks and preserves storage padding and overlapping ranges.
The existing packed K/V attention path is retained. The V2 runner has separate
state-management and sampling paths; the legacy copy regressions alone do not
validate V2 execution.

The image's accelerator memory query reports allocator pool sizes rather than
whole-device memory. A TXDA-only compatibility hook routes
`torch.accelerator.get_memory_info` through `torch.txda.mem_get_info` before
vLLM constructs its initial memory snapshot.

## Base image

The [application image index](https://flagos-ai.github.io/release-info/application/)
links to the [Tsingmicro vLLM 0.24 image](https://flagos-ai.github.io/release-info/application/vllm0.24.0-tsingmicro-tsm260610/).
On 2026-11-04 its documented tag `2.2.0-0.3.0rc2.post2` returned
`manifest not found`. The same Harbor repository provides the newer
`2.3.0-0.3.0`, pushed on 2026-10-26. Use this available application image:

```text
harbor.baai.ac.cn/flagos-app/vllm0.24.0-tsingmicro-tsm260610:2.3.0-0.3.0
sha256:d7bdff01d50914300d07fddf1b0b384f03d98a1bdc14ddb304f462e291ea2d6c
```

The image contains vLLM 0.24; the installation below replaces it with the
upstream 0.32 empty build. The vendor packages supplied by this image are:

| Component | Image version |
| --- | --- |
| PyTorch | 2.11.0+cpu |
| torch_txda | 0.1.0+20260728.f6fbdb71 |
| FlagTree | 0.7.0+tsingmicro3.6 |
| FlagGems | 5.4.0 |
| FlagCX | 0.14.0+tsm260610 |

Use the image's `/flagos` Python environment. The compiler and device
dependencies live under `/usr/local/llvm` and `/usr/local/tx8_deps`.
Run commands through a login shell so its environment is initialized.

## Isolated container

Use a dedicated task directory and adapted plugin checkout. Mount model files
read-only. Run only on allocated devices after checking their current users.

```bash
plugin_checkout=/path/to/vllm-plugin-FL
task_root=/root/tsingmicro-vllm-032
mkdir -p "$task_root/outputs"
IMG=harbor.baai.ac.cn/flagos-app/vllm0.24.0-tsingmicro-tsm260610:2.3.0-0.3.0

docker run -d --init --name codex-tsingmicro-032 --network host --ipc=host \
  --device /dev/accel --device /dev/accel_drv_mgr \
  -v /models:/models:ro \
  -v "$plugin_checkout:/work/vllm-plugin-FL" \
  -v "$task_root:/work/task" \
  -w /work/vllm-plugin-FL --entrypoint /bin/bash "$IMG" \
  -lc 'exec tail -f /dev/null'
```

Put long installation and inference jobs in host `tmux` sessions, for example:

```bash
tmux new-session -d -s txda-032-setup \
  "docker exec codex-tsingmicro-032 bash -lc 'bash /work/task/setup.sh' >'$task_root/outputs/setup.log' 2>&1"
tmux capture-pane -t txda-032-setup -p
```

Keep the working directory writable: the compiler backend may create its launch
log relative to the current directory.

## Experimental Python 3.10 bootstrap

This image currently supplies Python 3.10.20 and cp310 vendor extensions.
Upstream vLLM 0.32.0 requires Python >=3.11,<3.15 and imports 3.11 typing APIs
including `Self` and `Never`. The unmodified 0.32 package cannot be installed
and served in this image as-is. Replacing Python alone also invalidates the
vendor extension ABI.

The opt-in bootstrap in `tools/tsingmicro_py310/sitecustomize.py` supplies
missing `typing` symbols from `typing_extensions` and `datetime.UTC` before
vLLM is first imported, including in spawned workers. It runs only with
`FL_TXDA_PY310_COMPAT=1`, `GEMS_VENDOR=tsingmicro` and Python 3.10. Existing
stdlib attributes are preserved. No `.pth` file is installed.

This is an experimental compatibility port, not upstream Python 3.10 support.
The installation deliberately uses `--ignore-requires-python --no-deps` while
retaining upstream metadata and the cp310 vendor stack. Full imports and model
acceptance must pass before this combination can be considered validated.

## Install vLLM empty and the plugin

For the image above, save these commands in `/work/task/setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export GEMS_VENDOR=tsingmicro
export FL_TXDA_PY310_COMPAT=1
export PYTHONPATH="/work/vllm-plugin-FL/tools/tsingmicro_py310:${PYTHONPATH:-}"
cd /work/task
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm.git vllm-0.32.0
VLLM_TARGET_DEVICE=empty python -m pip install \
  --ignore-requires-python --no-build-isolation --no-deps ./vllm-0.32.0
cd /work/vllm-plugin-FL
env -u VLLM_VENDOR python -m pip install --no-build-isolation --no-deps -e .
python .github/scripts/install_flaggems_vllm.py
python -m pip show vllm vllm-plugin-fl torch torch-txda flagtree flag-gems flaggems-vllm
```

Resolve missing vLLM Python support dependencies against the 0.32 requirements
without replacing the vendor PyTorch, torch_txda, compiler, FlagGems or FlagCX.
The repository installer pins FlagGems-vllm to `v0.2.0` commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19` and verifies the existing runtime
versions remain unchanged.

## Runtime configuration

Save the following in `/work/task/env.sh`. The current image packages FlagCX
as a wheel with its ctypes wrapper in `flagcx.api` and its library under
`flagcx/lib`. The plugin also retains support for older source checkouts with
`plugin/interservice/flagcx_wrapper.py` and `build/lib/libflagcx.so`.

```bash
export GEMS_VENDOR=tsingmicro
export FL_TXDA_PY310_COMPAT=1
export PYTHONPATH="/work/vllm-plugin-FL/tools/tsingmicro_py310:${PYTHONPATH:-}"
export FLAGCX_PATH=$(python -c 'from pathlib import Path; import flagcx; print(Path(flagcx.__file__).resolve().parent)')
test -f "$FLAGCX_PATH/lib/libflagcx.so"
export LD_LIBRARY_PATH="$FLAGCX_PATH/lib:${LD_LIBRARY_PATH:-}"
export VLLM_PLUGINS=fl
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30
export TXDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FL_BACKEND=tsingmicro
export FL_MODEL_BASE_PATH=/models
export NCCL_P2P_DISABLE=1
export TXDA_LAUNCH_KERNEL_SYNC=0
```

`TXDA_VISIBLE_DEVICES` controls logical-to-physical device mapping. Validate
collectives on the allocated group before serving a model. Earlier TX8110 runs
required `NCCL_P2P_DISABLE=1` to avoid stalled FlagCX warmup; it selects socket
transport and is not forced by the plugin.

An explicit `--kv-cache-memory-bytes` overrides `VLLM_TXDA_KV_CACHE_SIZE`.
If neither is set, the plugin uses its 20 GiB fallback and skips unsafe
peak-memory profiling. Use explicit bounded settings for validation.

## Model acceptance

Use the unchanged cases in `tools/adaptation-gate-cases` for both
`/models/Qwen/Qwen3.6-27B` and `/models/Qwen/Qwen3.6-35B-A3B`. Each eager
run contains five scenarios and 26 requests: detailed text, OCR, eight
concurrent text requests, eight concurrent image requests and eight mixed
requests. Keep the original 512/128-token budgets, image fixtures and quality
assertions. Graph acceptance is outside this eager delivery.

Run one hardware job at a time in the isolated container. In a host tmux job:

```bash
source /work/task/env.sh
export MODEL_PATH=/models/Qwen/Qwen3.6-27B
export PORT=18320
cd /work/vllm-plugin-FL/tools/adaptation-gate-cases
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
  --shutdown-timeout 60 --tensor-parallel-size 8 --enforce-eager \
  --trust-remote-code --disable-custom-all-reduce --max-model-len 4096 \
  --max-num-batched-tokens 512 --max-num-seqs 8 \
  --kv-cache-memory-bytes 1073741824 \
  --allowed-local-media-path "$PWD/images"
```

From a second tmux job in the same container:

```bash
source /work/task/env.sh
cd /work/vllm-plugin-FL/tools/adaptation-gate-cases
MODEL_PATH=/models/Qwen/Qwen3.6-27B PORT=18320 \
  REQUEST_TIMEOUT=7200 SERVICE_TIMEOUT=1800 \
  RESULTS_DIR=/work/task/outputs/adaptation-gate bash ./run_test.sh
```

Then shut down the first server and repeat for the MoE model. Send SIGTERM to
the API server and allow at least 90 seconds for cleanup before forcing owned
workers to exit. The TXDA communicator bounds native Abort to five seconds;
Abort does not guarantee complete SDK resource release or safe in-process
reinitialization. Keep shell scripts in LF format.

## Validation status

The candidate passed **351 CPU tests** in one run on 2026-11-04 (Python 3.13.9,
PyTorch 2.10.0+cpu, pytest 9.0.2; exit 0, no skips). These cover TXDA platform
configuration, packed/legacy K/V numerics, Mamba copy semantics, sampling,
workspace and allocator contracts, accelerator memory routing, FlagCX wrapper
packaging, stream use and communicator cleanup. The Mamba cases include
overlapping copies, padded storage, multiple types, uniform cache groups,
noncopyable tail states and failed-plan rollback.

This CPU environment does not have vLLM installed. Real vLLM 0.32 imports,
model startup and the two-model hardware gate have **not passed**. They remain
pending resolution of the Python prerequisite and deployment of this candidate with the opt-in bootstrap.
The historical 0.28 TP8 passes do not establish 0.32 correctness.

There is no Tsingmicro hardware CI runner. Skipped GitHub jobs and successful
imports are not model acceptance passes.

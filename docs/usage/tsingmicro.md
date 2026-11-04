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
- Initial V2 validation uses `--mamba-cache-mode none` and
  `--no-enable-prefix-caching`; V2 fused state migration is a separate path
  from the retained V1 copy fallback and has not been validated.
- Attention uses paged SDPA; MLA, sparse attention, ALiBi and logits soft caps
  are unsupported. Normalization, rotary and SiLU use reference implementations.
- Generic FlagGems ATen replacement is limited to add/mul, with four extra
  selectors for MoE dispatch. This conservative configuration establishes no
  throughput guarantee.

The legacy FL/V1 Mamba copy callbacks are keyed by Mamba type in vLLM 0.32.
The retained TXDA native tensor-copy fallback resolves the spec and callbacks for each layer, including
uniform cache groups containing different state specs. It copies only states
with upstream callbacks and preserves storage padding and overlapping ranges.
The existing packed K/V attention path is retained. The V2 runner has separate
state-management and sampling paths; the legacy copy regressions alone do not
validate V2 execution. Its eager Stream/Event calls use a module-local TXDA
device view, preserving the real upstream V2 runner class and global
`torch.cuda` namespace. TXDA V2 staged writes, block-table gathering and slot
mapping use native Tensor copies and host integer metadata instead of the
vendor compiler's incorrect pointer-table kernels. This synchronizes metadata
and prioritizes correctness over throughput. V2 KV block zeroing uses real tensor views with the upstream segment geometry,
including storage offsets, packed layer spans and virtual block splitting. The
upstream pointer-table zeroing kernel left selected blocks unchanged on TXDA.
The adapter clears only selected manager blocks and validates every ID before
writing; unselected data and allocation padding are preserved.

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
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty python -m pip install \
  --ignore-requires-python --no-build-isolation --no-deps ./vllm-0.32.0
cd /work/vllm-plugin-FL
env -u VLLM_VENDOR python -m pip install --no-build-isolation --no-deps -e .
python .github/scripts/install_flaggems_vllm.py
python -m pip install --no-deps \
  transformers==5.19.0 xgrammar==0.2.8 oss-harmony==0.0.11 modelscope==1.35.3
python -m pip install pytest==9.0.2
python -m pip show vllm vllm-plugin-fl torch torch-txda flagtree flag-gems flaggems-vllm
```

Resolve missing vLLM Python support dependencies against the 0.32 requirements
without replacing the vendor PyTorch, torch_txda, compiler, FlagGems or FlagCX.
The repository installer pins FlagGems-vllm to `v0.2.0` commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19` and verifies the existing runtime
versions remain unchanged.

## Build the pinned FlagCX candidate

The current adaptation tests the official FlagCX main revision
`4a3dca96fb5a87dd6bd016e67359d7abeb0c6933` separately from the image's
0.14 wheel. This source revision builds against the existing TXDA SDK:

```bash
cd /work/task
git clone https://github.com/flagos-ai/FlagCX.git FlagCX-latest
cd FlagCX-latest
git checkout 4a3dca96fb5a87dd6bd016e67359d7abeb0c6933
git submodule update --init --depth 1 third-party/json
make -j16 USE_TSM=1 USE_BOOTSTRAP=1 COMPILE_KERNEL=0 \
  DEVICE_HOME=/usr/local/kuiper CCL_HOME=/usr/local/kuiper
```

The plugin loads the matching standalone `src/flagcx/api.py` when
`FLAGCX_PATH` names this checkout, and loads `build/lib/libflagcx.so` by
its explicit path. Installing a new Torch backend binding is unnecessary.

## Runtime configuration

Save the following in `/work/task/env.sh`. The current image packages FlagCX
as a wheel with its ctypes wrapper in `flagcx.api` and its library under
`flagcx/lib`. The plugin also retains support for older source checkouts with
`plugin/interservice/flagcx_wrapper.py` and `build/lib/libflagcx.so`.

```bash
export GEMS_VENDOR=tsingmicro
export FL_TXDA_PY310_COMPAT=1
export PYTHONPATH="/work/vllm-plugin-FL/tools/tsingmicro_py310:${PYTHONPATH:-}"
export FLAGCX_PATH=/work/task/FlagCX-latest
test -f "$FLAGCX_PATH/src/flagcx/api.py"
test -f "$FLAGCX_PATH/build/lib/libflagcx.so"
export VLLM_TXDA_SYNC_COLLECTIVES=1
export VLLM_PLUGINS=fl
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30
export TXDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FL_BACKEND=tsingmicro
export FL_MODEL_BASE_PATH=/models
export NCCL_P2P_DISABLE=1
# Use the real host interface; ens29f0 is the validated TX8110-249 setting.
export NCCL_SOCKET_IFNAME=ens29f0
unset NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_NET
export TXDA_LAUNCH_KERNEL_SYNC=0
unset VLLM_OMP_NUM_THREADS_SET_BY_VLLM
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export FL_TXDA_USE_VENDOR_GEMM=1
```

`VLLM_TXDA_SYNC_COLLECTIVES=1` explicitly enables synchronized TXDA
collective phases. The control group coordinates ranks around native stream
synchronization; all tensor communication still goes through FlagCX. Full-group
collectives execute as separate synchronized phases, including
inside logical groups. The shipped TCCL grouped path was observed to leave
later outputs unwritten; native P2P batches retain their grouping. Other backends
and the
default mode retain their original scheduling. This prioritizes correctness
and does not establish an overlap or throughput guarantee.

`TXDA_VISIBLE_DEVICES` controls logical-to-physical device mapping. Validate
collectives on the allocated group before serving a model. Earlier TX8110 runs
required `NCCL_P2P_DISABLE=1` to avoid stalled FlagCX warmup. This disables
P2P but does not disable SHM or prove Socket is selected. Inspect the actual
transport logs; the plugin does not force these transport settings.

An explicit `--kv-cache-memory-bytes` overrides `VLLM_TXDA_KV_CACHE_SIZE`.
If neither is set, the plugin uses its 20 GiB fallback and skips unsafe
peak-memory profiling. Use explicit bounded settings for validation.

## FlagCX preflight

Run the real data-path probe with at least two allocated devices before model
acceptance. It checks the library identity, float32/bfloat16 reductions, integer
aggregation, variable-size operations, broadcast and nested groups. Tensor data
stays on the FlagCX path; Gloo coordinates the CPU control group only.

```bash
source /work/task/env.sh
python /work/vllm-plugin-FL/tools/tsingmicro_py310/probe_flagcx.py \
  --world-size 2 --output /work/task/outputs/flagcx-probe.json
```

TXDA multi-device startup raises if FlagCX initialization fails and records the
loaded library and synchronization mode in the server log. A native TCCL-only
comparison does not satisfy the FlagCX acceptance gate.

## V2 preflight

After loading `env.sh`, run the real preprocessing and greedy-sampling probe
before loading a model:

```bash
python /work/vllm-plugin-FL/tools/tsingmicro_py310/probe_v2.py \
  --device 0 --output /work/task/outputs/v2-kernel-probe.json
```

It checks fused writes across two cache groups, gathered block tables with
padded rows, slot mappings across block boundaries and greedy sampling across
two Triton blocks in both float32 and bfloat16. Passing this probe does not
replace the model acceptance cases.

The block-zeroing probe compares the entire underlying allocation against an
independent physical-layout oracle, including nonzero guards and unselected
blocks:

```bash
python /work/vllm-plugin-FL/tools/tsingmicro_py310/probe_kv_zeroer.py \
  --implementation txda --dtype bfloat16 \
  --output /work/task/outputs/kv-zeroer-bf16.json
```

Repeat with `--dtype float32`. The default `--implementation upstream` tests
the unmodified upstream zeroer, which failed all four BF16 layouts in this
image. All eight native adapter cases (four layouts per dtype) passed on TXDA.

## BF16 GEMM preflight

The V2 TXDA worker explicitly selects the image's vendor implementation
`flag_gems.runtime.backend._tsingmicro.ops.mm.mm` for bias-free BF16
unquantized linear layers, including the LM head. Batch dimensions are
flattened and restored around the vendor GEMM. Other dtypes, biased layers
and autograd retain the original provider. Set `FL_TXDA_USE_VENDOR_GEMM=0`
to opt out. The first call records the provider, shapes, device and actual
PyTorch CPU thread count in the worker log.

```bash
source /work/task/env.sh
python /work/vllm-plugin-FL/tools/tsingmicro_py310/probe_vendor_mm.py \
  --bf16-only --output /work/task/outputs/vendor-mm-bf16.json
```

All 18 audited BF16 cases passed on TXDA: decode and prefill MLP shapes,
vision projection, LM head and narrow GDN projections. The independent CPU
float32 reference is rounded to BF16; the normalized L2 limit is 0.01 and
maximum absolute error limit is 0.002. The largest observed absolute error
was 0.0009765625. A further 30 black-box product GEMM cases passed across feature-strided,
transposed, expanded-row and noncontiguous-weight inputs, without modifying
input or weight storage. The separate FP32 router case failed its normalized L2
limit, so FP32 is excluded from this adapter.

Explicit CPU thread settings preserve 32 threads when vLLM creates each
worker. The 1800-second model RPC timeout permits first-use vendor kernel
compilation; it does not relax the original gate's output assertions or
512/128-token budgets.

## MRoPE lookup preflight

The V2 TXDA worker keeps upstream MRoPE rotation math but gathers its cache
rows through basic tensor slices and device copies. Only the small position
array moves to the CPU. Advanced indexing of the full cache was observed to
consume about 87% of active worker samples through the SDK CPU fallback.
Negative positions, repeated IDs and noncontiguous tensors retain Torch
lookup semantics; all indices are checked before writing the output.

```bash
source /work/task/env.sh
python /work/vllm-plugin-FL/tools/tsingmicro_py310/probe_mrope.py \
  --output /work/task/outputs/mrope-probe.json
```

All 28 hardware comparisons passed: FP32/BF16 strided and large-cache lookup,
and complete 1D/three-axis MRoPE with contiguous/interleaved sections at
1, 8 and 65 tokens. Results matched unchanged upstream CPU math exactly,
including the unrotated head tail. Selecting three rows from a 262144-by-64
cache took about 1.2 ms in the isolated probe; this is not a model throughput
measurement.

## Empty-build MoE routing

V2 supplies a padding mask to MoE routing even when a batch contains no padded
rows. The upstream masked path invokes `_moe_C.topk_softmax`, which an empty
build does not contain. TXDA routes these small logits through CPU FP32
softmax/sigmoid and stable expert selection. It retains lower-ID tie ordering,
renormalization, nonfinite-score handling, padding IDs of -1 and source-row
indices. Expert GEMMs remain on device and other backends retain their paths.

```bash
source /work/task/env.sh
python /work/vllm-plugin-FL/tools/tsingmicro_py310/probe_router.py \
  --output /work/task/outputs/router-probe.json
```

All 48 device comparisons passed against an independent scalar CPU oracle.

## GDN state-cache indexing

The TXDA V2 adapter retains the complete upstream GDN core and scopes a
`TorchDispatchMode` to its exact SSM state tensor. Eligible leading-row
gather/scatter operations use native row copies. Other tensors, storage
aliases, unknown layouts, gradients, duplicate IDs, accumulation and general
broadcasting retain their original operators. Only single-row setter values
whose leading dimension Torch removed are normalized back to one row.

```bash
source /work/task/env.sh
python /work/vllm-plugin-FL/tools/tsingmicro_py310/probe_gdn_state.py \
  --output /work/task/outputs/gdn-state-probe.json
```

The eight product-mode hardware cases passed for FP32/BF16, 32/167 padded
pages and one/eight selected rows. Whole allocations matched the CPU oracle,
including all unselected rows and padding. In an isolated FP32 167-page test,
eight-row scatter took about 2.6 ms through the scoped mode versus 2.7 s
through the advanced-index fallback. This is an operator latency comparison,
not a model throughput result. Actual V2 inference logs the first copied cache
shape, strides, device and row count.

## Model acceptance

Use the unchanged cases in `tools/adaptation-gate-cases` for both
`/models/Qwen/Qwen3.6-27B` and `/models/Qwen/Qwen3.6-35B-A3B`. Each eager
run contains five scenarios and 26 requests: detailed text, OCR, eight
concurrent text requests, eight concurrent image requests and eight mixed
requests. Keep the original 512/128-token budgets, image fixtures and quality
assertions. Graph acceptance is outside this eager delivery.

Give each model an exclusive allocated device group. Concurrent runs use
distinct physical masks and server ports; each worker owns one device. In a
host tmux job:

```bash
source /work/task/env.sh
export MODEL_PATH=/models/Qwen/Qwen3.6-27B
export PORT=18320
cd /work/vllm-plugin-FL/tools/adaptation-gate-cases
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" \
  --shutdown-timeout 60 --tensor-parallel-size 2 --enforce-eager \
  --trust-remote-code --disable-custom-all-reduce --max-model-len 4096 \
  --mamba-cache-mode none --no-enable-prefix-caching \
  --max-num-batched-tokens 512 --max-num-seqs 8 \
  --kv-cache-memory-bytes 4294967296 \
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

Repeat for the MoE model on an independent device group and unused port, or
shut down the first server before reusing its devices. Send SIGTERM to
the API server and allow at least 90 seconds for cleanup before forcing owned
workers to exit. The TXDA communicator bounds native Abort to five seconds;
Abort does not guarantee complete SDK resource release or safe in-process
reinitialization. Keep shell scripts in LF format.

## Validation status

The candidate passed **799 CPU tests** in one run on 2026-11-04 (Python 3.13.9,
PyTorch 2.10.0+cpu, pytest 9.0.2; exit 0, no skips). These cover TXDA platform
configuration, packed/legacy K/V numerics, Mamba copy semantics, sampling,
workspace and allocator contracts, accelerator memory routing, FlagCX wrapper
packaging, stream use and communicator cleanup, plus opt-in bootstrap and V2
stream namespace isolation, 0.32 worker/PP protocol, native IR priorities and
context-free SDPA page capabilities, and BF16 GEMM math, dispatch guards and
provider error propagation. KV zeroing regressions cover seven physical
layouts, virtual blocks, mixed-dtype overlays, storage lifetime, manager mapping
and complete ID validation before any write. MRoPE regressions check independent
lookup/rotation oracles, partial tails, dispatch isolation and error propagation.
MoE routing cases cover both scores, padding, nonfinite inputs, stable ties,
normalization and backend isolation. GDN cases check exact allocation bytes,
identity scoping, single-row setter normalization, overlap/gradient fallback
and unchanged upstream core execution. The Mamba cases include
overlapping copies, padded storage, multiple types, uniform cache groups,
noncopyable tail states and failed-plan rollback.

This CPU environment does not have vLLM installed. In the target image, the
vLLM 0.32 empty build imports, device memory snapshot, native streams and V2
preprocessing/sampling preflight have passed with the opt-in bootstrap.
The pinned source-built FlagCX passed the TP2 preflight: 15 checks on each rank,
including float32/bfloat16 reductions, int32/int64 aggregation, variable-size
collectives, broadcast and nested groups. All tensor communication used FlagCX.
Qwen3.6-27B has reached API readiness with both ranks reporting the actual
upstream V2 runner class and pinned FlagCX library. The two-model hardware
acceptance gate has **not passed**. The earlier inference run was stopped
to address CPU GEMM fallback performance. After the native KV zeroer passed all
eight hardware cases, profiling identified the MRoPE cache lookup bottleneck.
Its native lookup then passed 28 hardware comparisons. Subsequent diagnosis
validated the current setup on physical devices 4 and 5 with SHM enabled and
`NCCL_SOCKET_IFNAME=ens29f0`: both default-stream and nondefault-stream FlagCX
preflights passed all 30 checks. No worker stream patch was needed. Explicitly
disabling SHM and forcing Socket failed in both FlagCX and native TCCL probes
in this image; that configuration is not the acceptance setup. The pre-state-indexing
27B run passed long text, eight concurrent text requests and single-image OCR
(10/26); the 35B run with native routing passed both text scenarios (9/26).
After the scoped GDN state-copy adapter passed its eight product hardware
cases and all CPU regressions, complete acceptance is being rerun on the final
code. The new 27B run uses physical devices 6,7; 35B is being drained before
restarting with the same final code on 2,3. Results from earlier runs are kept
separate. Neither final-code model gate has passed yet.
The historical 0.28 TP8 passes do not establish 0.32 correctness.

There is no Tsingmicro hardware CI runner. Skipped GitHub jobs and successful
imports are not model acceptance passes.

Pausing a live TXDA worker for stack sampling coincided with
`tsm_jobs_ioctl` kernel NULL-pointer faults and an unreleased device context.
Do not use blocking live-process profilers for this image. The affected task
processes were stopped without resetting the shared host; subsequent checks
use other verified idle devices within the allocated group.

# Ascend graph execution with vLLM 0.32

This adapter targets upstream `0.5.0-dev` and the vLLM 0.32 v1 runner.
`FULL_DECODE_ONLY` captures and replays decode with `torch.npu.NPUGraph`;
prefill, vision encoding and sampling execute outside the decode graph.

## Runtime

The 910C validation environment uses:

| Component | Version |
|---|---|
| Python | 3.11.15, aarch64 |
| CANN / driver | 9.0.0 / 25.5.0 |
| Torch / torch-npu | 2.10.0+cpu / 2.10.0 |
| vLLM | 0.32.0+empty, built from official v0.32.0 source |
| FlagTree | 0.6.2a1+ascend3.5 (Triton 3.5.1) |
| FlagGems | 5.4.0 (`6ed2f39071ef85db26b7e64f0652a162f434959c`) |
| FlagGems-vllm | v0.2.0 (`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`) |
| NumPy / PyYAML | 2.2.6 / 6.0.1 |
| cann-shmem | 1.6.0 |

Retain the vendor Torch and torch-npu pair when installing the empty build.
Use the [Ascend Dockerfile](../docker/ascend/Dockerfile) and runtime constraints
to install vLLM's common Python dependencies without replacing this pair.
Do not install standalone `triton` or `triton-ascend` alongside FlagTree.

## Start a service

Load the CANN environment, select the assigned devices, and enable the audited
FlagGems ATen whitelist:

```bash
export VLLM_PLUGINS=fl VLLM_FL_PLATFORM=ascend
export USE_FLAGGEMS=1 VLLM_FL_FLAGOS_WHITELIST=zeros
export TRITON_ALL_BLOCKS_PARALLEL=1 TORCHINDUCTOR_COMPILE_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1
unset VLLM_FL_FLAGOS_BLACKLIST FLAGCX_PATH

vllm serve /models/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.80 --trust-remote-code \
  --allowed-local-media-path /path/to/vllm-plugin-FL/tools/adaptation-gate-cases/images \
  --no-enable-chunked-prefill --no-async-scheduling --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

This profile enables FlagGems registration and its `zeros` kernel. It does not
establish correctness for unrestricted ATen replacement. Serving uses native
HCCL with `FLAGCX_PATH` unset. The validated scope is BF16, TP2, DP1, without
expert parallelism, speculative decoding, prefix caching or chunked prefill.

## Adaptation details

- The worker and runner follow the vLLM 0.32 interfaces, dispatcher padding,
  metadata, warmup and cache initialization contracts while retaining the FL
  backend factory and platform graph runtime.
- Attention uses the native BLHNC packed cache, with 128-token attention pages
  and independent Mamba state page sizes. State views retain the allocator's
  shared storage and strides.
- GDN decode reads live device indices during replay. A row-copy kernel gathers
  and writes only active state rows from strided shared caches, preserving the
  padding slot and avoiding a full-cache materialization during capture.
- FIA task updates retain graph padding lengths; MoE workspaces are allocated
  before capture. Decode graph failures are propagated rather than silently
  falling back to eager execution.
- Padded softmax routing uses the native Ascend provider and retains vLLM's
  `-1` route sentinel. The fixed-size MoE adapter maps inactive routes to a
  valid expert with zero weight before calling the native routing kernel.
- Empty builds use native IR providers. FlagGems 5.4's duplicate `ne_` aliases
  are excluded while the canonical operators remain available; RoPE preserves
  the caller's mutation and object-identity contract.

## Gate and regression checks

Run the unchanged [adaptation gate](../tools/adaptation-gate-cases/README.md)
against the service:

```bash
cd /path/to/vllm-plugin-FL/tools/adaptation-gate-cases
MODEL_PATH=/models/Qwen3.6-27B SERVED_MODEL_NAME=Qwen3.6-27B PORT=8000 \
  ./run_test.sh
```

Repeat with `Qwen3.6-35B-A3B`, then run each model's eager control using
`--enforce-eager` in place of the graph configuration. Each model/mode has five
pytest scenarios containing 26 requests, for 104 requests in total. With
`VLLM_LOGGING_LEVEL=DEBUG`, check completed capture and actual request batches
with `cudagraph_mode: FULL` at padded sizes 1, 2, 4 and 8.

Run the platform test suites with the same interpreter and operator policy:

```bash
python tests/run.py --platform ascend --device 910c --scope unit
python tests/run.py --platform ascend --device 910c --scope functional
```

Regression coverage includes native packed-cache ownership, changing GDN
indices across graph replays, bounded state-copy memory, FIA padded lengths,
MoE expert routing, FlagGems registration and RoPE mutation semantics.

## Validation results

On 910c-201, the runtime above passes all unchanged README gate scenarios with
`USE_FLAGGEMS=1` and `VLLM_FL_FLAGOS_WHITELIST=zeros`:

| Model | Eager | FULL_DECODE_ONLY |
|---|---:|---:|
| Qwen3.6-27B | 26/26 requests | 26/26 requests |
| Qwen3.6-35B-A3B | 26/26 requests | 26/26 requests |

Both graph runs complete capture and record FULL request batches at sizes
1, 2, 4 and 8 without fallback to NONE. The installed wheel's Python files and
dispatch YAML files are checked against the source before serving validation.

The full Ascend unit suite passes 2,071 tests, with 193 platform/dependency
skips (counting repeated collection skips only once across the parallel and
serial passes). Functional validation on two visible NPUs passes 29 tests,
with 58 skips. The CI xdist run on one NPU separately covers 19 passing tests
and 68 skips, including the multi-device skips.

A standalone probe verifies the actual FlagGems `zeros` provider on 42
shape/dtype combinations and three native NPU graph replays. CPU `zeros`
retains its native provider. The CI image also exercises xdist's success and
failure reporting before the NPU interpreter shutdown workaround takes effect.

The clean CI candidate also passes both models' configured offline inference
and serving jobs (four configurations, six pytest checks), plus the serving
benchmark smoke. These checks use the CI's eager controls; graph model
validation is the separate 104-request matrix above.

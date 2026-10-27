# Ascend graph execution with vLLM 0.28

The Ascend adapter uses vLLM 0.28's v1 model runner and supports
`FULL_DECODE_ONLY` with native `torch.npu.NPUGraph` capture and replay.
Prefill, vision encoding and sampling execute outside the decode graph.

## Component versions

The 910C environment used for this adaptation has the following matrix:

| Component | Version |
|---|---|
| Python | 3.11.15 |
| CANN | 9.0.0 |
| Torch | 2.10.0+cpu |
| torch-npu | 2.10.0.post2 |
| vLLM | 0.28.0+empty, built from the official v0.28.0 source |
| FlagTree | 0.7.0+ascend3.5 |
| FlagGems | 5.4.0.dev0, commit `af7585071294075f5044ac512286de1478f16b53` |
| FlagCX | 0.13.0 installed; serving uses native HCCL |

Install FlagTree and FlagGems explicitly into an existing Ascend environment:

```bash
python3 -m pip install --no-deps 'flagtree==0.7.0+ascend3.5' \
  --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple

git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
git checkout af7585071294075f5044ac512286de1478f16b53
CMAKE_ARGS=-DFLAGGEMS_BUILD_C_EXTENSIONS=OFF \
  python3 -m pip install --no-deps --no-build-isolation -e .
```

Build vLLM from its official v0.28.0 source with `VLLM_TARGET_DEVICE=empty`.
Its common Python requirements must be satisfied while retaining the vendor
Torch, torch-npu and compiler versions above:

```bash
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.28.0+empty \
  python3 -m pip install --no-deps --no-build-isolation /path/to/vllm-0.28.0
python3 -m pip install --no-deps --no-build-isolation -e /path/to/vllm-plugin-FL
```

Select this plugin (`USE_FLAGGEMS` controls ATen replacement):

```bash
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=0
export TRITON_ALL_BLOCKS_PARALLEL=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset FLAGCX_PATH
```

The adapter ports main's `GraphRuntimeController` and native attention task
updates onto `0.4.0-dev`, retaining its vLLM 0.28 factory, activation, allocator
and vision-attention contracts. Unquantized grouped MoE uses fixed-size native
routing for ordinary, bias-free experts. GDN uses the same reference math in eager and graph modes.
vLLM 0.28's global Qwen Triton warmup is skipped because these kernels are
replaced by the Ascend GDN implementation; other warmup and capture still run.

The validated serving configuration uses `USE_FLAGGEMS=0`. FlagTree, FlagGems
and FlagCX are installed. `USE_FLAGGEMS=1` is outside this passing configuration;
previous 0.28 adaptation experiments encountered incorrect text/image outputs
with ATen replacement enabled. The complete model gate must pass separately
before enabling it. Graph validation here covers BF16, TP2, DP1 without expert
parallelism, speculative decoding, ubatching, prefix caching or chunked prefill.

## Model gate

For each model in `tools/adaptation-gate-cases/README.md`, start a service with
the following settings, substituting its model path, device pair and port:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm serve /models/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.80 --trust-remote-code \
  --allowed-local-media-path /path/to/vllm-plugin-FL/tools/adaptation-gate-cases/images \
  --no-enable-chunked-prefill --no-async-scheduling --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

Run the unmodified gate client:

```bash
cd /path/to/vllm-plugin-FL/tools/adaptation-gate-cases
MODEL_PATH=/models/Qwen3.6-27B SERVED_MODEL_NAME=Qwen3.6-27B PORT=8000 \
  ./run_test.sh
```

Repeat with `Qwen3.6-35B-A3B`. For each eager control, replace the graph
configuration with `--enforce-eager` and retain its results separately.
Each model/mode has five pytest scenarios containing 26 requests.

With `VLLM_LOGGING_LEVEL=DEBUG`, verify both completed graph capture and real
request batches with `cudagraph_mode: FULL`. Startup alone does not verify replay.

The adapter keeps GDN decode state indices on device, updates native attention
tasks with live lengths, preserves padded FIA query lengths, and reserves MoE
workspace before capture locks its addresses. vLLM 0.28's large-batch top-k/top-p
filter uses the native Ascend implementation, outside model graph execution.

## Validation results

On 910c-201 with the component matrix above, TP2, BF16 and
`USE_FLAGGEMS=0`, all unchanged README gate cases passed:

| Model | Eager control | FULL_DECODE_ONLY |
|---|---:|---:|
| Qwen3.6-27B | 26/26 requests | 26/26 requests |
| Qwen3.6-35B-A3B | 26/26 requests | 26/26 requests |

Both graph runs recorded real FULL replay batches at padded batch sizes
1, 2, 4 and 8, without fallback to NONE. The eager
runs are separate controls; graph decode does not fall back to eager.

Additional checks cover CPU graph/runtime contracts, shared hybrid cache views,
factory and activation interfaces, twelve native attention/GDN/MoE/vision NPU
checks, Qwen3-0.6B one-token prompts, and Qwen3.5-4B PP2 requests with layer
partition `31,1`. MoE capture tests change the live expert routes between replays
and compare against a CPU reference. `USE_FLAGGEMS=1` is outside this passing
configuration.

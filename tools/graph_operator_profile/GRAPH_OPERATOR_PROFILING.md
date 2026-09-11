# Runtime Operator Profiling

This tool profiles one vLLM serving scenario at a time. It sends one warmup
batch, starts the vLLM profiler, sends one measured batch, stops the profiler,
and extracts the rank-0 runtime trace.

The default workload is fixed in `run_concurrent_requests.py`: 64 concurrent
requests, 4096 input tokens, and 256 output tokens per request. CUDA Graph
capture and the warmup batch are outside the profiling window.

Run commands from the vllm-plugin-FL repository root.

## 1. Start one server

Choose a new run directory for every invocation. The directory used by
`torch_profiler_dir` must be `<run-dir>/profile`.

The scenario is selected with two controls:

| Scenario | `VLLM_PLUGINS` | vLLM argument |
|---|---|---|
| plugin graph | `fl` | none |
| plugin eager | `fl` | `--enforce-eager` |
| native graph | empty | none |
| native eager | empty | `--enforce-eager` |

### Qwen plugin graph example

```bash
RUN_DIR=/vllm-workspace/graph_operator_profile_runs/qwen_plugin_graph_4096_256
mkdir -p "$RUN_DIR/profile"
printf -v PROFILER_CONFIG \
  '{"profiler":"torch","torch_profiler_dir":"%s/profile","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_with_memory":false,"ignore_frontend":true}' \
  "$RUN_DIR"

VLLM_PLUGINS=fl VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
vllm serve /models/Qwen3.6-35B-A3B \
  --served-model-name qwen \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cudagraph_num_of_warmups":0}' \
  --profiler-config "$PROFILER_CONFIG"
```

### DeepSeek plugin eager example

```bash
RUN_DIR=/vllm-workspace/graph_operator_profile_runs/deepseek_plugin_eager_4096_256
mkdir -p "$RUN_DIR/profile"
printf -v PROFILER_CONFIG \
  '{"profiler":"torch","torch_profiler_dir":"%s/profile","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_with_memory":false,"ignore_frontend":true}' \
  "$RUN_DIR"

VLLM_PLUGINS=fl VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
vllm serve /models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --safetensors-load-strategy prefetch \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --enforce-eager \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cudagraph_num_of_warmups":0}' \
  --profiler-config "$PROFILER_CONFIG"
```

For a native vLLM run, change `VLLM_PLUGINS=fl` to `VLLM_PLUGINS=""`. For
graph mode, omit `--enforce-eager`. Other model-specific vLLM arguments are
independent of this profiling tool.

## 2. Collect one profile

In another terminal, pass the served model name and the same run directory:

```bash
bash tools/graph_operator_profile/profile.sh qwen \
  /vllm-workspace/graph_operator_profile_runs/qwen_plugin_graph_4096_256
```

The script waits for `/health`, runs the warmup batch, profiles the second
batch, and writes results to `<run-dir>/results`. It rejects a profile directory
that already contains a trace so separate runs cannot be mixed accidentally.

## Output

`operator_list.csv` is the deduplicated operator-to-kernel inventory:

- `operator_id`
- `operator_name`
- `operator_kind`
- `kernel_name`

Every physical kernel is retained. Missing attribution is represented by
`operator_name=null`; it is never dropped. ATen operators appear first and pure
communication operators appear last. Communication operators still receive
IDs. Different `torch_compile` or `triton_compiled` kernel names receive
different IDs. Custom kernel specializations with the same normalized callable
name share one ID. `moe_align_block_size_stage*` kernels share one ID.

`kernel_time.csv` contains one row per unique operator/kernel relation:

- `operator_name`
- `kernel_name`
- `kernel_call_count`
- `kernel_time_us`
- `percent`

`percent` uses the total rank-0 runtime kernel duration as its denominator.

`kernel_shape_dtype.csv` contains every kernel/operator/shape/dtype/mapping
combination. Missing shape or dtype values are written as `null` with an
explicit `mapping_status` instead of removing the kernel.

`summary.json` records trace scope, event counts, mapping coverage, memcpy and
memset activity, and conservation checks. A valid extraction requires every
value in `conservation` to be `true`.

## Coverage boundary

The CSV files cover every GPU kernel event in the selected rank-0 runtime
trace. They do not include CUDA Graph construction or CPU-only operators.
Graph replay usually lacks the original PyTorch CPU events, so some graph
kernels cannot recover operator names, input shapes, or dtypes; these kernels
remain present with `null` metadata.

Rank 0 is a reproducible single-rank view. Different ranks, requests, sequence
lengths, sampling settings, and MoE routing can activate different kernels and
shapes.

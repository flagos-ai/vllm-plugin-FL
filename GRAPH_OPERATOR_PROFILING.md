# Graph-Mode Operator Collection

This procedure collects logical operator names, input shapes, input dtypes,
call counts, and runtime GPU timing in vLLM graph mode. Run all commands from
the vllm-plugin-FL repository root.

The fixed workload is 64 concurrent requests. Every request has exactly 4096
input tokens and exactly 1024 output tokens. Prefix caching is disabled and
`ignore_eos` is enabled so every response reaches the configured output length.

The final inventory combines two sources:

- `actual_capture`: operators executed while each CUDA Graph is constructed.
- `runtime`: operators and GPU events observed between the upstream
  `/start_profile` and `/stop_profile` endpoints.

Capture traces provide logical operators hidden by runtime CUDA Graph replay.
Runtime traces provide request-time operations outside graph capture and are
the only source used for execution timing.

All ranks are collected. The supplied parser selects rank 0 when producing the
default CSV results; nonzero-rank raw traces remain available for audit.

## DeepSeek-V4-Flash

Terminal A:

```bash
bash tools/graph_operator_profile/serve_deepseek_v4_flash.sh
```

Terminal B, after the server is ready:

```bash
bash tools/graph_operator_profile/profile_request.sh \
  deepseek_v4_flash \
  tools/graph_operator_profile/deepseek_v4_flash_request.json
```

Results:

```text
/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash/results/
```

## Qwen3.6-35B-A3B

Terminal A:

```bash
bash tools/graph_operator_profile/serve_qwen3_6_35b_a3b.sh
```

Terminal B, after the server is ready:

```bash
bash tools/graph_operator_profile/profile_request.sh \
  qwen3_6_35b_a3b \
  tools/graph_operator_profile/qwen3_6_35b_a3b_request.json
```

Results:

```text
/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b/results/
```

## Warmup isolation

`profile_request.sh` executes this sequence:

1. Build and save one deterministic 4096-token prompt.
2. Run a complete 64-request, 4096/1024 warmup batch and wait for all responses.
3. Call `/start_profile` only after every warmup response has returned.
4. Run the complete 64-request profiled batch with the saved token IDs.
5. Validate every response reports 4096 prompt tokens and 1024 completion tokens.
6. Call `/stop_profile` only after every profiled response has returned.

The warmup batch is never inside the runtime profiler window. Its responses
and metrics are retained beside the profiled responses and metrics.

## Primary result files

Use `operator_summary.csv` as the combined inventory. It has one row per
`(operator, input_shapes, input_dtypes)` combination:

| Field | Meaning |
|---|---|
| `operator` | Logical operator or compiled logical-call name |
| `input_shapes` | Profiler `Input Dims`, in argument order |
| `input_dtypes` | Profiler `Input type`, aligned with argument positions |
| `actual_capture_call_count` | Observations across selected-rank capture traces |
| `runtime_call_count` | CPU operator calls in the selected-rank runtime trace |
| `runtime_cpu_duration_total_us` | Sum of runtime CPU event durations; nested events make this non-additive |
| `runtime_kernel_event_count` | Runtime kernel events attributed by `External id` |
| `runtime_kernel_time_total_us` | Sum of attributed runtime kernel durations |
| `runtime_kernel_time_avg_per_call_us` | Attributed kernel time divided by runtime CPU call count |
| `runtime_kernel_time_pct_of_all_runtime_kernels` | Share of all rank-0 runtime kernel activity |
| `observed_in` | `actual_capture`, `runtime`, or both |

`runtime_operator_timing.csv` contains only runtime rows and is sorted by
attributed kernel time. Timing is never copied from capture traces.

Do not deduplicate by operator name alone. One operator can have multiple shape
or dtype combinations.

## Audit result files

| File | Purpose |
|---|---|
| `actual_capture_operator_shape_dtype.csv` | Every selected-rank capture `cpu_op`, including rows without input metadata |
| `runtime_operator_shape_dtype.csv` | Every selected-rank runtime `cpu_op`, including rows without input metadata |
| `*_effective_operator_shape_dtype.csv` | Rows that contain both shape and dtype |
| `*_operator_shape_dtype_by_rank.csv` | Audit output with an explicit rank column |
| `runtime_gpu_event_summary.csv` | Every runtime kernel, memcpy, and memset, including attribution status and duration |
| `runtime_unattributed_kernel_summary.csv` | Runtime kernels that cannot be mapped to an operator with shape and dtype |
| `summary.json` | Trace, operator, GPU event, timing, and attribution coverage counts |

Kernel attribution uses the profiler `External id`. An attributed kernel must
have a matching runtime `cpu_op`, and that CPU event must contain both
`Input Dims` and `Input type`. The coverage denominator is all rank-0 runtime
kernel duration, including kernels inside CUDA Graph replay.

Unattributed kernels remain in the audit output. Common causes are CUDA Graph
replay, fused Inductor or Triton kernels, library-internal kernels, asynchronous
launches with no matching CPU event, CPU events without input metadata, and
events crossing the profiler boundary. A CUDA Graph kernel may be visible at
runtime while its logical operator and shape are visible only during capture;
the two must not be joined by kernel name alone.

## Collection behavior

Capture profiling is enabled by the server scripts through:

```bash
export VLLM_FL_GRAPH_CAPTURE_PROFILE_DIR=.../capture_traces
```

When the variable is unset, the original `_dummy_run` path executes directly.
When it is set, an explicit profiler branch wraps the same `_dummy_run` call.
Every distributed rank writes a separate `graph_capture_rank_<N>_...` trace.

The upstream `/start_profile` and `/stop_profile` behavior is unchanged and
also produces one runtime trace per rank. The parser invocation is:

```bash
python3 tools/graph_operator_profile/extract_operator_shapes.py \
  --capture <capture_trace_directory> \
  --runtime <runtime_trace_directory> \
  --rank 0 \
  --output-dir <result_directory>
```

Change `--rank` to inspect another rank. Rank selection affects generated CSV
files only; it does not delete raw traces.

## Graph-mode details

The launch scripts do not set `cudagraph_mode`. vLLM 0.24 resolves the tested
configuration to its default `FULL_AND_PIECEWISE` mode.

Both `max_num_seqs` and workload concurrency are 64. The launch scripts capture
CUDA Graph token sizes 1, 2, 4, 8, 16, 32, and 64.

Native vLLM 0.24 automatically enables breakable CUDA Graph for DeepSeek-V4
unless `VLLM_USE_BREAKABLE_CUDAGRAPH` is explicitly set. Breakable FULL replay
causes a CUDA illegal memory access in the tested software combination, so the
scripts use:

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

This selects the standard CUDA Graph path; it does not enable eager mode.

The implementation does not use `sitecustomize` or `logical_reference`.

The NVIDIA dispatch configuration blacklists FlagGems `linear` and `mm`.
DeepSeek-V4 uses dynamic logits-projection shapes during concurrent prefill and
decode; FlagGems 5.3.4 autotuning can hit an illegal memory access for those
shapes. Blacklisting only `linear` is insufficient because the native PyTorch
linear path decomposes to the FlagGems `mm` implementation. The two entries
select the native PyTorch implementations and keep graph mode enabled.

## Coverage limits

For a fixed model, TP size, capture-size set, request, and selected rank, the
parser retains every emitted trace, every profiler `cpu_op`, and every runtime
GPU kernel/memcpy/memset event.

- Capture expands the logical shape inventory but does not provide runtime
  execution time.
- CUDA Graph replay hides many runtime logical operators, so not every runtime
  kernel can be assigned to a logical operator and input shape.
- Different prompt lengths, batch sizes, concurrency, prefill/decode lengths,
  sampling settings, or capture sizes can activate additional shapes or paths.
- Rank 0 may not represent an asymmetric path on another rank. Parse that rank
  separately when per-rank behavior matters.
- Some profiler annotations do not contain `Input Dims` or `Input type`. They
  remain in the non-`effective` CSV and are reported in `summary.json`.
- Summed kernel duration measures GPU activity, not elapsed wall time. Concurrent
  kernels can overlap, and CPU event durations can be nested.
- Capture call counts describe graph construction and are not per-request calls.

If extraction is empty, check `serve.log`, confirm that the server reached
`FULL_AND_PIECEWISE`, verify that `/stop_profile` completed, and check for
rank-specific `*.pt.trace.json.gz` files under the run's profile directory.

# Graph-Mode Operator Collection

This procedure collects logical operator names, input shapes, input dtypes, and
call counts in vLLM graph mode. Run all commands from the vllm-plugin-FL
repository root.

The final inventory combines two sources:

- `actual_capture`: operators executed while each CUDA Graph is constructed.
- `runtime`: operators observed between the upstream `/start_profile` and
  `/stop_profile` endpoints for one fixed request.

Capture traces are required because runtime replay does not expose every
logical operator inside a CUDA Graph. Runtime traces are also required because
request-time operations outside captured graphs do not appear during capture.

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

## Result files

Use `operator_summary.csv` as the primary inventory. It has
one row per `(operator, input_shapes, input_dtypes)` combination:

| Field | Meaning |
|---|---|
| `operator` | Logical operator or compiled logical-call name |
| `input_shapes` | Profiler `Input Dims`, in argument order |
| `input_dtypes` | Profiler `Input type`, aligned with argument positions |
| `actual_capture_call_count` | Observations across selected-rank capture traces |
| `runtime_call_count` | Observations in the selected-rank runtime trace |
| `observed_in` | `actual_capture`, `runtime`, or both |

Do not deduplicate by operator name alone. One operator can have multiple shape
or dtype combinations.

Additional files:

| File | Purpose |
|---|---|
| `actual_capture_operator_shape_dtype.csv` | Every selected-rank capture `cpu_op`, including rows without input metadata |
| `runtime_operator_shape_dtype.csv` | Every selected-rank runtime `cpu_op`, including rows without input metadata |
| `*_effective_operator_shape_dtype.csv` | Rows that contain both shape and dtype |
| `*_operator_shape_dtype_by_rank.csv` | Audit output with an explicit rank column |
| `summary.json` | Trace, event, row, operator-name, and missing-metadata counts |

## Collection behavior

Capture profiling is enabled by the server scripts through:

```bash
export VLLM_FL_GRAPH_CAPTURE_PROFILE_DIR=.../capture_traces
```

When the variable is unset, the original `_dummy_run` path executes directly.
When it is set, an explicit profiler branch wraps the same `_dummy_run` call.
Every distributed rank writes a separate `graph_capture_rank_<N>_...` trace.

The upstream `/start_profile` and `/stop_profile` behavior is unchanged and
also produces one runtime trace per rank. `profile_request.sh` invokes:

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

The launch scripts capture CUDA Graph token sizes 1, 2, 4, 8, 16, 32, and 64.

Native vLLM 0.24 automatically enables breakable CUDA Graph for DeepSeek-V4
unless `VLLM_USE_BREAKABLE_CUDAGRAPH` is explicitly set. Breakable FULL replay
causes a CUDA illegal memory access in the tested software combination, so the
scripts use:

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

This selects the standard CUDA Graph path; it does not enable eager mode.

The implementation does not use `sitecustomize` or `logical_reference`.

## Coverage limits

For a fixed model, TP size, capture-size set, request, and selected rank, the
parser retains every emitted trace and every profiler `cpu_op` event.

- Different prompt lengths, batch sizes, concurrency, prefill/decode lengths,
  sampling settings, or capture sizes can activate additional shapes or paths
  and must be profiled separately.
- Rank 0 may not represent an asymmetric path on another rank. Parse that rank
  separately when per-rank behavior matters.
- Some profiler annotations do not contain `Input Dims` or `Input type`. They
  remain in the non-`effective` CSV and are reported in `summary.json`.
- Inductor or Triton fused kernels do not normally expose a one-to-one mapping
  between runtime kernel events and high-level input shapes. This output is a
  logical-operator inventory, not a kernel-argument inventory.
- Capture call counts describe graph construction and are not per-request call
  counts.

## Validated results

| Model | Raw capture traces | Raw runtime traces | Rank-0 capture traces | Rank-0 runtime traces | Union rows/names |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash, TP8 | 88 | 8 | 11 | 1 | 1273 / 94 |
| Qwen3.6-35B-A3B, TP2 | 22 | 2 | 11 | 1 | 675 / 75 |

Each rank produced eleven capture traces in the validated runs. PIECEWISE
captured all seven configured token sizes, while FULL decode captured one graph
because `max_num_seqs` is 1. Both models used `FULL_AND_PIECEWISE`, returned
four completion tokens, and completed without a CUDA error, illegal memory
access, traceback, or engine initialization failure.

If extraction is empty, check `serve.log`, confirm that the server reached
`FULL_AND_PIECEWISE`, verify that `/stop_profile` completed, and check for
rank-specific `*.pt.trace.json.gz` files under the run's profile directory.

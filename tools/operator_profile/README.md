# Operator Profiling

This directory profiles one serving scenario at a time by using vLLM's native
`/start_profile` and `/stop_profile` endpoints. It does not alter vLLM's rank
policy: every rank profiles normally, while the offline extractor reads rank 0
only.

The workload is fixed at 64 concurrent requests with 4096 input tokens and 256
output tokens per request. The first batch is warmup. Only the second batch is
inside the profiling window.

Run all commands from the vllm-plugin-FL repository root. Use a stable directory
under `/workspace/op_profile` for each scenario and replace that directory when
rerunning it. Directory names use the complete normalized model name, followed
by implementation, execution mode, input length, and output length. Replace
hyphens with underscores and preserve meaningful version dots, for example
`qwen3.6_35b_a3b_plugin_graph_4096_256` and
`deepseek_v4_flash_plugin_graph_4096_256`.

## Start a server

`serve.sh` is a thin wrapper around `vllm serve`. It adds only the native torch
profiler configuration and output-directory checks. Model selection and other
vLLM arguments remain explicit command-line inputs.

Qwen plugin graph example:

```bash
RUN_DIR=/workspace/op_profile/qwen3.6_35b_a3b_plugin_graph_4096_256
VLLM_PLUGINS=fl USE_FLAGTUNE=0 bash tools/operator_profile/serve.sh \
  /models/Qwen3.6-35B-A3B qwen "$RUN_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config \
    '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

Qwen native-vLLM graph baseline:

```bash
RUN_DIR=/workspace/op_profile/qwen3.6_35b_a3b_native_graph_4096_256
VLLM_PLUGINS="" bash tools/operator_profile/serve.sh \
  /models/Qwen3.6-35B-A3B qwen "$RUN_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config \
    '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

DeepSeek plugin graph example:

```bash
RUN_DIR=/workspace/op_profile/deepseek_v4_flash_plugin_graph_4096_256
VLLM_PLUGINS=fl USE_FLAGTUNE=0 bash tools/operator_profile/serve.sh \
  /models/DeepSeek-V4-Flash deepseek-v4-flash "$RUN_DIR" \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --safetensors-load-strategy prefetch \
  --no-async-scheduling \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config \
    '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

The examples do not set `VLLM_USE_BREAKABLE_CUDAGRAPH`; vLLM retains its native
model- and version-specific selection. Omit `--enforce-eager` for graph mode and
add it only for an eager run.

`USE_FLAGTUNE=0` disables the optional remote FlagTune cost model while keeping
FlagGems kernels and their local Triton autotuning enabled. It makes offline runs
independent of FlagTune model downloads. Remove it when remote FlagTune models
are available and intentionally part of the test.

Before startup, `serve.sh` removes an earlier FlagGems evidence file. Its path
is read from `FLAGGEMS_ENABLE_OPLIST_PATH`, defaulting to
`<run-dir>/flaggems_enable_oplist.txt`. The server exports this path to its
workers, and `profile.sh` derives the same default from the run directory.
Set the same exported value in both terminals only when using a custom path.
This keeps concurrent runs from sharing evidence and prevents stale plugin
evidence from entering a native baseline.

## Collect one profile

Run this in another terminal after the server starts:

```bash
bash tools/operator_profile/profile.sh qwen \
  /workspace/op_profile/qwen3.6_35b_a3b_plugin_graph_4096_256
```

The script waits for server health, sends one warmup batch, calls the native
`/start_profile` endpoint, sends the measured batch, calls `/stop_profile`, and
then parses rank 0. An output directory containing an existing trace is rejected.

## Output files

- `operator_list.csv`: normalized operator inventory, stable operator IDs, and
  each row's share of total rank-0 runtime kernel time. Attributed and
  unattributed events for the same operator ID and kernel are merged, preferring
  the attributed operator name. `1.23` means `1.23%`;
  the column header carries the `%` unit, and values below `0.01%` are written
  as `<0.01`. The denominator is the sum of profiled GPU kernel durations;
  CPU time and end-to-end wall time are not included.
- `kernel_time.csv`: API/kernel call counts, durations, and runtime time shares.
  Its columns are `operator_name,refer,kernel_name,kernel_call_count,kernel_time_us,percent`.
  `refer` is `eager-refer` if any API attribution in that row uses an eager
  reference, including merged direct/reference records; otherwise it is `null`.
  A `null` reference marker does not imply successful API attribution.
- `kernel_shape_dtype.csv`: kernel/operator/shape/dtype mapping variants.
- `summary.json`: trace scope, mapping coverage, and conservation checks.
- `flaggems_enable_oplist.txt`: current plugin evidence, when generated.

Missing operator attribution remains `null` unless a unique eager reference API
is supplied. Missing runtime shape or dtype remains `null` with an explicit
mapping status. CUDA Graph replay does not expose every kernel's
input tensors, so graph-mode shape/dtype coverage is best effort; no kernel is
dropped. Every check under
`summary.json.conservation` must be `true`.

Mapping and grouping rules are isolated in `rule/rule_map.py`. Coverage rules
are isolated in `rule/rule_coverage.py`. ATen uses its API name as the operator
identity. A custom API uses its API name and concrete kernel function, grouping
template specializations of that function; generic launch wrappers retain the
full callable. Unattributed kernels retain their full callable unless a
verified kernel-family rule applies. Triton compiled kernels remain distinct.
The full kernel name is always retained in the CSV, so grouping does not drop
kernel events, shapes, calls, or time.

## Optional eager references for graph results

Supply the results of an eager run from the same model, implementation
(native or plugin), GPU platform, software versions, and relevant serving
configuration. Only the execution mode should differ. Historical result files
do not contain a complete environment manifest, so the caller must verify this
condition; the extractor checks rank and runtime scope and records the source
CSV checksum. It does not automatically select or verify an environment.

For an existing graph trace:

```bash
python3 tools/operator_profile/extract_operator_shapes.py \
  --trace-dir /workspace/op_profile/qwen3.6_35b_a3b_plugin_graph_4096_256/profile \
  --output-dir /workspace/op_profile/qwen3.6_35b_a3b_plugin_graph_4096_256/results \
  --eager-reference /workspace/op_profile/qwen3.6_35b_a3b_plugin_eager_4096_256/results
```

For a new graph profile, pass the eager results directory as the third argument
to `profile.sh`.

Matching uses the exact full kernel name, including template parameters.
Inductor-generated `triton_poi_*`, `triton_red_*`, `triton_per_*`, and
`triton_for_*` kernels are excluded from reference matching. Handwritten
Triton kernels remain eligible. No capture data or graph-name fallback is added.

`operator_list.csv` always contains five columns: `operator_id`, `operator_name`,
`operator_kind`, `kernel_name`, and `kernel_time_percent(%)`. Only rows whose
original kind is `unattributed` can receive an eager label. When candidates
exist, `operator_kind` becomes `eager-refer` and `operator_name` contains the
sorted, deduplicated API names joined by commas, without JSON brackets. For
example, `aten::zero_,aten::zeros` means both APIs are candidates, not that both
executed. CSV quoting keeps multiple names in one cell. Without candidates,
the row remains `unattributed`. A unique ATen candidate shares an `operator_id`
with direct mappings of the same API. Rows with the same API and full kernel
name are merged using their unrounded runtime durations before calculating
the time share; a direct mapping takes precedence over an eager label.
Reference-only rows keep `eager-refer`. Ambiguous API candidates are not merged
with individual APIs. IDs are reassigned consecutively, with ATen APIs first.
The detail report retains reference provenance for merged rows. A reference
API may identify a parent wrapper rather than the exact internal computation.

In `kernel_time.csv`, only a `null` API name can be filled from one distinct
eager API candidate for the exact full kernel name. Multiple candidates or a
reference conflicting with the runtime API candidates remain `null`, and
existing non-null names are never overwritten. Matching API/kernel
rows are merged by summing runtime call counts and unrounded durations before
recalculating `percent`; `refer` remains `eager-refer` if any merged record used
a reference. Eager calls and durations never enter the runtime totals.

`kernel_shape_dtype.csv` also fills a `null` API name only when the eager API
candidate is unique. It retains the original graph mapping status, inputs,
variant rows, event counts, and durations. In particular, a filled name with a
status such as `missing_external_id` is still a reference, not a direct match.
All five reference fields are retained when `--eager-reference` is supplied:
`eager_operator_name`, `eager_input_shapes`, `eager_input_dtypes`,
`eager_candidates`, and `mapping_source`.
Extra `eager_input_shapes` and `eager_input_dtypes` contain only unique reference
values for missing fields; multiple values remain `null`.
`eager_candidates` keeps paired API/shape/dtype candidates so that unrelated
inputs are never combined. `mapping_source=eager-refer` indicates reference
metadata is available, even when multiple API candidates prevent name filling.
References do not increase the direct graph coverage reported in `summary.json`.
The inventory is still built from raw graph evidence, not the filled output
names. These display backfills do not further alter inventory IDs or direct
mapping statistics.
Reference inputs cannot be chained from API names filled by an earlier run.
Without `--eager-reference`, no API names are filled, all `refer` values are
`null`, and the five detail reference fields are omitted.

Add an exceptional specialization to `KERNEL_GROUPING_RULES` with a named
`match` and `key` lambda. `kernel_signature()` exposes the callable symbol and
top-level template arguments without splitting nested C++ types. Keep semantic
arguments in the key, add a positive grouping test and a negative collision
test, and verify every conservation check in `summary.json`. Rules describe
kernel families, not individual models.

## Generate FlagOS coverage

After reparsing either inventory with eager references, rerun this command to
refresh coverage. Neither the extractor nor `profile.sh` updates coverage CSVs
automatically.

```bash
python3 tools/operator_profile/generate_flagos_coverage.py \
  --baseline /workspace/op_profile/qwen3.6_35b_a3b_native_graph_4096_256/results/operator_list.csv \
  --plugin /workspace/op_profile/qwen3.6_35b_a3b_plugin_graph_4096_256/results/operator_list.csv \
  --flaggems-oplist /workspace/op_profile/qwen3.6_35b_a3b_plugin_graph_4096_256/results/flaggems_enable_oplist.txt \
  --output /workspace/op_profile/qwen3.6_35b_a3b_flagos_coverage/operator_flagos_coverage.csv
```

The command also writes `operator_flagos_coverage_summary.csv` beside the
detailed `operator_flagos_coverage.csv` report. It contains `covered_operator_count`,
`undetermined_operator_count`, `total_operator_count`, and
`coverage_percent(%)` in one row. The percentage uses two decimal places.
Operators without enough evidence to make a coverage decision have empty
`flagos_covered` and `flagos_type` fields; they remain in the total operator
count.

The current policy counts every observed Triton operation in the numerator.
Other operations require auditable FlagGems evidence. An ATen API is covered
when a recorded FlagGems callable maps to that API through the installed
FlagGems ATen registration table. Without an enable-op list, ATen coverage is
undetermined rather than false. Kernel names are not used as secondary ATen
replacement evidence. Communication remains in the
denominator and requires FlagCX evidence to enter the numerator. Coverage is
based on operator kinds and is not weighted by calls or execution time. The
report writes Boolean `flagos_covered` values and a `flagos_type`
classification before the `kernel_name` column.

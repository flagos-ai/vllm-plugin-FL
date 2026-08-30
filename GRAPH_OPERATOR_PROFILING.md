# Graph-Mode Operator, Input Shape, and Dtype Collection

This document provides a reproducible procedure for collecting graph-mode
logical operator names, input shapes, input dtypes, and call counts without
modifying the vLLM source tree. The procedure has been validated end to end
with DeepSeek-V4-Flash and Qwen3.6-35B-A3B in the same `plugin-cz-24`
container.

The collection has two phases:

1. `actual_capture`: every worker profiles the real model execution used to
   build each CUDA Graph and writes a rank-specific trace.
2. `runtime`: after the server becomes ready, the upstream `/start_profile`
   and `/stop_profile` endpoints profile one fixed request on every rank.

The parser selects rank 0 from both phases and unions rows by
`(operator, input_shapes, input_dtypes)`. Collection is intentionally
all-rank; rank filtering happens only during result extraction.

Runtime profiling alone cannot expose all logical operators inside a stable
CUDA Graph because replay is normally represented as a graph launch. Capture
profiling supplies those graph-construction operators and their input metadata.
Runtime profiling supplies operations that occur only after server startup,
including sampling and request-time execution outside a captured graph.

## Validated environment

| Item | Value |
|---|---|
| Container | `plugin-cz-24` |
| Repository | `/vllm-workspace/vllm-plugin-FL` |
| Installation | vllm-plugin-FL editable install |
| vLLM package | `0.24.0+cu129` |
| vllm-plugin-FL | `0.3.0rc0+gf4319bd2a`, branch `main` |
| DeepSeek model | `/models/DeepSeek-V4-Flash`, TP=8 |
| Qwen model | `/models/Qwen3.6-35B-A3B`, TP=2 |
| Graph mode | vLLM default `FULL_AND_PIECEWISE` |
| Capture size | `1` |

The reproduction tools are under `tools/graph_operator_profile/`:

- `serve_deepseek_v4_flash.sh` starts DeepSeek in the default graph mode.
- `serve_qwen3_6_35b_a3b.sh` starts Qwen in the default graph mode.
- `profile_request.sh` performs the health check, warmup request, profiling
  request, profiler stop, and rank-0 extraction.
- `extract_operator_shapes.py` retains every trace, selects the requested rank,
  and writes phase-specific and union CSV files.
- The two `*_request.json` files fix the input so shape results are repeatable.

## Preflight checks

Log in and enter the container:

```bash
ssh sz
docker exec -it plugin-cz-24 bash
cd /vllm-workspace/vllm-plugin-FL
```

Check versions and model directories:

```bash
python3 - <<'PY'
import importlib.metadata
import vllm

print("vllm runtime:", vllm.__version__)
print("vllm package:", importlib.metadata.version("vllm"))
print("vllm-plugin-fl:", importlib.metadata.version("vllm-plugin-fl"))
PY
test -d /models/DeepSeek-V4-Flash
test -d /models/Qwen3.6-35B-A3B
```

Install the repository in editable mode if needed:

```bash
VLLM_VENDOR=cuda pip install --no-build-isolation -e \
  /vllm-workspace/vllm-plugin-FL
```

Check the supplied tools:

```bash
bash -n tools/graph_operator_profile/serve_deepseek_v4_flash.sh
bash -n tools/graph_operator_profile/serve_qwen3_6_35b_a3b.sh
bash -n tools/graph_operator_profile/profile_request.sh
python3 -m py_compile tools/graph_operator_profile/extract_operator_shapes.py
```

## Reproduce DeepSeek-V4-Flash

Run both cleanup commands separately. Either command alone can leave workers
behind:

```bash
pkill -9 -f vllm
pkill -9 -f VLLM
```

Start the server in terminal A:

```bash
cd /vllm-workspace/vllm-plugin-FL
bash tools/graph_operator_profile/serve_deepseek_v4_flash.sh
```

Open terminal B, enter the container, and run the fixed request after the
health endpoint becomes ready:

```bash
docker exec -it plugin-cz-24 bash
cd /vllm-workspace/vllm-plugin-FL
bash tools/graph_operator_profile/profile_request.sh \
  deepseek_v4_flash \
  tools/graph_operator_profile/deepseek_v4_flash_request.json
```

The run directory is:

```text
/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash/
```

## Reproduce Qwen3.6-35B-A3B

Stop the DeepSeek service with the same two separate commands:

```bash
pkill -9 -f vllm
pkill -9 -f VLLM
```

Start the server in terminal A:

```bash
cd /vllm-workspace/vllm-plugin-FL
bash tools/graph_operator_profile/serve_qwen3_6_35b_a3b.sh
```

Run the fixed request in terminal B:

```bash
docker exec -it plugin-cz-24 bash
cd /vllm-workspace/vllm-plugin-FL
bash tools/graph_operator_profile/profile_request.sh \
  qwen3_6_35b_a3b \
  tools/graph_operator_profile/qwen3_6_35b_a3b_request.json
```

The run directory is:

```text
/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b/
```

After validation, run both cleanup commands separately again:

```bash
pkill -9 -f vllm
pkill -9 -f VLLM
```

## Output files

Each model's `results/` directory contains:

| File | Meaning |
|---|---|
| `actual_capture_operator_shape_dtype.csv` | All selected-rank capture `cpu_op` rows, including rows without input metadata |
| `actual_capture_effective_operator_shape_dtype.csv` | Capture rows that contain both shape and dtype |
| `runtime_operator_shape_dtype.csv` | All selected-rank runtime `cpu_op` rows, including rows without input metadata |
| `runtime_effective_operator_shape_dtype.csv` | Runtime rows that contain both shape and dtype |
| `logical_operator_shape_dtype_union.csv` | Union of effective capture and runtime rows; use this as the primary logical-operator inventory |
| `*_operator_shape_dtype_by_rank.csv` | Audit output with an explicit rank column |
| `summary.json` | Trace counts, event counts, effective rows, unique operator names, and events without input metadata |

The union CSV fields are:

| Field | Meaning |
|---|---|
| `operator` | Logical operator or compiled logical-call name observed by PyTorch Profiler |
| `input_shapes` | Profiler `Input Dims`, in input argument order |
| `input_dtypes` | Profiler `Input type`, aligned with the input argument positions |
| `actual_capture_call_count` | Observations across all selected-rank capture traces; this is not a per-request count |
| `runtime_call_count` | Observations in the selected-rank fixed runtime request |
| `observed_in` | `actual_capture`, `runtime`, or both |

One operator can produce multiple rows when its shape or dtype changes. Do not
deduplicate by operator name alone.

## Collection and rank-selection behavior

When `VLLM_FL_GRAPH_CAPTURE_PROFILE_DIR` is unset, the original `_dummy_run`
path executes directly and no capture profiler is created. When it is set, an
explicit profiling branch wraps the same `_dummy_run` call. Every distributed
rank writes its own trace with `rank_<N>` in the filename.

The upstream `/start_profile` behavior is unchanged and also writes one runtime
trace per rank. `extract_operator_shapes.py --rank 0` applies rank selection
only while generating CSV files. Raw traces from nonzero ranks remain on disk
for later audit or per-rank comparison.

## Coverage definition and limitations

For a fixed model, TP size, capture size, request, and selected rank, this
procedure retains every emitted capture/runtime trace and every profiler
`cpu_op` event. It does not incorrectly deduplicate files that share a worker
name or token count.

The following limits still apply:

- The primary CSV represents rank 0. If TP ranks have asymmetric execution,
  inspect additional ranks by running the parser with another `--rank` value.
- A different prompt length, batch size, concurrency level, prefill/decode
  length, structured output, sampling configuration, or capture size can
  activate additional shapes or branches and must be profiled separately.
- An Inductor or Triton fused kernel may represent several source operators.
  Runtime kernel events do not normally carry a one-to-one high-level input
  shape. This procedure produces a logical-operator inventory, not a kernel
  argument inventory.
- Some profiler `cpu_op` annotations do not provide `Input Dims` or
  `Input type`. They remain in the non-`effective` CSV files and are listed in
  `summary.json`; the missing metadata cannot be reconstructed from the event.
- Capture call counts describe graph construction and may include PIECEWISE,
  FULL, warmup, or multiple graph variants. They are not end-to-end request
  call counts.

## Implementation notes

The launch scripts intentionally omit `cudagraph_mode` from
`--compilation-config`. The resolved vLLM 0.24 configuration must contain:

```text
cudagraph_mode=<CUDAGraphMode.FULL_AND_PIECEWISE: ...>
```

Native vLLM 0.24 automatically enables breakable CUDA Graph for DeepSeek-V4
unless the user explicitly sets `VLLM_USE_BREAKABLE_CUDAGRAPH`. In this tested
software combination, the breakable FULL replay path causes a CUDA illegal
memory access. The launch scripts therefore set:

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

This disables the breakable wrapper but does not select eager mode. The
resolved mode remains the default `FULL_AND_PIECEWISE` using the standard
CUDA Graph path.

Capture profiling is opt-in:

```bash
export VLLM_FL_GRAPH_CAPTURE_PROFILE_DIR=.../capture_traces
```

The implementation does not use `sitecustomize` or `logical_reference`.
`/start_profile` and `/stop_profile` remain the upstream vLLM endpoints.

## Result validation

After both runs complete:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/vllm-workspace/graph_operator_profile_runs")
for model in ("deepseek_v4_flash", "qwen3_6_35b_a3b"):
    summary = root / model / "results" / "summary.json"
    data = json.loads(summary.read_text())
    print(model, json.dumps(data, indent=2))
    assert data["actual_capture"]["trace_files"] > 0
    assert data["runtime"]["trace_files"] == 1
    assert data["actual_capture"]["cpu_operator_events"] > 0
    assert data["runtime"]["cpu_operator_events"] > 0
    assert data["logical_union_shape_dtype_rows"] > 0
PY

grep -F 'CUDAGraphMode.FULL_AND_PIECEWISE' \
  /vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash/serve.log
grep -F 'CUDAGraphMode.FULL_AND_PIECEWISE' \
  /vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b/serve.log
```

Do not hard-code the number of capture variants in automation. vLLM can emit
multiple PIECEWISE and FULL traces. The parser retains every actual trace and
selects the requested rank by filename.

## Validated results

The table below records the rank-0 extraction from the fixed requests. These
values prove that the procedure ran successfully; they are not universal
expectations for other request configurations.

| Model | Graph mode | Rank-0 capture traces | Capture `cpu_op` | Capture effective rows/names | Rank-0 runtime traces | Runtime `cpu_op` | Runtime effective rows/names | Union rows/names |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | `FULL_AND_PIECEWISE` | 4 | 52,395 | 362 / 86 | 1 | 2,704 | 173 / 54 | 405 / 91 |
| Qwen3.6-35B-A3B | `FULL_AND_PIECEWISE` | 4 | 22,530 | 174 / 56 | 1 | 1,473 | 125 / 44 | 245 / 75 |

The all-rank collection was revalidated after enabling capture traces on every
worker. DeepSeek produced 32 capture traces and 8 runtime traces; Qwen produced
8 capture traces and 2 runtime traces. Rank-0 extraction still selected four
capture traces and one runtime trace for each model. Both fixed requests
returned four completion tokens, and neither log contained a CUDA error,
illegal memory access, traceback, or engine initialization failure.

If extraction is empty, inspect `serve.log` for graph-capture profiler messages,
confirm that the resolved mode is `FULL_AND_PIECEWISE`, verify that
`/stop_profile` completed, and check for rank-specific
`*.pt.trace.json.gz` files under the profile directory.

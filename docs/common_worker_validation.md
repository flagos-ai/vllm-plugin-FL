# Common worker review validation

The sections below record the earlier snapshot. The September 20 follow-up
adds enforced installed-worker identity, cross-batch request reuse checks and
combined FlagGems/MM/metadata execution. See `common_worker_review2.md` for the
current results; historical numbers here are not new-code acceptance.

The reviewed starting point was `4e6c15d`, based on `940480f`. Runtime fixes were
tested at `fd21d86` on an H100 with vLLM 0.24.0+cu129 and PyTorch 2.11.0+cu129.
The Python wheel was built with `VLLM_VENDOR` empty, installed into a separate
directory, and imported from there while running a test-only directory. The four
changed runtime files matched the source SHA256 values recorded in
[common_worker_validation.json](common_worker_validation.json). Later changes
only add test guards, a reproduction script and documentation.

## Component and dispatcher checks

All 83 tests passed against the isolated installed wheel, including the final
platform-guard revision, across these files:

```bash
python -m pytest -q \
  tests/unit_tests/patches/test_flaggems_mm_shape_aware.py \
  tests/unit_tests/worker/test_common_metadata_lifecycle.py \
  tests/unit_tests/worker/test_model_runner.py \
  tests/unit_tests/worker/test_worker.py \
  tests/functional_tests/worker/test_common_attention_metadata_graph.py \
  tests/functional_tests/patches/test_shape_aware_mm_dispatch.py
```

Coverage includes real dispatcher ownership changes in isolated processes;
actual FlagGems MM registration, native/FlagGems routing and graph replay;
stock/eager/graph metadata equality; prefix positions, request reorder and
padding; stale receipts and replaced buffers; InputBatch invalidation; and
FULL/PIECEWISE capture through `_warmup_and_capture` and `_dummy_run` up to the
model boundary. The real H100 provider was loaded as `hopper.ops.mm`, which is
why provider validation checks its source beneath the loaded FlagGems package
instead of requiring a `flag_gems` module-name prefix.

## Full worker reproduction

[common_metadata_probe.py](../tests/e2e_tests/inference/common_metadata_probe.py)
creates a seeded, two-layer Llama checkpoint and tokenizer locally. It runs
stock/eager/graph in separate processes, with four server slots, eight requests
followed by the same eight in reversed order, prefix caching and asynchronous
scheduling enabled. Each request generates 12 greedy tokens. `USE_FLAGGEMS=0`
isolates metadata policy from ATen MM registration; the actual FL worker still
runs, using FlashAttention. MM with real FlagGems is covered separately above.

With the plugin wheel already installed, use fresh output directories:

```bash
python tests/e2e_tests/inference/common_metadata_probe.py \
  --output-dir /tmp/common-metadata-decode \
  --installed-root /opt/installed-common \
  --expected-runtime-manifest /artifacts/common_runtime_manifest.json \
  --cudagraph-mode FULL_DECODE_ONLY
python tests/e2e_tests/inference/common_metadata_probe.py \
  --output-dir /tmp/common-metadata-piecewise \
  --installed-root /opt/installed-common \
  --expected-runtime-manifest /artifacts/common_runtime_manifest.json \
  --cudagraph-mode PIECEWISE
```

The manifest is a relative package path to SHA256 mapping generated from the
wheel at build time. The script copies its probe to an isolated directory,
replaces inherited PYTHONPATH with explicit paths, sets the child working
directory, and checks module paths/hashes returned by actual workers. Optional
`--dependency-path` entries locate pinned dependencies such as FlagGems;
`--combination` enables the full worker FlagGems policy plus shape-aware MM.

The script stores the generated checkpoint, per-process logs, exact token IDs,
top-5 logprobs, and worker producer counters. Token IDs and top-5 identities must
match exactly. FULL_DECODE_ONLY requires exact logprob equality. PIECEWISE uses
an absolute logprob tolerance of `1e-5` for separate Inductor compilations;
metadata tensor tests remain exact. Final runs of the committed script passed:

| Model graph mode | Requests per policy | Max top-5 logprob delta | Runtime metadata replays |
|---|---:|---:|---:|
| FULL_DECODE_ONLY | 16 | 0 | 43 |
| PIECEWISE | 16 | 4.291534423828125e-6 | 44 |

Both rows had exact token IDs and top-5 identities across all three policies.
Capture counters were 5 and 2 respectively, including profiling/recapture.
This is a small integration regression, not an accuracy evaluation or serving
performance benchmark.

## Known limitation: mixed FULL model graphs

The same probe with `--cudagraph-mode FULL` exposed output instability after
request reorder/reuse. Its first batch matched across all three metadata modes,
but eager differed in the second batch. More significantly, even stock changed
the greedy continuation of the same prompt between the two batches. A control
using both worker files from main `940480f` reproduced that within-run change.
Making CUDA launches blocking did not remove it. These observations do not
identify the root cause or prove that every difference shares one cause.

With model graphs disabled, stock/eager produced identical results across all
16 requests and repeated prompts. FULL_DECODE_ONLY also matched that control.
The separate native-vLLM diagnostic used a different runner/backend and was
not treated as an exact numerical acceptance baseline. Mixed FULL model-graph
integration remains unresolved; component-level FULL capture tests must not be
reported as full-model correctness acceptance. The default metadata policy
remains stock, and broader platform/scheduler/model validation remains open.

This branch retains the original per-group metadata kernel. PR442's fused
producer and the local PR455+PR442 Qwen serving experiment are separate; neither
their throughput nor model-quality results validate these common-worker changes.

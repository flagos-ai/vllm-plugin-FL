# Qwen4 all-on checkpoint — 2026-09-07

The measured plugin checkpoint is commit
`efc05b365ed12675098d95ac65f80c80457f16f6`, based on
`fadbba0ea59bbaa46c77b06d465321ab88b44643`. Its Qwen adaptation,
self-developed QSA, metadata graph and ATen plan-cache prerequisites were
subsequently merged into **day0-qwen4**, preserving that branch's newer
framework, quantization and other-model support.

### day0 integration boundaries

- The all-on environment explicitly enables the packed arena and 2D slot
  producer. Without that switch, day0 retains its existing common-attention
  metadata producer, including padded-row cleanup.
- The packed path keeps the measured builder-side padded-row cleanup; it
  must not inherit the older day0 producer's claim that rows are already
  cleared. Both graph caches are cleared before rebinding input buffers.
- A plan-cache request overrides Qwen's older generic-native-ATen policy,
  so generic FlagGems dispatch is actually exercised. Explicit plugin-off
  still overrides the direct FlagGems cache switch.
- Newer MXFP8/W8A8, shape-aware MM, breakable graph and other-model hooks
  are preserved. Existing vendored sources are retained.
- Six CPU-only integration regression tests cover routing, cache-policy
  precedence, require gates, padded cleanup, attention workspace warmup and
  the existing vLLM 0.24 multimodal-pruning compatibility setup.
  The kernel math is not retuned by this merge.

**The performance numbers and `source_manifest.json` below describe the
measured efc05b3 snapshot, not a new E2E run of the merged day0 branch.**
Re-run the harness on the merged branch before claiming an identical E2E
score. The source manifest is deliberately retained as historical evidence,
not regenerated to make a different source tree appear previously measured.

Run the merge-boundary checks without importing vLLM or touching a GPU:

```bash
python -m unittest discover -s tests/unit_tests/worker \
  -p test_allopt_day0_integration.py -v
```

## All-on changes

- QSA split8 with FP32 partial/merge, preserved BF16 gate rounding, and
  per-layer/per-bucket fixed-address workspace warmed before graph capture.
- Persistent packed block-table and slot-mapping arena, one active-prefix
  H2D transfer, and a 2D request/group producer.
- GDN two-16-row register layout while retaining FP32 beta/state and the
  original wrapper launch contract.
- Delayed HC combine/grouped-norm with packed injection strides and explicit
  materialization at PLE/DeepStack/PP/final boundaries.
- Real ATen plan-cache activation with modern `enable()` ABI validation;
  fail closed when the requested cache or optimized runtime path is absent.

MoE/dense ownership and model weights were unchanged during this experiment.
There were no single-component E2E ablations.

## Results

TP8, 8x H100 80GB; c64; random input/output 1024/1024; seed 12345;
temperature 0; ignore EOS; request rate infinity; prefix caching off.
Each group used 64 full-shape warmup requests followed by three 128-request
rounds. Both groups used the same live server and identical flags/source.

| Group / round | Output tok/s | Mean TTFT ms | Mean TPOT ms |
|---|---:|---:|---:|
| Initial 1 | 3153.26 | 1951.17 | 18.3968 |
| Initial 2 | 2108.33 | 5711.06 | 24.7861 |
| Initial 3 | 2984.59 | 1477.22 | 20.0069 |
| Warmed repeat 1 | 3400.68 | 1438.73 | 17.4172 |
| Warmed repeat 2 | 3392.14 | 1439.62 | 17.4645 |
| Warmed repeat 3 | 3334.52 | 1359.01 | 17.8671 |

All rounds completed 128 requests, zero failures, and exactly 131072 input
and output tokens. The initial group had 38.01% throughput range/mean drift,
so the entire unchanged protocol was repeated after that exceeded 5%.
FlagGems warmup transitions are retained, not hidden or labeled as steady
kernel regression. The repeat group averaged **3375.78 tok/s**, with
**1.96% drift**, mean TTFT **1412.45 ms**, and mean TPOT **17.5830 ms**.
All six round throughputs average 3062.25 tok/s; these are arithmetic means
of round metrics, not pooled throughput over the whole server lifecycle.

Historical steady references were 3197.07 tok/s for selfdev+cache and
3572.19 tok/s for the official image: the warmed combination is respectively
**+5.59%** and **-5.50%**. These are cross-run references, not paired causal
estimates. No individual optimization contribution can be inferred.

See [initial results](initial_summary.json), [warmed results](steady_summary.json),
and [all-round analysis](analysis.json). Startup capture alone did not fully
warm later runtime cache keys. Initial p99 ITL remained near 18 ms but included
sparse multi-second gaps; a new trace was not collected to attribute each gap
to a particular compilation or tuning event.

## Validation and limits

- 82 component tests: cache bridge 10, QSA 32, HC 20, runner 5, GDN 15.
- Changing-input CUDA Graph replay and recurrent-state precision checks
  passed. This is not a full-model accuracy evaluation or production soak.
- [Worker cache gate](plan_cache_gate.json): 8/8 ranks enabled/installed,
  each with 19511 hits and 1265 misses after model warmup. Before shutdown,
  each had 162546 hits / 2222 misses (98.65% cumulative hit rate); that
  lifecycle total includes API smoke, warmups and both test groups.
- Actual graph mode was `FULL_DECODE_ONLY` on every rank, as required by GDN.
- Whole-lifecycle 1 Hz telemetry peaked at 51 C. Every sample with GPU
  utilization >=50% had SM frequency 1830 MHz; thermal/HW slowdown and
  software power-cap samples were all inactive. Sampling cannot exclude
  events between samples.
- MTP/speculative decoding was disabled. Its pre-existing vLLM import ABI
  mismatch and the old runner's ROCm-only shutdown lint issue are outside
  this NVIDIA checkpoint's validation scope.

## Reproduction

The tested image used Torch 2.11.0+cu129 and vLLM 0.24.0; its image ID was
`sha256:b7faa60041af787c35de236685120fb072697a6547fa4ef5877095f7f1519d87`.
Use an already validated H100 container and mount:

- this checkout read-only at `/opt/vllm-plugin-FL`;
- the matching FlagGems tree read-only at `/opt/FlagGems`;
- these harness files at `/opt/qwen4-allopt-harness`;
- the checkpoint at `/models/Qwen3.8-Flash-Next`;
- a fresh writable results directory at `/artifact`.

The entry/cache-module FlagGems hashes are checked in `all_on_env.sh`.
The expected cache module hash is
`d9afa71ffb3aada02b82468569d9e02c07b5981a83066fab56f82bd2f3390921`;
the older install-only tree is not interchangeable. Checkpoint config hash:
`889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`.
Weights and external dependency trees are not distributed in this commit.

Inside that container, set the fixed server limits and launch:

```bash
export MAX_MODEL_LEN=2048 MAX_NUM_SEQS=64 MAX_NUM_BATCHED_TOKENS=16384
export GPU_MEMORY_UTILIZATION=0.85
bash /opt/qwen4-allopt-harness/launch_server.sh
```

From a second shell in the same container:

```bash
bash /opt/qwen4-allopt-harness/run_e2e.sh
```

The runner checks HTTP health, all eight real worker cache records, API
smoke, then warmup and three rounds. It does not start Nsight Systems.
If repeating the full group for warmup convergence, use a fresh result path
and retain the existing server log:

```bash
ARTIFACT_ROOT=/artifact/repeat2 SERVER_LOG=/artifact/server.log \
  bash /opt/qwen4-allopt-harness/run_e2e.sh
```

Full raw logs, detailed per-request JSON, telemetry and component validation
remain in the local delivery artifacts rather than this source commit. The
original full-source archive SHA256 is
`cd6742f046e417fb55a4a887750a3dcc9f8a5303fc6785f331ebc2c186c23946`;
the full result archive SHA256 is
`92274af02e8159aa4187d34fe8798fdfac5cd53791b9b9282959ba235da3ea49`.
The source hash manifest records the tested files; the internal HC handoff
note is omitted from publication, and this benchmark bundle is additive.

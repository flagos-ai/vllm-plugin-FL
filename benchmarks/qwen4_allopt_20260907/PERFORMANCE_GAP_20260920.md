# Qwen throughput gap investigation (2026-09-20)

The subsequent [GPU-history and QSA experiment](PERFORMANCE_ASYNC_20260920.md)
restores short-context throughput to 3376.54 tokens/s. This document preserves
the earlier sorting diagnosis and its 1803.55 tokens/s result.

The measured QSA sorting bottleneck is addressed by keeping `sort` and
`sort_stable` native in the existing NVIDIA-only model dispatch policy.
Stable score ties, canonical logical-index order, the corrected request map
and the synchronous PLE-history guard are preserved. Other vendors retain
their existing dispatch. The local PR442 fusion was not pushed.

A same-node, same-source, unprofiled dispatch ablation improved mean output
throughput from **1514.47 to 1803.55 tokens/s
(+19.09%)**, with mean TPOT **34.1978 ms** versus
39.8603 ms. Historical 3375.78 remains unrecovered.

## Controlled serving comparison

Both arms used the same eight H100s, image, checkpoint, immutable local fusion
`55f106c` (PR442 `4e3540a`), runtime files, TP8, 64 slots, 2048 context, 16384
batch tokens, 0.85 memory fraction, disabled prefix caching and custom AR,
MoE Triton, packed/common metadata, and verified plan cache on all eight ranks.
The sole recorded dispatch change appended `sort,sort_stable` to
`VLLM_FL_FLAGOS_BLACKLIST`. There was no profiler in either throughput run.
The runs were separate service launches, not a randomized crossover.

The benchmark used random 1024/1024 requests, concurrency 64, seed 12345,
temperature 0 and ignored EOS: 64 warmups and three 128-request formal rounds.
Initial throughput was 1388.99 / 1802.97 / 1808.59 tokens/s. The measured
25.17% range/mean drift triggered the
predeclared >5% full-repeat rule; all initial results are retained.

| Repeated round | Output tokens/s | Mean TPOT (ms) |
|---|---:|---:|
| 1 | 1806.91 | 34.1288 |
| 2 | 1803.13 | 34.2118 |
| 3 | 1800.61 | 34.2527 |
| Mean | **1803.55** | **34.1978** |

Repeated drift was 0.35%. Every formal round completed 128 requests,
zero failures and 131072 output tokens. A subsequent 384-request marker check
at client concurrency 64/128 ended entirely with `stop`, without marker
violations. These are isolation checks, not full model-quality acceptance.
Two-second sampling recorded 30–49 degrees C;
no sampled thermal-slowdown or power-brake flag was active. Sub-sampling
transients are not excluded. The owned service was stopped and GPUs released.

The benchmark's exact-profile identity gate rejected the first attempt with
a changed blacklist before server startup. A copied diagnostic harness then
changed only that expected blacklist; source, model, image and eight-rank
plan-cache checks remained in force. No runtime source was modified for the
ablation. The final model policy makes the same two exclusions automatic on
NVIDIA; it was validated separately with the tests below.

## What the trace established

A warmed, bounded Nsight Systems 2026.4.1 capture covered steady decode at
Running64/Waiting0. Every rank replayed 49 model graphs and 49 metadata graphs;
98.24% of aggregate kernel time was inside graphs. The separate metadata
launches are excluded from the model-step denominator. Mean rank launch skew
was 0.741 ms. This rules out graph fallback or missing common-metadata replay
as the explanation in the captured window.

Rank 0 had a 41.91 ms model-to-model interval, with 22.09 ms of union kernel
execution and 19.81 ms without a kernel running. The latter is a kernel-activity
measure, not proof that every copy engine was idle. The serial chain averaged:

| Consecutive interval | Profiled ms |
|---|---:|
| Model GPU start to model GPU end | 24.12 |
| Model GPU end to output Event completion | 0.39 |
| Output Event completion to next memcpy API call | 10.68 |
| First next memcpy call to next model GPU start | 6.71 |

The Event's host wait overlaps GPU execution and must not be added to these
intervals. Profiling overhead is present; these timings are diagnostic, not
promised unprofiled speedups.

1. **Sorting is a measured GPU hotspot.** Each QSA layer runs a stable FP32
   score sort followed by an int64 logical-index sort. FlagGems radix sorting
   makes 8+16 sweep passes. Across 12 layers, the selection between score
   generation and index expansion contains 816 kernels and consumes **5.44 ms
   of GPU kernel time per rank/model step**. Sweep, histogram and broadcast
   account for 4.91 ms of that total. The ordering is required for correctness;
   the change selects a faster implementation of the same ordering.
2. **Scheduler mode changed.** The exact 3375.78 server log enabled async
   scheduling; the current run disables it through the `d79cd85` guard because
   the selected runner builds PLE n-gram context from CPU token history.
   The serial trace and large inter-step gap make a correct GPU-history async
   path the next priority. There is no correctness-preserving matched async
   experiment yet, so the entire gap cannot be assigned to this flag.
3. **The recent Event default is not a standalone cause.** The exact 3375.78
   run could not import the native completion extension and already fell back
   to accelerator Event synchronization.

An older September 7 contextual trace settled around 3021.36 tokens/s, not
3375.78. Its model graph had 2712 nodes versus the current 3426. The current
QSA-named kernel family takes 1.30 ms versus 1.53 ms there; the expensive sort
kernels have generic names and are accounted separately above. MoE kernel
time is higher (7.23 versus 5.11 ms per rank/step), while its source and launch
geometry match. Expert routing/active work was not collected, so the MoE
increase remains unassigned. Different source revisions and capture conditions
prevent treating these historical differences as a controlled causal estimate.

## Correctness and reproduction boundaries

The production selection function was extracted unchanged for CUDA-Graph
operator checks. At 64×1600 with nonnegative decode scores, full FlagGems
selection measured 0.481503 ms/call versus
0.068228 ms/call with only sorting native.
The native-sort arm exactly matched native stable-sort indices on all tested
shapes/distributions, including ties, padding, noncontiguous input and the
selection budget boundary. These isolated timings are not serving throughput.

An extended synthetic signed-zero test found a FlagGems cutoff tie difference
between +0 and -0 at 2048 columns. Canonicalizing zero or selecting native sort
restored the reference indices. The live score producer sums nonnegative
ReLU dot products; no signed-zero incident in serving was established.

Final policy and QSA checks: **44 passed in 15.25s**. These include actual
CUDA graph replay across the selection budget boundary and equality of sparse
attention output over repeated replay. The new policy is scoped to NVIDIA and
rejects a conflicting explicit sort whitelist, consistently with the existing
native-primitive policy. No new device kernel is introduced.

The next larger change requires GPU-resident PLE token history that survives
request reorder, queueing, slot reuse, prefix/chunk boundaries and graph replay.
Only after exact-history and model checks should the async guard be relaxed.
Do not revert the request-map or stable-selection fixes to recover an old score.

The long-reasoning issue remains open, and the 2K diagnostic is not the 100K
release profile. Shared-layer PR544 is still validated separately.

Machine-readable evidence and immutable report/SQLite hashes:
[performance_gap_validation.json](performance_gap_validation.json).

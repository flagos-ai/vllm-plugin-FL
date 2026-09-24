# PR 455 review follow-up (2026-09-19)

Current implementation and 1K/1K + 4K/1K acceptance are recorded in
[the September 20 integration review](REVIEW2_20260920.md). The optional native
completion subsystem has moved to draft [PR547](https://github.com/flagos-ai/vllm-plugin-FL/pull/547).
PR455 now uses ordinary accelerator Event completion. The sections below are
historical records, including the former opt-in flag and native fault-test scope.
The earlier scheduling/QSA experiment reached 3376.54 output tokens/s with
17.66 ms TPOT; see [its original evidence](PERFORMANCE_ASYNC_20260920.md).

This iteration addresses the four concrete findings against `d79cd85`.
[review_followup_validation.json](review_followup_validation.json) binds the
checks to runtime source hashes. The historical `efc05b3` serving numbers in
this directory remain historical, not results for this iteration.

## Changes

- **Async output completion:** ordinary accelerator Event synchronization is
  again the default. `VLLM_FL_ASYNC_OUTPUT_NATIVE_COMPLETION=1` explicitly opts
  into the native callback path. Its nonblocking eventfds are polled at most
  every 100 ms with a 30-second deadline; the recorded copy event is queried
  between polls to expose asynchronous errors. Errors and cancellation propagate.
  Worker shutdown cancels pending waits before accelerator cleanup. A slot whose
  notification was not consumed is never reused or closed underneath a late
  callback. The process stops using the pool after such a failure; at most 64
  descriptors are retained until process exit. This optional pool has a single
  worker-process lifetime. Real CUDA-context fault injection remains untested;
  CPU fault tests use real Linux eventfds and simulated accelerator events.
- **PLE weight completeness:** shard coverage is accumulated over the complete
  model-loading transaction. This matters because vLLM's AutoWeightsLoader can
  call the same module several times for nonadjacent checkpoint fragments.
  Partial calls do not certify the parameter; the outermost load rejects missing
  locally required rows. Duplicate shards, repeated full weights, and mixed
  full/sharded representations are rejected. Nonintersecting TP shards remain
  legal and shape checked. A new model reload starts a fresh transaction.
- **PLE hashing:** per-token windows gather from the flattened input and the
  corresponding request's small history. The request-by-capacity padded grid
  and its broadcast hash intermediates are removed. Tests check exact hash
  equality, EOS boundaries, empty request rows, chunking, request reorder,
  graph padding and graph replay with changed boundaries. The old test embedding
  could produce a zero-width output; it now returns real hash data and asserts
  nonempty output.
- **QSA identity:** the provenance now describes the actual local QSA
  composition. Vendored QSA files are retained reference sources, not the active
  implementation. `QSAIndexer.runtime_status()` uses the same compression
  selector as execution and reports callable/source hashes. The old pre-indexer
  status API reports that the vendor pre-indexer is disabled. Selection reports
  are explicitly distinguished from evidence of device kernel launches.
- **Public platform tests:** the two MoE platform tests now replace the live
  `vllm.platforms.current_platform`, which the implementation resolves at call
  time, instead of a removed module-level alias. No production routing behavior
  is changed by this test correction.

## Validation boundaries

128 targeted tests passed on H100 with vLLM 0.24.0 and PyTorch 2.11.0+cu129,
including the real AutoWeightsLoader fragmentation regression and PLE CUDA
Graph replay. Another 15 platform/MoE dispatch tests passed. A Python wheel
(`VLLM_VENDOR` empty) was built, installed into a separate directory, and its
changed runtime files compared byte-for-byte with the tested source. Its 37
loader/identity/completion tests passed. The native extension was not rebuilt;
the default Event path works in an empty-build runtime. The added eventfd fault
tests do not inject a real CUDA fault.

An isolated eager CUDA hash benchmark used 64 requests, one token per request,
ngram size 3 and eight heads per group, with an echo embedding (no table lookup).
The old class was extracted from `d79cd85`; old/new outputs were exactly equal.
At capacity 32768, extra peak allocation fell from 504,156,160 to 312,320 bytes;
20-call timing was 0.864 versus 0.506 ms/call. At capacities 1024 and 16384 the
new path was 0.518 and 0.505 ms/call. The remaining capacity-dependent allocation
is the one-dimensional position buffer. These are isolated operator results,
not full-model serving throughput or proof of model accuracy.

The complete model loaded all 131 checkpoint files. Both the plugin and native
reference passed 96 marker-isolation requests (two rounds each at client
concurrency 16 and 32). The plugin also passed 64 requests with mixed input and
output lengths, including queuing and slot reuse. All these requests ended with
`stop` and no marker violations. These checks address the tested isolation
patterns; they are not an exhaustive proof of absence of cross-talk.

Eight previously truncated GPQA cases were replayed once per arm at
temperature 1.0 and a 32768-token output budget. The request bodies and input
token sequences matched exactly. Against the existing labels, the plugin had
5 correct, 1 wrong and 2 truncated responses; the native reference had 6 correct,
1 wrong and 1 truncated response. Completed-answer accuracy was 5/6 versus 6/7;
the all-question counts were 5/8 versus 6/8. One case truncated in both arms;
another truncated only in the plugin. Manual inspection still found repeated
chemical-structure reasoning. Both arms selected the same option for the case scored incorrect against the
existing label; its label was not changed.

This is a selected diagnostic sample and one stochastic run, not a replacement
for the historical full-64 evaluation. The reference uses vLLM 0.29.0, PyTorch
2.13.0 and Transformers 5.16.1, versus 0.24.0, 2.11.0 and 5.12.1 in the plugin
arm. Its image preserves official runtime files but omits unused cron entries
to accommodate the test host's image-import restrictions. Framework versions
and active kernels differ, so the comparison cannot isolate a floating-point
root cause. The long-reasoning issue remains open. No claim is made that all
CI, other accelerator vendors, or every supported scheduler mode passed.

## Serving throughput and local PR 442 integration

The standalone PR455 quality configuration retained 16 server slots, a
102400-token context limit and disabled plan cache. Two 256-request 1k/1k
rounds completed without failures at 610.947 / 611.708 output tokens/s and
24.938 / 24.906 ms mean TPOT. Client concurrency was 64; measured server
concurrency was only 16, with up to 48 waiting. These settings differ from
the historical 64-slot throughput experiment.

The earlier-host local-only merge includes PR442 at `ba2343b` and PR455 at `b7905af`.
It preserves packed-arena transfers while running PR442's common metadata
producer, including capture tests with and without packed storage. The fused
snapshot passed 150 targeted tests and 24 tests against an independently
installed Python wheel. It was not pushed or merged into either remote branch.

That snapshot reran the historical TP8/H100 workload: 64 server slots,
2048-token context, 16384 batch tokens, packed arena, verified plan-cache
activation on all eight ranks, 64 warmup requests and three 128-request 1k/1k
rounds. Initial throughput was 1019.77 / 1286.45 / 1296.31 tokens/s. Its 23.03%
range/mean drift triggered the same full-repeat rule used for the historical
experiment. The unchanged warmed repeat produced:

| Round | Output tokens/s | Mean TPOT (ms) |
|---|---:|---:|
| 1 | 1288.23 | 47.2344 |
| 2 | 1291.74 | 47.0958 |
| 3 | 1294.50 | 46.9925 |
| Mean | **1291.49** | **47.1075** |

All formal requests succeeded; warmed throughput drift was 0.49%. A separate
384-request marker check at client concurrency 64/128 ended entirely with
`stop` and no marker violations; telemetry observed Running64/Waiting64.

The owner subsequently reported thermal stability problems on this host;
its figures are not a clean code-performance baseline. The second-node rerun
below supersedes this host for the current throughput observation.

Historical 3375.78 tokens/s has **not** been recovered. Historical QSA and
indexer source hashes match the pre-request-map-fix implementation, so that
number is not a correctness-equivalent acceptance baseline. This observation
does not establish the cause of the remaining throughput gap. Changing both
configuration and code also prevents attributing the standalone-to-fused
improvement solely to PR442.

Six identical next-token requests using early fixed prefixes agreed on top1,
but shared top20 log probabilities differed (maximum absolute difference
about 2.3125). This diagnostic does not establish full-vocabulary or
long-context numerical equivalence.

The separately reviewed `day0-common` MM registration lifecycle and metadata
ownership changes are delivered and validated separately; they are not part
of the fused Qwen benchmark above. Cross-vendor and every-scheduler validation
remain outside these NVIDIA results. No new device kernel was added in the
PR455 review fixes; existing local QSA kernels and PLE composition retain the
FlagGems/FlagTree follow-up described in the provenance document.

## Rerun on a second H100 node (2026-09-20)

The owner reports thermal stability problems on the earlier benchmark host.
Those measurements remain diagnostic and cannot establish a code regression;
the old discrete telemetry cannot quantify the thermal contribution.

PR442 was refreshed to `4e3540a`. Its changes since `ba2343b` are tests and
documentation only. The new local fusion `55f106c` has byte-identical runtime
files to the previous `1d334c0` fusion and was not pushed or merged remotely.
Nine common-metadata GPU tests passed on the new node. The image, driver,
FlagGems source, model metadata/shard sizes and workload were held constant.
All eight ranks passed the plan-cache gate.

Initial three-round throughput was 1195.41 / 1512.59 / 1513.39 tokens/s. Range/mean drift
of 22.60% triggered the predeclared full-repeat rule on the same
service. All initial results are retained. The repeated group produced:

| Round | Output tokens/s | Mean TPOT (ms) |
|---|---:|---:|
| 1 | 1526.85 | 39.5651 |
| 2 | 1532.58 | 39.4189 |
| 3 | 1483.99 | 40.5970 |
| Mean | **1514.47** | **39.8603** |

Every formal round completed all 128 requests without failures; repeated
throughput drift was 3.21%. This is 17.27% above the earlier host's
1291.49 tokens/s, but still below historical 3375.78. Node changes do not
isolate thermal causality, and the historical QSA implementation predates
the known request-map correctness fix. No root cause is established here.

Two-second hardware sampling throughout both groups, including warmup,
recorded 2832 GPU observations: 30–48 degrees C, SM clock
1830 MHz and memory clock 2619 MHz. No sampled hardware or
software thermal-slowdown flag was active. This does not exclude events
shorter than the sampling interval. Benchmark telemetry observed a Running
peak of 64. A subsequent 384-request marker check at client
concurrency 64/128 passed with every response ending in `stop`; its measured
Running/Waiting peaks were 64/64. The owned test service was stopped
and the eight GPUs released after collection.

This remains the short-context diagnostic profile, not a 100K release or
long-reasoning acceptance result. The separate common-layer changes in #544
are not included. Exact aggregate metrics and immutable artifact hashes are
in `review_followup_validation.json` under `second_h100_node_rerun`.

## Throughput gap investigation and native QSA sorting (2026-09-20)

The subsequent same-source sorting-dispatch experiment reached
**1803.55 output tokens/s**, **34.20 ms TPOT**,
a **19.09%** throughput improvement over 1514.47. The measured QSA sorting
hotspot is now handled by the NVIDIA model policy, while stable ordering and
all existing correctness guards are retained. The serial CPU/output/input
chain remains the next larger target; historical 3375.78 is still unrecovered.
See [the gap investigation](PERFORMANCE_GAP_20260920.md) for complete rounds,
trace attribution, policy tests, source boundaries and remaining limitations.

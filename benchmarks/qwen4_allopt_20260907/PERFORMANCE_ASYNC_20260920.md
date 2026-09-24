# Restore short-context serving throughput with GPU PLE history

The final same-profile, second-node experiment averages **3376.54 output
tokens/s**, **17.66 ms mean TPOT**, with **0.65%** three-round drift. This
recovers historical 3375.78 throughput while retaining the QSA request-map and
deterministic-selection fixes. It does not establish long-context performance
or full model-quality acceptance.

This follows the [sorting investigation](PERFORMANCE_GAP_20260920.md).
The previous 1803.55 tokens/s result did not close the gap. GPU-resident PLE
history first enables correct asynchronous scheduling; then an exact
within-budget QSA path removes scoring that cannot affect token selection.

## PLE state and scheduling

`ModelRunnerFL` retains consumed input tokens in device memory, keyed by a
request-owned slot rather than the movable InputBatch row. Input preparation
first resolves async sampled IDs on the device. PLE then gathers its left
context and records those resolved IDs on the same stream. Generated output
history on the CPU is not consulted during ordinary decode.

New admissions, resumes and forward prefix jumps seed from authoritative CPU
history; negative placeholders are rejected. Finish, replacement and resume
invalidate the previous slot. Reordering preserves request ownership, rollback
invalidates discarded suffixes, and graph dummy runs do not advance history.
Indexed writes use the existing native policy because the FlagGems
`index_copy_` implementation introduces a device-to-host bounds check.

The full token bank costs four bytes per request slot per context token: about
0.5 MiB per worker at 64 slots/2048 tokens, or 25 MiB at 64/102400. This preserves
recomputation from earlier positions. MTP, pipeline parallelism and DBO remain
subject to the existing PLE restrictions. The CPU-history scheduling guard is
retained for a runner that actually declares CPU history.

## Matched scheduling experiment

Both arms used the same immutable local PR455+PR442 source, image, model and
second 8-H100 node. PR442 was checked again at `4e3540a` and remains a local
benchmark dependency. It was not merged or pushed into the model PR.

The profile is TP8, 2048 context, 64 server slots, 16384 batch tokens, memory
utilization 0.85, prefix caching off, custom all-reduce off, Triton MoE, packed
block tables and verified plan-cache activation on all eight ranks. Requested
FULL resolves to FULL_DECODE_ONLY. Both use ordinary Event completion.
Workload: fixed-seed random 1024-token input/1024-token output, client
concurrency 64, temperature zero, ignored EOS, 64 warmups and three 128-request
formal rounds. This is a sequential comparison, not a randomized crossover.

| Scheduling | Three warmed rounds, output tokens/s | Mean tokens/s | Mean TPOT | Range/mean drift |
|---|---|---:|---:|---:|
| Synchronous | 1812.33 / 1829.19 / 1817.18 | 1819.57 | 33.89 ms | 0.93% |
| Asynchronous | 3102.38 / 3117.07 / 3097.95 | 3105.80 | 19.25 ms | 0.62% |

Throughput increases 70.69%, while mean TPOT falls 43.19%. Every formal round
completed 128 requests without failures. The asynchronous result is still
8.00% below historical 3375.78 before the within-budget optimization below.

All earlier groups are preserved. Initial groups exceeded the predeclared 5%
drift threshold and triggered complete repeats. The first async repeat still
had 9.46% drift; one additional complete group passed. Its outlier contained
1.2/3.2-second pauses around the request-wave transition, while middle decode
remained near 18.3 ms. No new cache file or sampled thermal/power event explained
the pause; its cause is unresolved. Individual rounds were not substituted.
Two-second hardware sampling recorded 30–50 C, SM 1830 MHz and memory 2619 MHz,
with no sampled thermal or power-limit flag. Shorter events are not excluded.
The raw PYTHONPATH differed only by repeated copies of the same roots; ordered
unique import roots and source hashes matched.

Each arm passed 384 marker-isolation requests at client concurrency 64/128.
Four serial 512-token continuations matched every token and full top-5
logprob. Mixed concurrency is not bitwise invariant: 17/32 complete outputs
matched across arms, versus 16–19/32 when repeating the unchanged synchronous
arm. That establishes an existing batch-dependent variation baseline, not a
proof of full model-quality equivalence. Serializing the mixed corpus is an
additional check in the subsequent candidate.

## Exact selection within a fixed context budget

When a worker's lifetime context limit is at most the 2048-token QSA budget,
all visible compressed blocks must be selected. Their scores cannot change
membership, and the required canonical ordering is simply ascending block
index. The candidate reuses the existing expansion kernel for complete blocks,
causal tails, invalid requests and padding, while omitting indexer projection,
compression and score sorting.

The gate uses the configured lifetime limit, not a short batch observed during
capture. A longer-context worker retains the complete scoring path, including
for its short batches. KV-transfer configurations also retain the side-cache
updates, because a consumer could have a larger context limit. The shared
compressed metadata `prepare()` still runs before selection; this preserves
the request-map correctness fix. No new device kernel is introduced.

The source suite passed 180 checks, covering GPU history, PLE hash semantics,
QSA references, request mapping, cache layout and graph replay. The new selector
checks compare exact indices and attention output across small/odd/padded rows,
changed lengths and request maps, tied/random scores, padding, and empty/large
prefill batches up to 16384 rows. An independently installed model-only Python
wheel passed 88 checks outside the source checkout. Runtime files were matched
to their respective model-only and local-fusion sources. The native extension
was not rebuilt. A final 28-case installed-wheel pass covers every configured
decode graph bucket (1/2/4/8/16/24/32/40/48/56/64), additional rows 3/33/128,
and the empty/large-prefill cases. The last source change only expands these
test parameters; the measured runtime files remain byte-identical.

The isolated selector benchmark compares the complete scored selector, a
literal Torch semantic reference, and the all-visible selector. It covers rows
1/8/64, physical score widths 512/1600/2048, logical context 1024/2048, native
and FlagGems dispatch (with native deterministic sorting). All 36 cases match
exactly. After 25 warmups, each of three measurements replays 100 graphs with
32 calls per graph; timing includes all selector kernels and reports device
and wall-clock times. At 64 rows/1600 columns under the serving selection
policy, the scored chain takes 77.39–77.76 microseconds and the candidate takes
2.87–2.92 microseconds. The equivalent Torch composition takes 5.58 microseconds.
These are selector timings, not an end-to-end speedup; the omitted indexer
projection/compression is covered only by the subsequent model experiment.

The full model then ran the same workload and gates, changing only the QSA
source implementation relative to the asynchronous scored-selector arm:

| Round | Output tokens/s | Mean TPOT (ms) |
|---|---:|---:|
| 1 | 3388.62 | 17.5881 |
| 2 | 3374.27 | 17.6671 |
| 3 | 3366.72 | 17.7116 |
| Mean | **3376.54** | **17.6556** |

Range/mean drift is **0.6486%**. This is **8.72%** above asynchronous scored
selection and **85.57%** above the matched synchronous arm. It is within
0.03% of historical 3375.78; that tiny difference is not a meaningful speedup.
All 128 requests in every formal round completed successfully. Initial rounds
2566.18/3399.90/2698.54 had 28.87% drift and triggered the complete repeat;
they remain in the evidence. The slow initial rounds contain multi-second
pauses around wave transitions despite middle-decode medians near 16.8–16.9 ms.
The warmed group has no token interval above one second. The intermittent
pause's root cause is still open; the repeat measures warmed steady state.

The final candidate passed another 384 marker-isolation requests, all ending
in `stop`, with zero violations. Four serial 512-token cases match both prior
arms exactly, including full top-5 logprobs. All 32 mixed-corpus prompts also
match the synchronous control exactly when serialized. Concurrent outputs
match 18/32 across either prior arm; repeats of the unchanged final candidate
match 14–19/32, so batch invariance is not claimed.

Hardware sampling recorded 31–51 C, SM 1830 MHz and memory 2619 MHz, without a
sampled thermal/power-limit flag. All owned test services were stopped after
validation; eight GPUs had zero compute processes and zero MiB allocated.
Model runtime commit: `536003429`. Local-only PR442 fusion: `680aa5bc4`.
Model-only wheel SHA256:
`11730158c789833d3fb1206c83f441157e7358e2f32cefa6d539f177fd20b814`.
Exact rounds, source hashes, test scope and artifact bindings are recorded in
[async_performance_validation.json](async_performance_validation.json).

## Scope

The historical implementation predates the QSA request-map fix; its number is
a throughput target, not a correctness-equivalent reference. These short-context
measurements do not resolve the selected long-reasoning GPQA failures or certify
a 100K serving profile. Shared-worker PR544 remains separate, including its
reported mixed-FULL validation issue. Other accelerator vendors and full CI
are not covered by these H100 observations.

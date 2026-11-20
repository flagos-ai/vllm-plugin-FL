# NVIDIA PR provider integrations

This experimental integration targets the NVIDIA stack pinned by `0.5.0-dev`
(base `4a4dc3464e0f82e096bc7dd47ed8d226c930ccd0`). It retains the precompiled
vLLM/FlagTree/FlagGems stack and installs the plugin checkout with `--no-deps`.
Follow [the NVIDIA installation instructions](nvidia.md) and run both runtime
and full stack checks before serving.

The verified `v0.32.0-cuda-ci` image has registry manifest digest
`sha256:02b65e788e4ce2352ff1f41099744c33e196707b36a2e68461f78bfbe799a884`
and image configuration ID
`sha256:c7b68ce18668b8f796b112a9632f32898f4aed802f078288077bbfe32309fd0e`.
The manifest and configuration digests identify different registry objects.

## Provider selection

All new providers are disabled by default. Set the same environment on every
worker, alongside `VLLM_PLUGINS=fl USE_FLAGGEMS=1 GEMS_VENDOR=nvidia`.

| Environment | Path | Preconditions |
| --- | --- | --- |
| `VLLM_FL_ATTN_RES_PROVIDER=1` | vendored [AttnRes PR737](https://github.com/flagos-ai/FlagGems-vllm/pull/737) | supported Kimi-K3 architecture, Hopper, BF16, eight residual block slots |
| `VLLM_FL_FUSED_ALLREDUCE_RMS_PROVIDER=peer` | vendored [AllReduceRMSNorm PR773](https://github.com/flagos-ai/FlagGems-vllm/pull/773) | same-host TP2/4/8, peer access, native compilation fusion enabled |
| `VLLM_FL_FUSED_ALLREDUCE_RMS_PROVIDER=mnnvl` | explicit multicast backend of PR773 | TP2/4/8, multicast capability on every rank, native compilation fusion enabled |
| `VLLM_FL_MXFP4_MARLIN=1` | FlagGems MXFP4 experts adapter | Hopper, native MXFP4 W4A16 quantization method and Marlin selection; standard activation format |

AttnRes preserves the PR custom-op mutation schema, with explicit model/warmup
bindings and installed/executed logging. Its vendored source provenance records
runtime import adaptations and a forward-only check that permits Parameter
weights under `torch.no_grad()`.

PR773 uses vLLM's native AllReduceFusionPass patterns. Enable
`--compilation-config '{"pass_config":{"fuse_allreduce_rms":true}}'`.
The FL RMSNorm compile path emits native IR for this opt-in configuration.
The worker initializes provider and native fallback workspaces after native
device/TP initialization and before model loading, AOT cache loading or graph
capture. This also covers a saved AOT graph that skips fusion-pass construction.
Preflight metadata and peer checks use the TP CPU group. An unsupported preflight uses
native fusion; an error after collective execution starts is propagated.
FlagCX must not be bypassed. TP16/32 are outside this provider's support.
Installing the hook is not evidence that a model has a matching compiled graph.

The vendored PR773 includes a phase-barrier repair beyond the upstream PR:
every CTA acknowledges phase consumption before program 0 resets its counter.
Controlled delayed-CTA tests reproduce the original hang on peer and MNNVL;
the repair passes the same tests. Same-host Qwen3.6-27B TP2 controls change only
this vendor file: the original hangs after the first request, while the repair
returns the arithmetic and GPU-explanation responses plus a separate short
sequence response. The original 96-token sequence response is retained as
truncated, rather than counted as a correct next-number answer.

The peer backend also retires slots using the preceding launch's strategy.
A partial two-shot prefill uses a compact scatter layout that differs from
one-shot rank slices. With TP2, hidden size 5120 and workspace size 4096,
a 3500-token two-shot call followed by a 64-token rowwise or 1-token tiled
one-shot left 1525760 elements uncleared in the previous implementation.
The rowwise path also retained the two-shot strategy tag. Both one-shot paths
now use the existing strategy-aware clearing helper, and the rowwise path
publishes its own strategy. The same probes leave zero uncleared elements.

The manual regression `tests/manual/check_allreduce_rms_strategy_gpu.py`
checks those invariants and 360 mixed-shape calls with PDL on/off, including
144 graph replays with changed inputs. Numerical comparisons sample up to
128 rows per call against an independent reference. This qualification is
peer TP2/BF16; it does not qualify other topologies, dtypes or a concurrent
model workload. The earlier strategy-only dense TP2 candidate passed full
graph capture, three semantic requests and serial 1K/4K/16K/64K-input-token
requests, but its subsequent 64-choice, 512-output-token functional request
timed out. The matched native-fusion control passed that same batch, including all 64
choices and 32768 output tokens. The paired runtime and service arguments
were audited; neither qualification arm launched a throughput client.
No throughput gain or complete large-load repair follows from the operator
regression. Provider dispatch/workspace diagnostics use the vLLM logger
namespace so they remain visible under the vLLM CLI logging configuration.

A subsequent same-host diagnostic compares ordinary PDL forwarding with
forced PDL-off dispatch, leaving the kernel unchanged. Both arms complete
full graph capture, three semantic responses, four serial input lengths and
the mixed-length 64-choice batch with 143736 prompt and 32768 completion
tokens. Dispatch logs confirm PDL on/off on both ranks. This does not establish
PDL-off as a remedy for the earlier timeout, or isolate a host effect. No
PDL-disable workaround is included. Separately, setting both CUDA module
code and data loading to EAGER exhausts the bounded initialization deadline
in all four dense/MoE native/peer arms, before any functional request.
Unmodified combined-public-source native/peer qualification is separate;
the earlier failed qualification and performance rounds remain unchanged.

### CTA warp retirement

A separate instrumented dense TP2 run locates a remaining stall inside the
peer rowwise one-shot kernel, with a 64-CTA, 256-thread launch. Rank 0's
remaining CTA 0 warps read arrival count zero while the SASS loop waits for
64. Scalar polling executes in multiple sibling warps: a leader can retire
and reset the counter before the remaining warps observe its completed value.
The diagnostic times out and remains separate from performance evidence.

The repair adds CTA barriers before publishing metadata arrival, before a
leader retires a polled counter, and before a CTA acknowledges that its warps
have consumed a phase counter. PDL, backend selection and communication
layouts remain unchanged. A bounded controlled-warp-delay reproducer reports
96/224/992 affected threads in 4/8/32-warp configurations; with the retirement
barrier, nine eager checks and nine changed-input graph replays have zero
mismatches. This reproducer models the protocol race; it does not call the
complete AllReduce kernel or measure throughput.

The actual repaired kernels pass the 360-call/144-replay peer TP2/BF16
regression above. Additional full-matrix interleaved graph checks cover
peer TP2/FP16, MNNVL TP2/BF16, peer TP4/BF16 and peer TP8/BF16, with 84 checks
per rank for patterns 1/2 and explicit/in-place outputs. These operator
checks do not qualify whole models on those other topologies or FP4 output.
The combined repaired runtime passes 103 CPU contracts with zero skips.

Independent dense and MoE TP2 native/peer model pairs now complete full graph
capture, three semantic responses, four serial input lengths and the original
mixed-length C64/512-output functional batch. Each batch returns 64 choices,
143736 prompt tokens and 32768 completion tokens. Their exact running source,
service arguments and provider dispatch are audited. Separate official
repaired-source throughput comparisons are running; these functional checks
do not establish a throughput gain or general workload reliability.

```bash
python tests/manual/check_allreduce_rms_warp_retirement_gpu.py
```

H100 peer qualification covers TP2/TP8 operator correctness and the TP2 dense
model control above. TP8 checks exercise BF16/FP8, explicit/in-place outputs and
changing-input graph replay; they do not establish a TP8 whole-model result.
The CTA repair additionally has the scoped TP4 operator coverage above.
MNNVL qualification currently
covers TP2 operator correctness, including FP8
outputs and changing-input graph replay. It does not establish MNNVL TP8,
cross-host transport or whole-model correctness. These are correctness checks;
they do not measure throughput or dataset accuracy. All providers remain opt-in.

The MXFP4 adapter retains native checkpoint loading and prepare/finalize logic.
It uses the pinned FlagGems-vllm optimized GEMMs, including
[PR4483](https://github.com/flagos-ai/FlagGems/pull/4483) and
[PR4943](https://github.com/flagos-ai/FlagGems/pull/4943).
Activation is applied separately through native vLLM to preserve clamp and
specialized activation semantics. Standard EP maps are supported; batched EP,
EPLB, activation quantization, LoRA, and expert bias are rejected explicitly.
Operator checks alone do not establish multi-host service qualification; see
the frozen model-specific results below.

The current MXFP4 hook accepts the exact native `Mxfp4MoEMethod` contract.
Kimi-K3's compressed-tensors checkpoint instead selects
`CompressedTensorsW4A4Mxfp4MoEMethod`, whose H100 Marlin branch uses W4A16 and a
different loader/parameter lifecycle; it is not covered by this hook. In the
reference TP32/EP32/DP1/PCP1/SP1 configuration, the requested DeepEP v2 backend
is inactive and prepare/finalize uses Standard format. Batched EP is therefore
not the reason for Kimi's current exclusion. Latent H3584/I3072 alignment and
native SITU beta4/linear-beta25 make a dedicated adapter a feasible candidate,
but it still needs compressed-tensors loader, SITU, latent output-contract and
EP32 numerical/graph qualification. The current four-host test enables only
AttnRes alongside native Marlin; DeepSeek results do not qualify all MXFP4
checkpoint methods.

### MXFP4 storage lifecycle and optional I192 extension

With the MXFP4 provider enabled, `VLLM_FL_MXFP4_RETIRE_RAW=1` is the default.
After loading and packing, the original Parameter objects alias packed storage;
their loader identity is retained and the original raw storage is released.
Aligned parameter shapes are retained. These aliases no longer contain the raw
checkpoint layout. Processing is idempotent, stale pack caches are removed,
and parameter/packed storage signatures are checked before execution. Device,
dtype, layout or quant-method changes require reloading the model; a native
fallback cannot reinterpret retired bytes. Set `VLLM_FL_MXFP4_RETIRE_RAW=0` only
when deliberately retaining both copies for diagnosis.

`VLLM_FL_MXFP4_COMPACT_I=0` is the default. The explicit value `192` supports
DeepSeek-V4-Pro's original pure-TP16 local intermediate size, which native
Marlin pads to 256. The adapter verifies that all removed gate/up/down FP4 codes
have zero magnitude, then compacts the corresponding group32 scales without
changing precision, TP, EP or native activation/clamp. Nonzero padding fails.

The compact path includes a **new plugin K64 down-projection kernel** in
`mxfp4_marlin_k64.py`. It reuses the pinned PR decode helpers but changes scale
indexing to two group32 scales per tile. GEMM1 and aligned-I paths retain the
original K128 kernels. This K64 extension is additional integration work; its
performance must not be attributed to an unmodified PR4483/PR4943 result.

H100 correctness checks cover BF16/clamp10, both router-weight placements,
fold/nonfold scales, graph replay, and compact H7168/I192 at M1/16/128
(BLOCK_M16/32/64). Compact and original zero-padded K128 outputs are bitwise
equal in these cases, with independent dense checks of each GEMM. Simulated
standard-EP maps validate expert partial sums. These simulated checks alone do
not establish multi-host transport or whole-model qualification; see the frozen
model-specific results below. The K64 tests qualify the tested BF16 path; the
FP16 decoder branch has no GPU qualification here.

Real-size single-layer storage tests prove original storage expiry and packed
aliases: Flash TP8 E256/H4096/I256 uses 408MiB raw→480MiB packed, while Pro TP16
E384/H7168/loaded-I256→compact-I192 uses 1071MiB raw→945MiB packed. These are
weight-storage measurements, excluding KV cache, graph pools, workspaces and
communication. They establish memory feasibility at layer size and do not
measure model throughput or guarantee whole-model memory headroom.

The Flash TP8 service passed the three real smoke requests with and without
raw retirement. vLLM reported model-loading allocations of 40.06GiB per rank
for the raw-plus-packed adapter and 22.93GiB for the retired version, a
17.13GiB reduction. This is the adapter's storage-lifecycle comparison, not a
native-vLLM comparison, peak-memory measurement or throughput benchmark.

The opt-in adapter selects the dependency's existing four-stage alignment
fallback for its default alignment policy. The original TLE candidate failed
CUDA Graph capture in the Pro service; singleton and small-group policies are
unchanged. Exact Pro-sized operator checks pass native clamp10 and changed-input
graph replay. For more than 65535 token rows, top-k reduction uses balanced row
views and the original reduction kernel, retaining its per-token expert order
and FP32 accumulation, to avoid an illegal CUDA grid Y in its TOKENS=1
autotuning candidate. The exact M65536/H5120/top-k6 case, three dtypes,
strided output and changed-input graph replay pass independently. These are
integration repairs beyond the original PRs; whole-model qualification remains
specific to each frozen service attempt.

The current Standard-EP adapter maps foreign routes to a valid dummy local
expert and zeroes their outputs after GEMM2. It still runs both GEMMs and the
activation for those routes; zero router weights do not short-circuit the dot
loops. In the actual DeepSeek-V4.1 DP16/EP16 gathered M65536/top-k6 case, 15/16
of route slots are foreign in aggregate across ranks. Per-rank routing skew
varies. Its BF16 first/activated/second temporaries total 8.8125GiB per rank at
H5120/I2304, excluding routing, native communication, graph pools and weights.
This is an allocation formula, not an incremental-memory or peak measurement.
Functional EP success therefore establishes no acceleration over native
Marlin; future performance work must skip foreign GEMM work while retaining
safe zero outputs and native EP contracts. Likewise, Pro's compact I192/K64
storage feasibility does not imply a 25% compute or throughput improvement.

### Marlin profile observations

A separate Flash TP8 diagnosis collected 48 raw CPU/CUDA traces: both provider
arms, three bounded windows and all eight ranks. This uses functional batches
rather than the official throughput load. In the warmed 12-step, M64 decode
window, summed GEMM kernel durations are 59.74–60.12 ms per native rank and
58.62–59.47 ms per adapter rank. The corresponding GPU activity span is about
158.3 versus 170.8 ms. Thus this window does not support a general claim that
the adapter GEMM itself is slower; extra work and launch gaps also matter.
Cumulative attention key lengths differ slightly between these windows.

The long-input windows have matching execute annotations, but the adapter
launches 50455–52148 GEMMs and 40702–42058 zeroing kernels per rank, versus
1376 GEMMs in the native arm. The large additional launch count is consistent
with the pinned autotuner's benchmark/reset path. Its tuning key includes
`EM_BUCKET`, supplied with the actual aligned token-slot allocation length
except for the singleton specialization, and uses an align32 key strategy.
The three-length tuning-policy counterfactual below now tests this mechanism.
These observations do not establish the original PRs' individual benefits, a
critical-path percentage, or the cause of the EP16 regression.

The adapter's long diagnostic request timed out after raw trace export;
sampled worker stacks showed optional profiler CPU table aggregation.
That request is retained as failed, with no completed usage claim. Earlier
same-runtime official rounds remain separate successful evidence. Future
profiling disables the optional table while retaining raw CPU/CUDA traces.
The functional diagnostic's different request count and output length cannot
replace the formal C64/N256 comparison or qualify an optimization's throughput.

### Dynamic tuning-key counterfactual

A follow-up native/adapter diagnosis records all eight ranks in 1K, 4K and
16K functional windows. In the second 1K-input window, each adapter rank
encounters six new tuning keys, taking 168.763–169.372 seconds cumulatively
per rank. First 4K/16K windows each encounter two new keys, taking about
59 seconds per rank. CPU tuner annotations align with real execute events
and thousands of benchmark GEMM/zeroing launches. This directly establishes
request-time tuning work; its wall time includes benchmark, JIT and cache
work, not compilation alone. Parallel rank times must not be added.

The key includes padded routed-slot allocation length, which changes with
scheduled token batches even when client input length stays the same.
The tested candidate leaves singleton and small-EM keys unchanged and rounds
EM >= 8192 up to the next power of two for the **tuning key only**. Actual EM,
launch grid, valid-token count, strides and GEMM math remain unchanged.
Component tests using five observed large model shapes and both projections
reduce ten new-key misses to four, with total miss wall time falling from
295.991 to 117.027 seconds. Each arm passes 48 full-output numerical checks
and 48 changed-input/index graph replays. These use synthetic weights/routes
at actual shapes, not captured model tensors.

A separate Flash TP8 official comparison then completes all 18 rounds for
1K/4K/16K. Both arms use the original adapter; only the key helper and its
installation differ. No route fusion, mask/reduction fusion, observer or
profiler is included. The key policy is now integrated through the adapter's
own GEMM callable after separate combined qualification; this table remains
the earlier key-only experiment. Original rounds 2/3 means are:

| Input tokens | Original key output tok/s | Bucketed key output tok/s | Change | Retained spread before / after |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 1057.955 | 3001.290 | +183.69% | 95.61% / 1.68% |
| 4096 | 1633.745 | 1649.895 | +0.99% | 0.07% / 0.54% |
| 16384 | 574.515 | 571.885 | -0.46% | 0.84% / 0.23% |

Every round passes the original 256-request/fixed-length/raw-JSON audit.
At 1K, retained baseline rounds have 81.486 and 388.983 seconds cumulatively
in global token-arrival gaps longer than one second; neither candidate
retained round has such a gap. Its first round still has 136.688 seconds.
At 4K/16K, both arms' retained rounds have no such gaps and throughput is
approximately unchanged. Together with the component intervention, this
supports reducing repeated dynamic-key tuning stalls, not a general
steady-state GEMM speedup or a stable +184% deployment gain.

Both arms start with separate empty task caches; initialization and identical
semantic probes populate them before the official client starts. Caches are
not cleared between rounds. Allocation is 22.93 GiB/rank in both arms, but
reported KV capacity is 801144 versus 801070 tokens, not fixed KV. There are
only two retained samples. The formal run has no tuner observer, so its client
gaps cannot be assigned to individual keys or pure compilation. These new
cache histories cannot allocate percentages of the older native/adapter
regressions or erase their slow rounds.

A second independent candidate fuses route range checks, EP mapping and
zero selection into one kernel, retaining invalid/padding/NaN/Inf semantics.
Its component traces confirm seven launches become one; numerical and
changed-input graph checks plus three-length whole-model functional checks
pass. Its separate official comparison now completes all 18 original rounds,
with every 256-request detailed result, source, client and command audited.
Original rounds 2/3 output-throughput means are:

| Input tokens | Original route output tok/s | Fused route output tok/s | Change | Retained spread before / after |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 1131.255 | 1093.160 | -3.37% | 74.91% / 92.10% |
| 4096 | 1360.940 | 1727.825 | +26.96% | 38.72% / 0.77% |
| 16384 | 528.685 | 545.450 | +3.17% | 15.98% / 31.30% |

This route-only intervention retains the original tuning-key policy. At 1K,
the candidate's retained rounds still contain 82.408/362.580 seconds in global
token-arrival gaps longer than one second. At 4K, the baseline's third round
has a 77.457-second gap while neither candidate retained round has one. At
16K, the third baseline/candidate rounds contain 77.198/154.808 seconds in
such gaps. These unequal stalls and large retained spreads prevent treating
the aggregate changes as pure routing-kernel savings or a stable deployment
gain. No gaps or slow rounds are removed from the throughput calculation.

Allocation remains 22.93 GiB/rank; reported KV capacity is 789040 versus
801443 tokens, so this is not fixed KV. The two arms have separate initially
empty task caches populated by initialization and identical probes. This
route policy is now integrated through the adapter's CUDA routing entry point
after the separate combined qualification below. These individual candidate
results do not establish combined performance.

### Combined production integration

The adapter now calls the bucketed GEMM helper and fused CUDA routing helper
directly. It preserves the CPU routing reference, the separate K64 down
projection and the existing masked reduction. There are no process-wide
FlagGems function replacements or experimental observer hooks. The provider
remains opt-in. Real EM, launch grid, tensor strides and kernel arguments
remain identical to the pinned dependency; 540 CPU launch-contract cases
check this independently of the tuning-key change.

The exact combined runtime passes 103 CPU tests with no skips. GPU validation
uses the real adapter entry points: 8 adapter cases and 12 K64 cases pass with
unchanged references/tolerances; 16 observed-EM synthetic cases pass 48
full-output numerical checks and 48 changed-input/index graph replays.
Routing covers 48 nonempty cases with 144 numerical and 144 changed-value,
index and expert-map graph checks, plus 12 empty boundary cases. Empty cases
are not counted as GPU graph validation. Masked reduction passes 36 cases,
108 eager numerical phases and 132 changed-mask graph phases.

The combined Flash TP8 service passes three semantic responses, serial
1K/4K/16K probes with 8 output tokens, and C64 batches at each input length
with 128 output tokens. Each concurrent batch returns all 64 choices and
8192 completion tokens. Its original adapter messages lacked rank identity,
so aggregate READY/HIT counts were not used as per-rank dispatch proof.
A separate, smaller check uses the identical runtime and standard vLLM
logging configuration to attach PID/worker identity to the existing messages.
It passes the same semantic responses and one C64/1K/8-output batch; all eight
worker PIDs have READY, HIT and graph-completion records. The original full
functional results and this supplemental evidence are audited together,
without replacing the original results or changing numerical code.

The official three-length comparison of the original adapter against the
combined runtime completes all 18 original rounds. Every round passes the
256-request, fixed-input/output-length and detailed-result audit; both
37-file archives match the remote originals byte for byte. Original rounds
2/3 output-throughput means are:

| Input tokens | Original adapter output tok/s | Combined output tok/s | Change | Retained spread before / after |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 927.625 | 2604.310 | +180.75% | 47.44% / 20.24% |
| 4096 | 1634.335 | 1839.770 | +12.57% | 0.15% / 0.28% |
| 16384 | 572.720 | 668.230 | +16.68% | 0.35% / 0.71% |

At 1K, cumulative global token-arrival gaps longer than one second fall from
285.077/141.648 seconds in the retained baseline rounds to 32.834/13.155
seconds in the candidate. The candidate's first round still contains
127.819 seconds. Stalls are reduced but remain, and its 20.24% retained
spread does not establish a stable +180.75% deployment gain. At 4K, neither
arm's retained rounds has such gaps. At 16K, both baseline retained rounds
have none; the candidate's second round has one 1.032-second gap and its
third has none. The 4K/16K improvements are therefore also observed in
retained rounds without large global arrival stalls. These client gaps do
not identify individual tuning keys, pure compilation or kernel critical
paths. No gap time is subtracted from throughput and no slow round is removed.

Both arms retain TP8, FP8 KV, max context 67584, max sequences 64, batch tokens
4096 and memory utilization 0.85, with the official RUNS3/SKIP_FIRST1,
OSL1024/C64/N256 load. Separate task caches start empty and are populated by
initialization and identical semantic probes; they are not cold per round.
Model allocation is 22.93 GiB/rank on both sides, but reported KV capacity is
800902 versus 802191 tokens, so this is not fixed KV. There are only two
retained samples and no formal observer, profiler or logging override.

This is original-adapter to accumulated production integration, not native
versus adapter or a single-variable intervention. Individual key, route and
reduction percentages must not be added together or used to apportion older
regressions. The unchanged candidate source has separate per-rank dispatch
qualification above; the formal run's unlabelled aggregate READY/HIT counts
are not used as per-rank proof. All eight ranks complete graphs, both arms
pass the same three semantic responses, and both services exit after their
original rounds. Earlier slow and failed cases remain intact.

### Masked top-k reduction

The adapter now fuses route masking into the reduction loads, avoiding a
separate mask write and second read of GEMM2 output. Invalid routes are load
predicates, so dummy-expert NaN/Inf cannot reach arithmetic. Small-token
pairwise FP32 accumulation and large-token sequential accumulation retain
the pinned reduction order, including odd top-k; the flattened grid supports
large token counts. Unsupported layouts retain the legacy path.

Qualification covers 36 GPU cases, 108 numerical phases and 132 changed-mask
graph phases, including all-false masks, three dtypes and an assertion of the
actual fused dispatch. A frozen public-source Flash TP8 service completes full
graph capture, three semantic responses, serial 1K/4K/16K/64K-input probes and
a C64/1K/128-output functional batch. These are functional checks.

A separate Flash TP8 official comparison completes all 24 rounds. Both arms
use the adapter; only the three masked-reduction runtime files differ. The
means below use original rounds 2/3, with all three raw rounds retained and
256 requests of 1024 output tokens audited in every round.

| Input tokens | Original adapter output tok/s | Fused adapter output tok/s | Change |
| --- | --- | --- | --- |
| 1024 | 3255.375 | 2363.030 | -27.41% |
| 4096 | 1930.805 | 1982.860 | +2.70% |
| 16384 | 740.065 | 751.935 | +1.60% |
| 65536 | 198.130 | 211.570 | +6.78% |

The fused 1K retained-round spread is 83.30%, versus 0.33% in its control.
The slow retained round is included; available compilation logs do not prove
its cause. Only two samples contribute to each mean, so these are observed
differences, not established stable gains. Both arms report 22.93 GiB/rank
model allocation; automatic KV capacity is 815790 versus 815920 tokens, not
a fixed-KV counterfactual. This isolates the fusion within the frozen adapter
runtime, not the original upstream PR4483/4943 gain or a whole-source retest
of the later combined public runtime. It does not erase the earlier
native-versus-adapter regressions.

```bash
PYTHONPATH=. python tests/manual/check_marlin_masked_reduce_gpu.py
```

## Existing unquantized MoE path

The pinned dependency already contains
[PR3340](https://github.com/flagos-ai/FlagGems/pull/3340). `TritonExpertsFL`
uses it for ordinary BF16/FP16 experts. Its monolithic API has no
`activation_config` parameter, so configured clamp and specialized activation
variants use the modular FlagGems GEMM pipeline with native activation instead.
Quantized experts are not automatically covered by this unquantized path.

PR3340 separates GEMM1/GEMM2 launch defaults, selects tiles using routed-token
density and dtype, pairs gate/up projections into a wider dot in eligible
fused-SiLU GEMM1 calls, and permits atomic GEMM2 accumulation into the final
output to avoid a separate top-k sum. The large half-precision fast paths
generally require a chunk with at least 4096 rows, suitable alignment and no
overriding YAML configuration; direct accumulation additionally excludes
expert maps and input-side router weighting. Small decode batches usually do
not meet those thresholds. Request input length is not kernel row count:
the tested services chunk prefill at 4096 tokens.

MiniMax-M3's configured SwiGLU-OAI path has partial PR3340 kernel coverage:
the shared low-level kernel still omits N-index modulo and output-store
boundary masks for divisible output tiles. It uses vLLM's configuration for
both GEMMs and keeps native activation plus separate top-k reduction, so the
PR's stage-specific tuning, fused SiLU/pair-gate-up and direct-sum fast paths
are not enabled. This coverage does not establish a throughput gain.

### Independent PR3340 ablation results

A separate frozen investigation reverts exactly the seven function definitions
changed by PR3340 while retaining the rest of the pinned FL stack. This is
an independent old-versus-optimized comparison; the ablation instrumentation
is not part of this PR. It does not switch the entire MoE backend to native.

MiniMax-M3 BF16/TP16 completes all 24 official benchmark rounds. All detailed
JSON files pass the 256-request and fixed input/output-length checks. Means
use original rounds 2/3 under the protocol documented below:

| Input tokens | Pre3340 output tok/s | Optimized output tok/s | Change |
| --- | ---: | ---: | ---: |
| 1024 | 1247.21 | 1245.66 | -0.12% |
| 4096 | 816.76 | 680.215 | -16.72% |
| 16384 | 332.275 | 283.045 | -14.82% |
| 65536 | 97.78 | 79.615 | -18.58% |

This result covers only the shared low-level GEMM change described above.
It does not measure all of PR3340's monolithic fast paths or establish their
performance on other architectures. At 64K, mean TPOT improves by 43.89%,
while TTFT increases by 26.21% and output throughput falls by 18.58%; a faster
individual metric cannot be presented as an end-to-end gain.

Identical launch arguments did not yield identical effective KV capacity:
pre3340 reports 268928 tokens (3.98 maximum-length requests), versus 185472
(2.74) in the optimized arm, while both report 50.41 GiB model allocation per
rank. This capacity difference can affect scheduling. Synthetic routed GEMM
checks at the model's TP16 local dimensions match numerically and show mostly
small warmed-kernel improvements; they do not explain the entire model-level
regression. A fixed-capacity counterfactual has not been measured.

Qwen3.8-Flash-Next TP8 has valid 1K/4K paired means of 1420.88→1443.64
(+1.60%) and 996.49→1012.39 (+1.60%) output tok/s. These small observed changes
are not established stable speedups. Both arms fail the 16K/64K groups;
their failed rounds are preserved and excluded from throughput comparisons.

A separate repaired padding/no-TLE configuration passes serial long-input
probes, but both exact-ablation arms still fail the official 16K/64K groups.
All 24 raw rounds have been audited: each arm has six successful short-input
rounds and six failed long-input rounds. New fixed-source 1K/4K means are 1341.82→1324.03
(-1.33%) and 924.23→879.855 (-4.80%) output tok/s. Both short groups have
three successful raw rounds per arm. The first asynchronous error is reported
while GDN linear attention indexes its initial state with a Boolean mask.
A separate launch-blocking C64/16K functional diagnostic locates the first
synchronous CUDA failure at the later GDN recurrent-state write through
FlagGems `index_put`. The signed-int32 offset repair below passes the tested
functional batch; its new formal comparison remains separate and pending.
This diagnostic is not an official throughput result. Recovery
starts only the arm that never benchmarked after an SSH transport failure.
No measured optimized round or earlier configuration result is replaced.

A common four-stage alignment configuration completes Hy3 TP16 graph capture
on all ranks and three semantic requests in both ablation arms at max length
67584. Its exact-seven-function PR3340 comparison has audited 1K/4K/16K means
of 1675.96→1708.985 (+1.97%), 1181.125→1194.895 (+1.17%) and
473.20→485.415 (+2.58%) output tok/s. Original rounds 2/3 supply these means,
with all three raw rounds retained. The 64K group completes at
136.315→137.675 (+1.00%) output tok/s; all 24 rounds pass the detailed
256-request audit. Two retained samples do not establish stable speedups. Reported KV capacity is
870160→875792 tokens under the same launch flags, with both model allocations
34.82 GiB/rank; capacity was not independently fixed.

Qwen3.6-35B-A3B additionally requires the metadata-padding repair below.
Both exact-ablation arms pass full graph capture and three semantic requests
at the original full-context configuration. The distinct fixed-source TP2
comparison completes all 24 rounds with detailed 256-request and fixed-length
JSON audits. Means of original rounds 2/3 are:

| Input tokens | Pre3340 output tok/s | Optimized output tok/s | Change |
| --- | ---: | ---: | ---: |
| 1024 | 1758.02 | 1884.215 | +7.18% |
| 4096 | 1262.07 | 1296.53 | +2.73% |
| 16384 | 562.57 | 571.21 | +1.54% |
| 65536 | 172.575 | 175.905 | +1.93% |

At 1K, the baseline's two retained rounds have an 11.18% spread, versus
1.41% after optimization. These two-sample observations do not establish
stable speedups. Both arms report the same 32.81 GiB/rank model allocation and
3600108-token KV capacity. Earlier failed configurations and every measured
round remain unchanged.

### CUDA graph padding compatibility

The NVIDIA worker selects the pinned dependency's existing no-TLE four-stage
aligner before model construction. This avoids the speculative TLE alignment
path that invalidated graph capture in this stack. Changing the aligner alone
did not qualify Qwen3.6: a launch-blocking diagnostic located its initial
illegal access in alignment stage4, then a graph-cleanup collective obscured
that first exception.

Actual route snapshots identify native metadata-marked padding rows whose
expert IDs are all -1 while their normalized weights remain nonzero. Before
the monolithic FlagGems call, only rows marked by vLLM's padding mask receive
finite zero hidden states, zero weights and a safe dummy expert ID. The final
output is explicitly zeroed on those rows, including NaN/Inf dummy-expert
results. Active rows and their routes are preserved. The no-mask path retains
the original input aliases; no Python tensor-value reads are introduced.

Validation includes 95 CPU contracts, 24 old/new GPU numerical and changed-mask
graph checks, and 12 GPU cases from the public reproducer. The exact public
runtime, with all private ablation code removed, independently completes full
TP2 graph capture and three semantic requests for Qwen3.6-35B-A3B, BF16,
max length 67584. These checks establish the tested functional contract, not
dataset accuracy or throughput. Other model qualifications above retain their
own frozen sources and must not be described as reruns of this combined patch.

On an available NVIDIA GPU, run the public operator check from the repository:

```bash
PYTHONPATH=. python tests/manual/check_moe_padding_gpu.py
```

## Qwen GDN state-index width

A launch-blocking long-batch diagnostic records the same failing write on all
eight ranks: FP32 state shape `[7466, 6, 128, 128]`, element strides
`[1279488, 16384, 128, 1]`, and legal int32 indices `[1682, 1655]`.
The selected maximum element offset is 2152197119, beyond signed int32.
The pinned FlagGems generator multiplies the loaded int32 index by the stride
without widening it. The resulting illegal access occurs in recurrent-state
writeback; the earlier asynchronous Boolean-index report was a later reporting
site. An independent numerical counterexample also distinguishes failing int32
writes from correct native and int64 controls.

Only for the `qwen4_exp` architecture with FlagGems enabled, the worker wraps
that dependency entry point after device setup. FP32 rank-four inputs with a
single one-dimensional int32 index use int64 indices when the metadata-derived
maximum strided element offset exceeds signed int32. It uses no host tensor
value reads and preserves input/output identity, values and other paths.
A numel-only guard would miss the actual strided layout.

The isolated repair passes 18 numerical cases, 54 changed-value graph replays,
four additional cases with 12 changed-index/value replays, and eight CPU
contracts. A separate frozen public-source qualification passes eight-rank TP8
full graph capture, three semantic
responses, serial 1K/4K/16K/64K probes and the previously failing C64/16K batch
with 64 choices, 1048576 prompt tokens and 65536 completion tokens. This is
functional evidence, not an official 256-request throughput result or dataset
accuracy. A new exact-PR3340 pair shares the repair in both arms and keeps every
older failed round unchanged. That source-327 pair has completed all 24
baseline/optimized rounds successfully. All four paired groups pass detailed
256-request, fixed-input-length and fixed-1024-output-token audits:

| Input tokens | Pre3340 output tok/s | Optimized output tok/s | Change |
| --- | --- | --- | --- |
| 1024 | 1341.725 | 1348.515 | +0.51% |
| 4096 | 911.230 | 924.000 | +1.40% |
| 16384 | 403.650 | 411.420 | +1.92% |
| 65536 | 125.160 | 126.845 | +1.35% |

Means use original rounds 2/3 and retain all three. Before/after retained
spreads are 0.34/0.06%, 1.82/0.16%, 0.40/0.33% and 0.34/0.72%; two retained
samples do not establish stable gains. Both arms report 31.19 GiB model-loading
allocation per rank; reported KV capacity is 5667506 versus 5672648 tokens, so
this is not a separately fixed-KV experiment. This exact-seven-function
comparison shares the index repair in both arms. It is not a whole-source retest of the later CTA repair
or a replacement of any older failed round.

On a free 80GB GPU, the public check includes the captured strided layout and
both index widths, with native controls and changed-index/value graph replay:

```bash
PYTHONPATH=. python tests/manual/check_qwen_gdn_state_index_gpu.py
```

## MiniMax indexer compatibility

The pinned FlagTree AABS analyzer incorrectly associates the native MiniMax
top-k sort tile `BLOCK_SIZE_K` with `init_blocks`, which counts forced initial
blocks. With `init_blocks=0`, it shrinks the tile to one and violates the
sorter's `K > T` requirement during CUDA graph warmup. The MiniMax architecture
hook removes only this incorrect dependency from that JIT kernel's cached
analysis. It retains the original kernel, wrappers and autotuning configs.
This compatibility fix is separate from the MoE PR3340 coverage above.

## Kimi complex-operator compatibility

Kimi-K3's native vision RoPE builds complex frequency tensors. The pinned
FlagGems cat, slice and repeat implementations do not support this complex
path. `patch_kimi_complex_ops` captures the original CUDA kernels before
FlagGems registration, then routes complex cat (including `out=`), slice and
repeat to those saved PyTorch kernels. Real and BF16 inputs retain their
FlagGems implementations. The hook is gated on enabled FlagGems and the
`KimiK3ForConditionalGeneration` architecture; it preserves native view,
stride and output-alias semantics without CPU copies or changes to RoPE math.

The NVIDIA worker installs this hook before `_initialize_fl_runtime` enables
FlagGems. Installing it after the affected operators are registered fails
explicitly. `FL_KIMI_COMPLEX_OPS_INSTALLED` records registration;
`FL_KIMI_COMPLEX_CAT_EXECUTED`, `FL_KIMI_COMPLEX_SLICE_EXECUTED` and
`FL_KIMI_COMPLEX_REPEAT_EXECUTED` record actual complex dispatch. These
compatibility markers are separate from PR737 AttnRes execution.

Fourteen CPU tests cover policy, delegation, call signatures and registration
order. Twelve H100 GPU cases cover complex64/complex128 cat and out variants,
strides, promotion, empty and nonfinite inputs, native Kimi RoPE construction
and indexed frequencies, native packed BF16 Q/K rotation, and changed-input
CUDA graph replay. Given identical FlagGems arithmetic, the compatibility
path is bitwise equal to a saved-native-cat control. The existing full-stack
FlagGems RoPE arithmetic differs from the all-native reference by at most
1.526e-5 in frequency components in this test; the resulting BF16 Q/K maximum
absolute difference is 0.015625 and passes the original 2e-2 tolerance.
These operator checks retain the native vision path and do not themselves
establish whole-model inference, multimodal accuracy or a throughput gain.

## Kimi temporal-pool mean compatibility

The native maximum-image profile uses grid `[1,502,134]`. Its temporal-pool
input `[1,251,2,67,2,1024]` makes the pinned FlagGems BF16 mean vector kernel
launch grid `(1,67268)`, exceeding CUDA's grid-Y limit of 65535. The Kimi-only
`patch_kimi_mean_grid` hook captures native CUDA `aten::mean.dim` before
FlagGems registration and selects it only for BF16/FP16 single-axis reductions
whose original vector grid Y would exceed that limit. Other means retain
FlagGems. It preserves axis, keepdim, dtype, nonfinite values and distinct
output storage, including temporal length one; it does not replace pooling
with a view or bypass the vision encoder.

`FL_KIMI_MEAN_GRID_INSTALLED` records registration and
`FL_KIMI_MEAN_GRID_EXECUTED` records the actual oversized shape. Sixteen CPU
contract tests and eleven H100 GPU cases pass, including the exact native
pooling function at temporal lengths one and two, grid-Y boundaries
65535/65536, strided input, FP16 output-dtype override, NaN/Inf and changed-input
CUDA graph replay. Tested outputs are bitwise equal to saved-native CUDA
references, with NaNs compared as equal. These compatibility checks are
separate from PR737 qualification and whole-model inference.

A standalone check also runs the native 27-layer ViT and projector with the
actual BF16 config and FlashAttention v3 backend. Finite seeded random weights
complete the maximum-image eager path twice, producing finite
`[16817,7168]` outputs that change with the pixels. This qualifies execution
through the complete vision path, not checkpoint semantics, multimodal
accuracy or multi-host language generation. Encoder CUDA graphs stay disabled
as in the frozen service; mean and pooling graph replay are checked separately.

## Qualification and interpretation

Kimi-K3 passes three real text smoke requests on four H100 hosts with its
original checkpoint, TP32/EP32, native compressed-tensors Marlin/SITU experts
and the new AttnRes provider. All 32 ranks execute five PR737 variants in eager
and capture modes; graph capture completes. Native maximum-image ViT profiling
also completes with the two scoped compatibility hooks above. The original API
responses contain XTML closure markers; the unchanged native Kimi parser
extracts semantic answers 4, a reasonable GPU explanation, and 10. All three
finish with `stop` and no reasoning or tool call. Both raw and parsed content
are retained. This max-model-len 8192 smoke run establishes neither multimodal
accuracy nor throughput improvement. New AllReduceRMS and MXFP4 providers are
disabled; requested `deepep_v2` is not dispatched in this DP1/SP1 configuration.

DeepSeek-V4.1-Flash-1017 passes three real text smoke requests on two H100
hosts with the original FP4-expert checkpoint, TP1/DP16/EP16 and Engram CPU
offload. The Standard-format gather/reduce-scatter pipeline executes the MXFP4
provider at aggregated M65536/H5120/top-k6/clamp10; graph capture also completes.
This run uses max-model-len 8192 and graph capture size 16, without DSpark/MTP.
It does not reproduce the reference performance scenario or evaluate vision.

DeepSeek-V4-Pro passes three real text smoke requests on two H100 hosts with
the original FP4 checkpoint, pure TP16, BF16 activations and clamp10. The I192
compact extension and raw retirement execute on all ranks; piecewise and full
CUDA Graph capture complete. Each rank reports 61.91GiB model-loading
allocation. This max-model-len 8192 smoke run includes the integration repairs
above and provides no native-baseline comparison or throughput result.

MiniMax-M3 passes three real text smoke requests on two H100 hosts with BF16,
TP16, the configured SwiGLU-OAI activation and native sparse attention. The
specific AABS compatibility repair above executes on all 16 ranks; MoE uses
the modular native-activation pipeline with only the partial PR3340 kernel
coverage described above. The max-model-len 8192 smoke run establishes neither
a reference-scenario throughput gain nor dataset accuracy.

Numerical tests live under `tests/functional_tests/`; provider policy and
provenance tests live under `tests/unit_tests/`. Manual multi-rank GPU checks
are under `tests/manual/`. Preserve each attempt's source hashes, commands,
raw responses, failures, provider execution evidence, and process cleanup
identity. CUDA graph tests use changed inputs on replay.

A service becoming ready, a hook being installed, generated kernel source, and
successful real requests are distinct evidence. Report each separately. Smoke
requests do not measure end-to-end throughput or full dataset accuracy.
For LLM throughput measurements use the repository's
`benchmarks/benchmark_throughput_serve.py` under the task's approved test
protocol; do not substitute a smoke client for that benchmark.

## Throughput campaign snapshot (2026-11-19)

The completed functional scenarios above are separate from the ongoing
throughput campaign. The latter uses the official
`benchmarks/benchmark_throughput_serve.py` from main commit
`a58cee83ecbabc4a4e874f34a118c65470e9237c`, frozen with `RUNS=3` and
`SKIP_FIRST=1`. All three original rounds are saved; summaries use rounds 2
and 3. The client SHA256 is
`1c5a03b675d026f69aa95563dcb3509e1946f747ec38c2fe17f15973c9943fe1`.
The load is random input, ignored EOS, output length 1024, concurrency 64,
256 requests per round and input lengths 1024/4096/16384/65536. Added native
result-saving and tokenizer compatibility flags do not change that load.

The two arms retain the same qualified source, necessary compatibility fixes,
image, checkpoint, precision, topology and service arguments. They switch only
the applicable new provider and use separate compilation caches. This compares
the incremental provider path, rather than an unmodified branch that cannot
complete the required model initialization.

The completed DeepSeek-V4-Flash TP8 input-length groups currently show:

| Input tokens | Native output tok/s | Adapter output tok/s | Change |
| --- | ---: | ---: | ---: |
| 1024 | 3225.33 | 1766.80 | -45.22% |
| 4096 | 1940.14 | 1599.52 | -17.56% |
| 16384 | 709.72 | 524.30 | -26.13% |
| 65536 | 200.07 | 152.68 | -23.69% |

The adapter's retained-round throughput spread is 135.56% for input 1024
and 15.58% for input 16384. These are observed two-round means, not stable
effect estimates or statistical significance claims. All 24 Flash rounds
completed successfully, with per-request JSON checked for 256 requests and
1024 output tokens each. The adapter currently shows no demonstrated
acceleration in these completed groups. Both dependencies already contain PR4483/PR4943, so
this is a native-vLLM versus adapter comparison, not an isolated measurement
of applying either upstream optimization.

For the same Flash pair, vLLM reports model-loading allocations of
19.91GiB/rank natively and 22.93GiB/rank with the adapter. Reported KV capacity
changes from 882608 to 797651 tokens. This is distinct from the adapter's
40.06-to-22.93GiB raw-retirement comparison above and is not peak memory.

### AllReduceRMSNorm loading qualification

The short dense-model qualification above does not establish the larger
67584-token service configuration. With default module loading, that dense
service stalled in its first official benchmark request; the MoE provider-on
service also encountered startup stalls or CUDA errors. Those failures are
retained and their rounds are not replaced.

A separate environment-repair configuration sets this on every worker before
starting vLLM, in **both** the baseline and provider arms:

```bash
export CUDA_MODULE_DATA_LOADING=EAGER
```

This configuration passed sequential 16/1024/4096-input-token functional
requests for Qwen3.6-27B TP2 and Qwen3.6-35B-A3B TP2 without
`CUDA_LAUNCH_BLOCKING` or additional model synchronizations. The MoE case
passed on H100 driver 580.126.20; another host with 580.105.08 still failed
startup. Driver, host and cache effects have not been independently isolated.
CUDA module-loading stacks and NVIDIA's
[lazy-loading concurrency guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)
motivate this configuration, but do not establish one root cause for every
failure. The separate formal paired tests have now completed: both native
arms recorded 12 successful rounds, while both provider arms recorded 12
failed rounds. The short requests therefore did not qualify the official
load. All original failures are preserved; no AllReduceRMSNorm end-to-end
speedup is established.

### Repaired AllReduceRMSNorm formal progress

Independent native/peer comparisons use the published CTA-barrier repair,
ordinary PDL forwarding and common `CUDA_MODULE_DATA_LOADING=EAGER`.
The original TP2/BF16 service and official client load are preserved. Each
model has nine successful rounds per arm for the retained 1K/4K/16K scope.
Detailed JSON, source/configuration, client and actual commands pass audit.
The user subsequently excluded 64K; in-progress 64K clients were interrupted
and their evidence retained as user-directed, not model failures. Completed
older 64K rounds remain historical data. The following are
arithmetic means of original rounds 2/3, with all three raw rounds retained:

| Model | Input tokens | Native output tok/s | Repaired peer output tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 1024 | 1580.180 | 1616.175 | +2.28% |
| Qwen3.6-27B | 4096 | 1189.185 | 1209.115 | +1.68% |
| Qwen3.6-27B | 16384 | 592.920 | 592.420 | -0.08% |
| Qwen3.6-35B-A3B | 1024 | 1901.960 | 1889.435 | -0.66% |
| Qwen3.6-35B-A3B | 4096 | 1290.130 | 1276.485 | -1.06% |
| Qwen3.6-35B-A3B | 16384 | 571.765 | 561.855 | -1.73% |

Two retained samples do not establish stable gains. Model allocations match
within each pair, but reported KV capacities differ: dense 1353932 to 1344921
tokens, MoE 3600108 to 3589013. No fixed-KV counterfactual was measured.
Other GPU work was observed on the MoE host during the peer arm, and a read-only
CPU-client export overlapped its native arm. These are shared-host observations,
not an isolated causal estimate. No rounds are excluded or rerun. The separate
older public-source MoE pair retains its frozen source and currently has only
one successful peer round followed by a natural 7200-second `sample_tokens`
RPC timeout and engine shutdown. The second 1K round returns only 128 requests
and 73227 output tokens, so it is failed and cannot provide a valid throughput
comparison. Subsequent planned failures are retained. No debugger was attached
to this performance case, and it is not upgraded or rerun in place.

### Exact PR3340 OCR comparison

Unlimited-OCR enters `fused_experts_impl` in both exact seven-function ablation
modes. With the reference service defaults, TP1/BF16 and the original checkpoint,
both modes pass graph capture, two image requests and three native EvalScope
2048-square-image requests with 1K/4K/16K text budgets. Actual server tokenization
matches the frozen CPU tokenizer IDs. Friend is recognized; Centre still returns
an image box, so these checks do not establish full OCR accuracy.

The independent formal comparison used the approved EvalScope 1.5.1
`random_vl` client: one 2048x2048 RGB image, text budgets 1024/4096/16384,
C64, N1024 and output 1024. All five rounds per group are retained; the main mean
uses original rounds 2–5. It does not use the LLM text-generation client. The
baseline's five 1K rounds pass independent read-only native SQLite integrity,
1024-request, fixed-output, argument, hash and metric checks. This is baseline
progress, not a paired throughput result or a benefit claim. The user then
removed OCR from the remaining validation scope. The client and service were
stopped, with completed baseline results and interrupted-round evidence retained.
The optimized arm was never started. The interruption is user-directed, not a
model failure, and no paired OCR performance conclusion is available.

### Two-host MXFP4 results and capacity limits

The larger Pro TP16 service configuration with max model length 67584 failed
KV-capacity initialization in both arms, before any throughput client ran.
The adapter's I192 storage reduction does not qualify this context length.
These startup failures remain separate from the 8192-token functional result.

DeepSeek-V4.1 DP16/EP16 completed all 24 native/adapter rounds successfully.
Source/configuration/provider execution and per-request JSON were audited.
The following means use the same original rounds 2/3 as Flash:

| Input tokens | Native output tok/s | Adapter output tok/s | Change |
| --- | ---: | ---: | ---: |
| 1024 | 2202.73 | 155.32 | -92.95% |
| 4096 | 1942.31 | 151.11 | -92.22% |
| 16384 | 1285.73 | 150.14 | -88.32% |
| 65536 | 786.48 | 288.09 | -63.37% |

Retained-round adapter spreads are 12.07%, 28.01%, 56.28% and 83.53%,
respectively. These highly variable two-round means do not establish stable
effect estimates.
Live process stacks show real Marlin autotuning and Triton compilation during
this run; shared-memory broadcast warnings alone do not establish deadlock.
Foreign-route work described above is a concrete optimization candidate,
not a measured causal partition of this regression. These are provider A/B
results; they do not isolate PR4483 from PR4943.

New performance work uses only 1K/4K/16K at the user's request; historical
64K results above remain preserved. AttnRes Kimi TP32/EP32 throughput remains
pending a separate three-length control handoff and resource qualification.
A host with zero GPU memory but an active checkpoint-triggered GPU queue is
not considered idle. PR3340 is already present in
the baseline, so its existing coverage does not constitute a new migration
speedup. All new providers remain experimental and disabled by default.

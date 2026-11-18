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

H100 peer qualification covers TP2/TP8 operator correctness and the TP2 dense
model control above. TP8 checks exercise BF16/FP8, explicit/in-place outputs and
changing-input graph replay; they do not establish a TP8 whole-model result.
TP4 has not been qualified in this campaign. MNNVL qualification currently
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

## Existing unquantized MoE path

The pinned dependency already contains
[PR3340](https://github.com/flagos-ai/FlagGems/pull/3340). `TritonExpertsFL`
uses it for ordinary BF16/FP16 experts. Its monolithic API has no
`activation_config` parameter, so configured clamp and specialized activation
variants use the modular FlagGems GEMM pipeline with native activation instead.
Quantized experts are not automatically covered by this unquantized path.

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

An isolated common four-stage alignment configuration also completes Hy3
TP16 graph capture on all ranks and three semantic requests in both ablation
arms at max length 67584. Its new paired throughput campaign is running;
this common repair is not yet included in the production PR. Qwen3.6 remains
unqualified: a launch-blocking diagnostic locates an illegal access in the
four-stage aligner's stage4, followed by a graph-cleanup collective that hides
the first exception. Successful random-input operator checks do not qualify
that model's actual routing and execution context.

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

AttnRes Kimi TP32/EP32 throughput remains queued. PR3340 is already present in
the baseline, so its existing coverage does not constitute a new migration
speedup. All new providers remain experimental and disabled by default.

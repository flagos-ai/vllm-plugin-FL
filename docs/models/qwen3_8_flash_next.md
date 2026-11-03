# Qwen3.8-Flash-Next on MetaX

This change **depends on [plugin-FL PR #455](https://github.com/flagos-ai/vllm-plugin-FL/pull/455)
being merged**, using its fixed revision
`72d6440bbdb4649375aad33390e176196a47104a` as the integration base.
It reuses the common Qwen3.8/Qwen4 configuration, model, loader and cache
components and adds the validated MetaX execution contracts.

The MetaX scope remains BF16 text serving on 8 C550 cards, TP8/PP1, MRV1,
FP32 recurrent state, shared CPU N-gram and **MTP off by default**.
MTP K1/K2/K3 remain explicit configuration choices. Target prefill/mixed batches
stay eager; the graph policy covers target and optional draft pure-decode forwards. Lookup, metadata preparation, logits and sampling remain outside
those forward graphs. The capacity target is a 64K input, not 64 simultaneous
full-length inputs.

## Increment over #455

| Location | Added or changed responsibility |
|---|---|
| `models/qwen3_8_flash_next/config.py` | Existing configuration plus the input capability and compatibility fields |
| `models/qwen3_8_flash_next/common/` | Reuse HC/PLE/cache owners; preserve MetaX rounding/order, prepared metadata and live cache views |
| `models/qwen3_8_flash_next/gpu/model.py`, `gpu/ple_layer.py`, `gpu/qsa.py`, `gpu/indexer_qsa.py` | Shared forward structure with scoped MetaX execution/provider choices |
| `models/qwen3_8_flash_next/gpu/host_ngram.py`, `host_lookup.py` | CPU table loading, hashing, shared results and bounded pinned staging |
| `models/qwen3_8_flash_next/gpu/mtp.py` | Optional sequential draft model with real 4H feedback, using the shared decoder/loader/HC components |
| `worker/qwen3_8_flash_next.py`, `worker/qwen3_8_flash_next_ple.py`, `worker/qwen3_8_flash_next_sampling.py`, `spec_decode/` | Runner hooks, PLE state, sampling metadata and optional draft graphs |
| `patches/qwen3_8_flash_next.py`, `patches/qwen3_8_flash_next_mtp.py` | Existing registration plus the scoped vLLM 0.24 MTP compatibility |
| `dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/`, `dispatch/config/` | MetaX policy and reviewed launch configuration over existing library kernels |

The prerequisite contains MTP model source but rejects speculative decoding in
its default verifier. This integration retains the existing MetaX K1/K2/K3 paths as explicit options.
Other platforms retain the prerequisite's default execution choices. There is
one common model implementation, with no duplicate flat model or parallel
attention package. Original attribution to [#573](https://github.com/flagos-ai/vllm-plugin-FL/pull/573)
is retained where applicable; that branch's HY4/T-Head changes are not imported.

## Runtime and operator dependencies

The [operator composition](../../examples/qwen3_8_flash_next/metax_tuning/operator_dependencies.json)
records the exact required files and hashes. The active tuning configuration
checks its source guards before applying launch settings. The launcher only
selects explicitly supplied package directories; it never installs, patches or
rewrites third-party sources.

The fixed framework is vLLM0.24.0 `ee0da84`, with FlagGems5.3.5
`a7620cc` and FlagGems-vllm base `0f3df8e`. Retain the published component
references FG#7981 (wide index products), FGV#959 (PLE), FGV#1000
(inplace MoE), and FGV#1024 (per-row TopK); exact references are in the
composition file. Native vLLM sources are unchanged by this optimization.

The measured assembly additionally requires private FGV radix, generic
sampling fallback, Top20 filtering, race reduction, QSA packing and BF16
MCTlass implementations, plus the FG index_select CPU-overhead fix.
The host-preparation stage also requires the FG index_put boolean-scalar fix.

These remain separate operator-library changes. The model submission carries
integration, policy and fingerprints; it is not fully reproducible from public
package versions alone. MCTlass requires the matching installed C ABI.

Use MetaX Torch2.10.0+metax3.8.1.0, FlagTree0.6.1+metax3.6,
MACA SDK3.8.1.3 and driver3.8.1 as in the validation image. Communication
continues to use MCCL/torch.distributed without FlagCX.

## Memory and state placement

With `qwen38_ngram_mode=shared_buffer`, TP rank zero owns the full pageable CPU
N-gram table (95.368 GiB). It hashes and gathers once; all ranks consume the
shared result and copy through their own bounded pinned staging into fixed GPU
buffers. This lookup needs no GPU all-reduce. The optional sharded host mode
is retained. MTP has no additional independent N-gram table.

BF16 activations/weights and the native TP partitioning remain unchanged.
QSA Q/K/V and all QSA KV/side caches are BF16; accumulation, softmax and scores
use FP32. GDN recurrent state is FP32. This is not an FP8 KV-cache deployment.

MTP preserves the target's four residual streams (4H); the logits stream is H.
Native sampling and acceptance bookkeeping remain in vLLM. Main/raw/compressed
QSA groups keep separate metadata and slots. K1 requires
`disable_padded_drafter_batch=true`; unpadding preserves the CPU sequence-length
bound. Prefix caching, PP>1, EP, MRV2 and multimodal serving are outside this
MetaX acceptance scope.

## MetaX execution policy

Use [the policy](../../vllm_fl/dispatch/config/qwen3_8_flash_next_metax.yaml).
Defaults remain available; scoped provider choices retain the previous runtime:

- GDN prefill uses FGV's FP32 V-major implementation. Speculative recurrence
  and causal convolution retain vLLM 0.24's accepted-token/null-slot semantics.
  When all current decode draft counts are zero, the Qwen metadata builder
  retains the previous accepted-token state offset instead of dropping it.
  Only local classification markers change; scheduler counts, prefill sentinels
  and non-speculative metadata remain unchanged. This also protects K1.
- QSA retains FGV score/cache operations and the reviewed shape routing:
  score width N<=512 skips score/TopK; otherwise supported M<=128 query-row shapes use
  the corrected FGV non-TLE per-row path, larger M uses the existing radix
  path, and out-of-scope shapes retain the FG fallback. The FGV TLE selection
  path remains disabled on this compiler.
- N=1 BF16 projections keep FG mv and return a view without the stack copy.
  Contiguous N=4/K=10240 projections with M<=2048 use native aten.mm.dtype
  FP32 output rounded to BF16; other N=4 shapes keep FG mv/stack.
- GR down uses the same native FP32-output primitive for contiguous BF16
  [M,10240] x [10240,320], M1..64 or M1024..2048. BF16 storage still precedes
  division by hc_count. Other shapes keep F.linear. Small-M eager is slower;
  the choice is validated for the graph-dominated serving workload.
- These shape-specific native overloads bypass the slow FG linear path without
  disabling FlagOS globally. An operator-wide blacklist cannot express their
  shape and overload scope. The profile pins allow_tf32=false and
  allow_bf16_reduced_precision_reduction=false before warmup/capture.
- MoE retains the tuned BF16 G1/G2 fallback and assignment semantics.
  For C550 E512/topk10/H2560/local intermediate80, M31..2048 may use
  the pinned MCTlass G1 plan; M1024..2048 may also use its G2 plan.
  Each GEMM consumes an assignment matching its own block-M. Small-M G2,
  unsupported shapes, quantization, LoRA, bias and expert mapping retain
  the validated fallback. Activation, router weighting and BF16 boundaries
  are unchanged.
- Scoped MCTlass projections select measured large-M shapes through
  `mctlass_linear.py`; expert plans live in `mctlass_moe.py`. The library
  owns C ABI execution and unit-scale buffers.
- QSA BF16 [B,3,256] / one KV head / index width2051 uses reviewed
  decode folding and request-aligned prefill packing from FGV. Each layer
  owns graph-stable partial workspaces. Geometry and capture-time allocation
  misses retain the split-KV fallback and gate boundary.
- Sampling uses block Top20/top-p on FP32 [B,248320], B1..64, when CPU
  metadata certifies top_k20. Other requests retain the generic FGV/native
  filter. Native full-vocabulary exponential draws are preserved; only the
  eligible race reduction is fused. Seeded generators and FP64-Gumbel
  requests retain the native route. The positive-noise ContextVar is part
  of the numerical contract, not an experimental counter.

- The PLE gate computes sign in FP32 before returning to BF16. The rest of its
  rounding and operation order remains unchanged.
- Mixed GDN gathering makes the QKV slice contiguous at the required boundary.
- HC keeps the validated materialized schedule and ATen/FG rounding on MetaX;
  the prerequisite's fused/deferred path remains the default elsewhere.

- Host preparation uses batch-vectorized PLE hashing and CPU metadata
  mirrors where valid; pure prefill can reuse CPU input IDs. Shared-table
  barriers, pinned staging and H2D completion ordering remain intact.
- GDN reuses metadata within the same runner step while every cache group
  retains its own state indices and graph buffers. QSA metadata, packing-row
  caching and block-index expansion avoid redundant preparation; packing
  caches end with each metadata-build invocation.
- Eligible MetaX QSA Gemma RMSNorm widths use the existing FGV implementation.

The new choices are model/vendor policies over library primitives. No CPU
affinity or communication change is selected.

## Serving

Use the executable [MetaX entry point](../../examples/qwen3_8_flash_next/serve_metax.py).
It defaults to **MTP off** and prints a complete native vLLM command unless
`--execute` is supplied. It preserves BF16/TP8/PP1, MBT2048, maxseq64,
maxlen69632, memory utilization0.92, shared N-gram, QSA width buckets and the
existing FG/FGV tuning and shape-routing policy.

Select the matching runtime and explicit frozen operator package directories:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next
export OMP_NUM_THREADS=4
export QWEN38_FLAGGEMS_ROOT=/path/to/frozen/flag_gems
export QWEN38_FGV_ROOT=/path/to/frozen/flaggems_vllm
export QWEN38_CACHE_ROOT=/path/to/private/cache
python examples/qwen3_8_flash_next/serve_metax.py --output-dir /tmp/qwen38-off-plan
# An explicitly authorized service launch:
python examples/qwen3_8_flash_next/serve_metax.py --execute
```

The launcher selects this plugin checkout, the supplied operator packages, and
the packaged tuning configuration. OMP_NUM_THREADS is inherited from the
calling environment; set it to 4 as above to reproduce the tested CPU thread
configuration. The validated workers reported torch.get_num_threads()==4.
FG and FGV retain their separate native
caches (`$HOME/.flaggems` and `$HOME/.flagdnn`). The launcher places Triton and
the operator-hit log in mode-specific paths below QWEN38_CACHE_ROOT. Separate
container filesystems or homes must be supplied by the deployment when cache
isolation between services is required.
It does not enable benchmark worker extensions or server development RPCs.
The frozen dependency composition remains an explicit external prerequisite;
the launcher does not install, patch or replace operator source.

MTP remains available through `--mtp k1`, `--mtp k2` or `--mtp k3`.
Off removes the whole speculative configuration; it is not a K=0 proposer.
The supported sequential MTP configuration retains
`disable_padded_drafter_batch=true`, `parallel_drafting=false` and
`use_local_argmax_reduction=false`. Target graph token buckets track the
same request-count buckets:

| Mode | Target token buckets | Maximum |
|---|---|---:|
| off (default) | 1, 2, 4, 8, 16, 32, 64 | 64 |
| k1 | 2, 4, 8, 16, 32, 64, 128 | 128 |
| k2 | 3, 6, 12, 24, 48, 96, 192 | 192 |
| k3 | 4, 8, 16, 32, 64, 128, 256 | 256 |

All modes use target FULL_DECODE_ONLY and retain target eager prefill/mixed
execution. K1's corresponding first draft pass is eager there; later K2/K3
draft steps retain their existing shape-dependent graph path.
K2/K3 are existing explicit options; this configuration change adds no K2/K3
GPU, performance or accuracy acceptance.

### Asynchronous scheduling

For the current best MTP-off configuration, pass `--async-scheduling`.
Omitting the option keeps the synchronous fallback; `--no-async-scheduling`
is explicit. MTP K1/K2/K3 remain synchronous and async/MTP is rejected.

MRV1 supplies actual input tokens on GPU. PLE keeps a double-buffered
consumed-token suffix indexed by request identity and logical position;
new/resumed requests seed from the native restored history. CPU placeholder
tokens cannot seed the suffix. QSA packing remains enabled because logical
lengths are exact without speculation. CPU table lookup still runs before
forward. UVA prefetch is not included.

### Checkpoint sampling defaults

The launcher explicitly uses `--generation-config auto` without
`override_generation_config`. Requests that omit sampling fields inherit the
checkpoint generation configuration through the native vLLM Chat/Completion
parameter conversion. The current checkpoint resolves to temperature1.0,
top_p0.95 and top_k20, with random sampling. Keep the checkpoint file as the
authority; the launcher does not duplicate these values as request overrides.

A normal request uses the tokenizer's default thinking template:

```json
{
  "model": "qwen38",
  "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
  "max_tokens": 256
}
```

There is no forced temperature0, ignore_eos, fixed request seed, or
enable_thinking=false. Explicit client sampling parameters retain native
override behavior. Non-thinking is a separate explicit client choice, not
the default shown here.

The native MTP argmax proposal policy is unchanged; it does not force the
target/rejection sampler to be greedy. The existing
[MetaX top-k/top-p fallback](../../vllm_fl/dispatch/backends/vendor/metax/patches/topk_topp_sampler.py)
retains native RNG and adds the optional FGV deterministic filter described
above. Official random sampling was exercised in the fixed-output validation
below. Full-model accuracy evaluation remains outside the acceptance scope.

The [launch profile](../../examples/qwen3_8_flash_next/metax_tuning/) installs
reviewed configuration before model construction and graph capture. QSA work
uses width buckets512/2048/8192/cache-capacity from the CPU length bound, and
target/MTP graph keys retain the width. At maxlen69632 the QSA capacity is17600
groups. The width buckets and KV-cache layout remain unchanged.

## Validation and historical performance

### Current reviewed stage: Support asynchronous scheduling for Qwen PLE

| Fixed workload | Comparison | Reviewed stage | Observed change |
|---|---:|---:|---:|
| P1024 / exact O1024 / C64 / N128 | 661.6191 tok/s | 939.8054 tok/s | +42.05% |

Both use checkpoint T1/p.95/k20, MTP off, BF16 TP8 and OMP/Torch4.
The numbers belong to the corresponding frozen source snapshots. Cleanup
preserves tensor computations, parameters, fallbacks and defaults, and
validates relocation with source equivalence and CPU contracts; it does not
claim a new end-to-end measurement. Experimental module counters are removed
from production; benchmark observers remain outside the model tree.
Each mode has one formal measurement, not a stability or accuracy claim.
The async comparison is the same-source synchronous control (661.6191); the previous recorded best was661.0880.

The following records are historical milestones with their original scope.


The records below retain their historical configuration and sampling provenance.
Official sampling and default MTP off were introduced in 9fa633c.
The following result measures the later integrated hotspot configuration,
with the private dependencies above, on node126/C550x8:

| Fixed workload | Previous sampling version | Hotspot configuration | Change |
|---|---:|---:|---:|
| P1024 / exact O1024 / C64 / N128 | 221.0897 tok/s | 248.6844 tok/s | +12.48% |

Both use T1.0/top_p0.95/top_k20, MTP off and OMP/Torch threads4. Each version
has one warmup, one formal and one profile round; only the unprofiled formal
round supplies throughput. All 128 requests per round completed with exactly
1024 input and output tokens, no failures or preemption. ignore_eos is a
benchmark-only request setting, not the serving default. This is the observed
combined gain from one formal round per version, not per-operator attribution.
That historical248.6844 version excluded the later sampling/QSA candidates; the current reviewed stage includes them.

Private evidence: report36, hotspot-optimization-20261030/integrated/results/fixed-p1k,
the source manifest above, and FINAL_SOURCE_VERIFICATION_AFTER.json.
GPU component validation and source hashes bind this configuration to those
results; consolidation adds CPU policy/fallback tests and source-equivalence
checks, without claiming a new end-to-end run.

CPU contracts and opt-in device checks live under `tests/qwen3_8_flash_next`,
`tests/unit_tests/models`, `tests/unit_tests/dispatch` and
`tests/models/qwen3_8_flash_next`. They cover host-table partitioning,
accepted-history rollback, wide state addresses, cache rebinding, QSA indices,
MTP metadata, changed-input graphs and launch configurations. The CPU runner
uses isolated framework declarations where documented; it is not a full
accelerator-import check.

```bash
python tests/qwen3_8_flash_next/run_cpu_contracts.py --log-dir /tmp/qwen38-cpu
# Assigned-device checks additionally require QWEN38_RUN_DEVICE_TESTS=1 and
# QWEN38_ENGINE_CONFIG pointing to a local BF16/K1 EngineArgs JSON.
```

The post-consolidation TP8 regression completed eight short answers, P1024 and
P4096 with O64/C64/N128 (one warmup plus one check round), and sequential
16384/65536-token inputs. Both long prompts answered 42 and stopped normally.
On all eight ranks, the short check rounds replayed every pure-decode target and
MTP forward with no eager fallback or new capture; the long prompts first
captured their new MTP width bucket, then replayed it without eager fallback.
These O64 checks validate functionality and are not a new O1024 benchmark.

The following **historical O1024 results precede this consolidation** and belong
to [b201958](https://github.com/cshujie131-bot/vllm-plugin-FL/blob/b20195863bf59ff26a4241fe755750f702e445a9/docs/models/qwen3_8_flash_next.md).
Both variants include the MoE fix; N128/C64, one warmup plus one formal round:

| Input | MoE-fixed baseline output tok/s | Tuned output tok/s | Increase |
|---|---:|---:|---:|
| 1024 | 94.58 | 148.39 | 56.9% |
| 4096 | 55.59 | 90.32 | 62.5% |

P1K moved from node125 to the same C550 hardware on node102; P4K stayed on
node126. The P1K profile's cumulative decode TopK time fell 143.42 to
33.22 ms per rank/step; these are overlapping device durations, not request
wall time. Full historical evidence remains in the linked version.

Uncovered dynamic linear/MV shapes can still invoke runtime autotuning and
extra clears. That optimization and communication work remain separate.
Full-model accuracy and multimodal acceptance are not claimed.


## Additional MetaX serving capabilities

HC small fusions are selected only for the reviewed MetaX BF16 shapes. Grouped normalization, gate/reduce, residual injection and divide/SiLU retain the original intermediate rounding. Other shapes keep their existing paths.

Build the optional native mapping extension with `VLLM_VENDOR=metax python setup.py build_ext --inplace` in the matching MetaX Torch/MACA environment. With MTP off, `--ngram-mode uva_auto` selects broadcast or head-sharded mapped-host lookup; `shared_buffer` remains the default. Registration/table owners live for the service lifetime. GPU hashing preserves the model history semantics.

MTP-off serving accepts `--max-num-batched-tokens 4096` or `8192`; the compatibility default is 2048. Large prefill rows and long score buckets select the validated HC/projection/expert/attention paths. Larger MBT consumes more workspace and may worsen tail latency. Prefill stays eager.

The default MTP-off GDN host-metadata path reuses framework chunk boundaries and supplies a host-derived skew hint to the source-locked FGV interface. `--no-gdn-host-metadata` retains its fallback. Metadata groups keep separate state addresses; numerical GDN kernels and FP32 state dtype are unchanged.

The reviewed MetaX BF16 TP8 expert geometry can reuse activation and final output buffers. Final-output reuse requires the separately source-locked native vLLM `supports_output_buffer_alias` capability; it is default-off in the framework and is not part of an unmodified ee0da84 installation. Other layouts, LoRA and quantized paths retain their existing behavior.

These performance paths require the exact dependency composition in `metax_tuning/operator_dependencies.json`; not all numeric operator components are public revisions yet.

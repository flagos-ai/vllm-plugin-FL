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
| `worker/qwen3_8_flash_next.py`, `spec_decode/` | Committed-token preparation, accepted-state metadata, multi-group proposal and draft graphs |
| `patches/qwen3_8_flash_next.py`, `patches/qwen3_8_flash_next_mtp.py` | Existing registration plus the scoped vLLM 0.24 MTP compatibility |
| `dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/`, `dispatch/config/` | MetaX policy and reviewed launch configuration over existing library kernels |

The prerequisite contains MTP model source but rejects speculative decoding in
its default verifier. This integration retains the existing MetaX K1/K2/K3 paths as explicit options.
Other platforms retain the prerequisite's default execution choices. There is
one common model implementation, with no duplicate flat model or parallel
attention package. Original attribution to [#573](https://github.com/flagos-ai/vllm-plugin-FL/pull/573)
is retained where applicable; that branch's HY4/T-Head changes are not imported.

## Runtime and operator dependencies

This section is the deployment dependency reference. The tuning profile's
manifest records its historical parameter provenance, not a complete current
operator lock.

| Dependency | Fixed reference or required status |
|---|---|
| vLLM 0.24.0 | [ee0da84](https://github.com/vllm-project/vllm/commit/ee0da84ab9e04ac7610e28580af62c365e898389), matching empty-target build |
| FlagGems base | [5.3.5](https://github.com/flagos-ai/FlagGems/tree/v5.3.5), [a7620cc](https://github.com/flagos-ai/FlagGems/commit/a7620cc191a0b42e040194622c5758b22a7a25dc) |
| MetaX index address-product fix | Widen indices before stride multiplication; [FG #7981](https://github.com/flagos-ai/FlagGems/pull/7981), published fix commit [5586276](https://github.com/flagos-ai/FlagGems/commit/55862761bd838e3de0f1ec39f3ba96674efa7d18). Apply only this index file/fix to the pinned FlagGems 5.3.5 base above; production file `flag_gems/runtime/backend/_metax/ops/index.py` SHA256 `e02c30372d8794968ad23a74f5099b65249dcc322ae706f53cb4804b327ed984`. |
| FlagGems-vllm base | [0f3df8e](https://github.com/flagos-ai/FlagGems-vllm/commit/0f3df8ecbff9b48b2f8745c7160e0df8cb5486b5) |
| PLE fix | [FGV #959](https://github.com/flagos-ai/FlagGems-vllm/pull/959), PLE file from [22194f0](https://github.com/flagos-ai/FlagGems-vllm/commit/22194f0b2ee0cebb51f945a20a157b82c4e19c65) |
| MetaX inplace MoE fix | [FGV #1000](https://github.com/flagos-ai/FlagGems-vllm/pull/1000), [e04f51c](https://github.com/flagos-ai/FlagGems-vllm/commit/e04f51c1c7176d7a356838fdfa8b3dc4fbaf6f8a) |
| Per-row TopK boundary fix | [FGV #1024](https://github.com/flagos-ai/FlagGems-vllm/pull/1024), published fixed delta [1a811eb](https://github.com/flagos-ai/FlagGems-vllm/commit/1a811ebc3280e51b63fdbeef9cab89029508731d) |
| MetaX radix TopK | Required FGV candidate retained by the large-M route. Not published; a public fixed reference is pending. Production file `flaggems_vllm/runtime/backend/_metax/ops/radix_topk.py` SHA256 `f77b097c984199b324efc933d5b4756a5ed0b8a30f3cc280246e35818e61710e`. |
| Torch / FlagTree | `2.10.0+metax3.8.1.0` / `0.6.1+metax3.6` |
| MACA SDK / driver | 3.8.1.3 / 3.8.1 |

Keep the matching runtime/ABI combination. Communication uses MCCL and
torch.distributed; FlagCX is not used. The tested FGV composition applies the
listed component fixes to the fixed base and retains the radix candidate; the
references do not identify a single complete public release. Publicly
reproducible installation remains pending the unpublished FGV radix
candidate.

The current private validation bundle's
`three-commit-publication-20261029/runtime-dependencies/SOURCE_MANIFEST.json`
records 3,008 files in the final runtime assembly, including the operator
source fingerprints above. Historical `common/source-manifest.json` belongs to
the earlier greedy diagnostics and records the prior index source; it does not
identify the current assembly. These manifests are reproduction pointers for
private evidence, not portable dependency locks. The launcher accepts explicitly
selected package directories and checks their presence; it does not verify
their revisions, install fixes, or rewrite operator source. Operator changes
remain owned by FlagGems or FlagGems-vllm.

[FGV #982](https://github.com/flagos-ai/FlagGems-vllm/pull/982) is a related
compiler-capability solution; the inplace-MoE dependency above remains #1000.

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
- N=1/4 BF16 projections use existing FG mv/stack. Wide linear operations keep
  their default provider; an operator-wide linear blacklist would be too broad.
- The PLE gate computes sign in FP32 before returning to BF16. The rest of its
  rounding and operation order remains unchanged.
- Mixed GDN gathering makes the QKV slice contiguous at the required boundary.
- HC keeps the validated materialized schedule and ATen/FG rounding on MetaX;
  the prerequisite's fused/deferred path remains the default elsewhere.

These are the existing adaptation boundaries. No new operator exclusion or
communication optimization is introduced by the code consolidation.

## Serving

Use the executable [MetaX entry point](../../examples/qwen3_8_flash_next/serve_metax.py).
It defaults to **MTP off** and prints a complete native vLLM command unless
`--execute` is supplied. It preserves BF16/TP8/PP1, MBT2048, maxseq64,
maxlen69632, memory utilization0.92, shared N-gram, QSA width buckets and the
existing FG/FGV tuning and shape-routing policy.

Select the matching runtime and explicit frozen operator package directories:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next
export QWEN38_FLAGGEMS_ROOT=/path/to/frozen/flag_gems
export QWEN38_FGV_ROOT=/path/to/frozen/flaggems_vllm
export QWEN38_CACHE_ROOT=/path/to/private/cache
python examples/qwen3_8_flash_next/serve_metax.py --output-dir /tmp/qwen38-off-plan
# An explicitly authorized service launch:
python examples/qwen3_8_flash_next/serve_metax.py --execute
```

The launcher selects this plugin checkout, the supplied operator packages, and
the packaged tuning configuration. FG and FGV retain their separate native
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
is retained. Random sampling activates a path that earlier temperature0
benchmarks did not exercise. This change validates CPU parameter plumbing;
GPU sampling behavior, performance and accuracy under the inherited defaults
remain to be validated.

The [launch profile](../../examples/qwen3_8_flash_next/metax_tuning/) installs
reviewed configuration before model construction and graph capture. QSA work
uses width buckets512/2048/8192/cache-capacity from the CPU length bound, and
target/MTP graph keys retain the width. At maxlen69632 the QSA capacity is17600
groups. The launch change does not alter cache layout, operators or model code.

## Validation and historical performance

The records below retain their historical configuration and sampling provenance.
The current default-off/checkpoint-sampling configuration change has CPU
parameter-plumbing validation only; it does not convert earlier greedy results
into random-sampling GPU or accuracy acceptance.

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

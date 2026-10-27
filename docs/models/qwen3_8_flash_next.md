# Qwen3.8-Flash-Next on MetaX

This change **depends on [plugin-FL PR #455](https://github.com/flagos-ai/vllm-plugin-FL/pull/455)
being merged**, using its fixed revision
`72d6440bbdb4649375aad33390e176196a47104a` as the integration base.
It reuses the common Qwen3.8/Qwen4 configuration, model, loader and cache
components and adds the validated MetaX execution contracts.

The MetaX scope remains BF16 text serving on 8 C550 cards, TP8/PP1, MRV1,
FP32 recurrent state, shared CPU N-gram and **MTP K1 enabled**.
Prefill/mixed batches stay eager; target and draft pure-decode forwards replay
graphs. Lookup, metadata preparation, logits and sampling remain outside
those forward graphs. The capacity target is a 64K input, not 64 simultaneous
full-length inputs.

## Increment over #455

| Location | Added or changed responsibility |
|---|---|
| `models/qwen3_8_flash_next/config.py` | Existing configuration plus the input capability and compatibility fields |
| `models/qwen3_8_flash_next/common/` | Reuse HC/PLE/cache owners; preserve MetaX rounding/order, prepared metadata and live cache views |
| `models/qwen3_8_flash_next/gpu/model.py`, `gpu/ple_layer.py`, `gpu/qsa.py`, `gpu/indexer_qsa.py` | Shared forward structure with scoped MetaX execution/provider choices |
| `models/qwen3_8_flash_next/gpu/host_ngram.py`, `host_lookup.py` | CPU table loading, hashing, shared results and bounded pinned staging |
| `models/qwen3_8_flash_next/gpu/mtp.py` | K1 draft model with real 4H feedback, using the shared decoder/loader/HC components |
| `worker/qwen3_8_flash_next.py`, `spec_decode/` | Committed-token preparation, accepted-state metadata, multi-group proposal and draft graphs |
| `patches/qwen3_8_flash_next.py`, `patches/qwen3_8_flash_next_mtp.py` | Existing registration plus the scoped vLLM 0.24 MTP compatibility |
| `dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/`, `dispatch/config/` | MetaX policy and reviewed launch configuration over existing library kernels |

The prerequisite contains MTP model source but rejects speculative decoding in
its default verifier. This integration retains the tested MetaX K1 path.
Other platforms retain the prerequisite's default execution choices. There is
one common model implementation, with no duplicate flat model or parallel
attention package. Original attribution to [#573](https://github.com/flagos-ai/vllm-plugin-FL/pull/573)
is retained where applicable; that branch's HY4/T-Head changes are not imported.

## Runtime and operator dependencies

| Dependency | Fixed MetaX runtime |
|---|---|
| vLLM 0.24.0 | `ee0da84ab9e04ac7610e28580af62c365e898389`, empty-target build |
| FlagGems | 5.3.5 |
| FlagGems-vllm base | `0f3df8ecbff9b48b2f8745c7160e0df8cb5486b5` |
| PLE fix | [FGV #959](https://github.com/flagos-ai/FlagGems-vllm/pull/959), PLE file from `22194f0b2ee0cebb51f945a20a157b82c4e19c65` |
| MetaX inplace MoE fix | [FGV #1000](https://github.com/flagos-ai/FlagGems-vllm/pull/1000), `e04f51c1c7176d7a356838fdfa8b3dc4fbaf6f8a` |
| Torch / FlagTree | `2.10.0+metax3.8.1.0` / `0.6.1+metax3.6` |
| MACA SDK / driver | 3.8.1.3 / 3.8.1 |

Keep the matching runtime/ABI combination. Communication uses MCCL and
torch.distributed; FlagCX is not used. The tested FGV composition is the fixed
base plus the two explicit operator fixes, not an arbitrary moving main branch.
[FGV #982](https://github.com/flagos-ai/FlagGems-vllm/pull/982) is a related
compiler-capability solution; these results use #1000.

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
- QSA uses FGV scores/cache operations and FG top-k; the pinned FGV TLE
  selection path is unsupported by this compiler.
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

Set MODEL_PATH to the shared BF16 checkpoint and retain the matching runtime:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export USE_FLAGGEMS=1 VLLM_FL_USE_FLAGGEMS_VLLM=1 VLLM_FL_PREFER_ENABLED=1
export VLLM_FL_CONFIG="$PWD/vllm_fl/dispatch/config/qwen3_8_flash_next_metax.yaml"
export VLLM_TUNED_CONFIG_FOLDER="$PWD/examples/qwen3_8_flash_next/metax_tuning/moe-configs"
unset FLAGCX_PATH FLAGGEMS_CACHE_DIR FLAGGEMS_DB_URL GEMS_CACHE_PATH

vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 8 --pipeline-parallel-size 1 --dtype bfloat16 \
  --language-model-only --max-model-len 69632 --max-num-seqs 64 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.93 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --mamba-cache-mode none --mamba-ssm-cache-dtype float32 \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"parallel_drafting":false,"use_local_argmax_reduction":false,"disable_padded_drafter_batch":true}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,4,8,16,32,64,128],"max_cudagraph_capture_size":128}' \
  --additional-config '{"qwen38_ngram_mode":"shared_buffer","qwen38_mtp_graph":true,"qwen38_qsa_length_buckets":true,"qwen38_tuning_profile":true}'
```

The [launch profile](../../examples/qwen3_8_flash_next/metax_tuning/) installs
reviewed configuration before model construction and graph capture. It uses
native LibTuner/ConfigCache choices and launch metadata on existing kernels;
third-party sources are not rewritten and no diff artifacts are distributed.
FG and FGV keep separate native caches.

QSA work uses width buckets 512/2048/8192/cache-capacity, selected from the
existing CPU length bound. At maxlen69632, capacity is 17600 groups. This
changes work and graph geometry, not cache capacity or the K512 selection
algorithm. Target and MTP keys include the width; seven target token shapes
yield 28 target graphs. GDN output configuration retains context buckets
without adding an argument to the library kernel.

## Validation and historical performance

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

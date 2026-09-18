# DeepSeek V4 Flash W8A8 on MetaX C550

This opt-in path targets **8 x C550, TP8**, symmetric dynamic-token W8A8,
BF16 activations and FP8 KV cache on vLLM **0.20.2** (`release/0.2`).
It uses FlagGems plus the MACA-compatible Triton/deep_gemm/MCTlass runtime.
The archived build used FlagGems 5.0.2, Torch 2.8.0+metax3.7.0.7,
Triton 3.0.0+metax3.7.0.7 / FlagTree 0.5.1+metax3.0, and
MCTlassEx 0.1.1+metax3.7.0.7torch2.8.
The optimized kernels are included in FL; no private FlagGems overlay,
`sitecustomize`, `vllm_metax` Python import or machine-specific tuning map
is required by this implementation.

## Scope and entry points

Enable `VLLM_FL_DSV4_METAX_OPTIMIZATIONS=1` before starting workers.
Default is off; other models and the upstream INT8 indexer-cache path are unchanged.

| Code under `vllm_fl/ops/deepseek_v4_metax/` | Change |
|---|---|
| `attention.py`, `decode.py` | HQ16 padding, split-top-k, one block-diagonal PV accumulator for the two V halves; reuse QK/KV loads |
| `prefill.py`, `attention_staged.py` | Pairwise real-head packing; materialize chunk softmax state, replay PV halves in the same accumulation order |
| `gather.py` | Preserve padded cache-block strides without a full-cache clone |
| `indexer.py`, `paged_indexer_logits.py`, `tree_topk.py` | Direct paged logits, fused decode rows, valid-length early exits and exact top-512 |
| `prefill_indexer_rowshard.py`, `indexer_m4.py`, `indexer_score.py` | M/2 fallback and M/4 complete-query-row sharding, KV-load hoisting and invalid-tile skipping |
| `mhc.py`, `mhc_mm.py`, `mhc_split.py` | Small-M projection split-K; large-M RMS/coefficients/mix split; unchanged Sinkhorn iterations and casts |
| `mctlass_moe.py` | MCTlass expert GEMMs for both phases; cache kernel-M metadata without query casts or pointer/storage aliases |

M/4 ranks independently score **all historical KV for their assigned Q rows**.
Each row uses the original top-k implementation before gathering indices.
No KV-column partition, approximate selection or cross-rank top-k merge is used.
The accepted groups are [0,1,2,3] and [4,5,6,7]; M/8 was rejected because actual
model inputs differed between those groups and changed selected indices.

The groups passed the archived model tests, but replicated input equality is
not a universal guarantee across builds or tuning configurations.
`VLLM_FL_DSV4_METAX_VERIFY_M4=1` is therefore the default: compare full indices,
ordering and padding against the M/2 reference on every eligible call.
All eight ranks must agree. Set it to `0` **only after validating the new
environment**, before timing. Verification has intentional extra compute,
communication and CPU synchronization.

## Configuration

Use the vendor-matched MACA/PyTorch/Triton build, not a stock NVIDIA torch wheel.
Install the plugin from this branch and FlagGems using the project's normal
MetaX setup. MCTlass and MetaX deep_gemm must be importable. A checkpoint is
supplied separately; no weights or proprietary library binaries are included.

Set the native top-k exclusions before plugin/FlagGems initialization:

```bash
export USE_FLAGGEMS=1
export VLLM_PLUGINS=fl
export VLLM_FL_DSV4_METAX_OPTIMIZATIONS=1
export VLLM_FL_FLAGOS_BLACKLIST=topk,masked_fill,masked_fill_
export VLLM_FL_DSV4_METAX_VERIFY_M4=1

vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 8 --dtype bfloat16 \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 66560 --max-num-seqs 64 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 \
  --disable-custom-all-reduce \
  --attention-config '{"use_fp4_indexer_cache": false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
```

This is an integration example, **not a claim that this rebased PR was
service-tested**. TP/PP/DP/MTP and model dimensions are checked at initialization.
The optimized path requires TP8/PP1/DP1, no EP or MTP, Flash H4096/HQ64/L43.
It does not enable TLE/warp specialization.

**Keep prefill outside full graph replay.** The final source baseline uses
FULL_DECODE_ONLY: an earlier FULL_AND_PIECEWISE deployment replayed dummy
attention for short prefill requests. This is not the same as enforce-eager.

Use separate writable directories for `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `FLAGGEMS_CACHE_DIR` and
`DG_JIT_CACHE_DIR` for each build/experiment. Do not reuse compiler caches
from the previous deployment. C550 has a 64 KiB shared-memory limit.
The old global dense-mm tile patch and generated-kernel/rank tuning maps are
not copied here; compatible FlagGems launch configs still need verification
on the target build.

## Validation and historical results

Current PR preparation is **CPU-only**: contract tests, Python syntax,
relative-import checks, lint and kernel-body comparison against the frozen
source. GPU regression tests are provided, but were not executed during
this port. Re-run TP8 model/graph correctness before treating it as deployable.

Archived evidence from the source deployment:

- Decode blockdiag: standalone and service smoke passed; P1K/O1K/C64 output
  812.59 -> 898.60 tok/s (+10.58%); P4K/O1K/C64 536.53 -> 581.58 (+8.40%).
  Candidate values are the mean of three formal rounds after one warmup;
  the baseline was reused from an earlier date, not interleaved A/B.
- MCTlass/km_cache: P1K/O128/C64 output 438.04 -> 447.17 (+2.08%);
  warmup 1 + formal 2, report run3. The original test shared a writable
  compiler cache and later hit a cache-contamination recovery failure.
  This is a limitation of that experiment, not a recommended cache policy.
- Staged prefill attention: eight ranks x 344 real-call bitwise comparisons
  of output/max/LSE passed; P16K/O16/C64 total 5846.65 -> 7226.63 (+23.60%).
- MHC split: eight ranks x 172 real-call bitwise comparisons passed;
  P16K/O16/C64 total 7226.63 -> 7611.72 (+5.33%).
- M/4: 5,376 real-model rank/call comparisons passed with exact top-k
  indices, ordering and padding. M/8 was rejected, not tolerance-relaxed.
- The final independent source release passed 11 cold-cache smoke cases,
  including the 9-token greeting, short math/copy, 63/64/65/66-token
  boundaries and a near-1K control. This is not a full GSM8K/C-Eval/MMLU score.

The prefill/MHC comparisons reuse historical controls; each candidate used
warmup 1 + formal 2 and reports run3. They are short-output prefill-heavy
results and must not be presented as isolated decode gains.

The combined **earlier** configuration produced the following archived
results (64 requests, concurrency 64, O1024, warmup 1 + formal 3, report run4):

| Prompt tokens | Output tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|
| 1024 | 946.18 | 1892.36 | 4483.62 | 62.99 |
| 4096 | 687.18 | 3435.91 | 17021.88 | 75.20 |
| 16384 | 330.72 | 5622.18 | 68472.94 | 121.02 |
| 65536 | 100.31 | 6520.06 | 294978.15 | 301.15 |

These predate the final independent-runtime/graph-policy change and are
**not benchmarks of this PR**. Gains above are incremental measurements
against different baselines and must not be added together.

## Checks

```bash
# CPU only; deliberately does not import torch/vLLM or touch a GPU.
python3 tests/unit_tests/test_deepseek_v4_metax_contracts.py -v

# Only with an approved isolated GPU window and matching dependencies:
DSV4_RUN_GPU_TESTS=1 pytest -q tests/unit_tests/ops/test_deepseek_v4_metax_kernels.py
```

Before promoting the port: test the short-prompt boundaries, compare actual
M/4 indices on 16K/64K, exercise decode graph capture/replay with changed
inputs, then benchmark with verification disabled and isolated caches.
Logprobs/prompt_logprobs had an unresolved FlagGems log_softmax issue in the
source environment; unused KV slots containing NaNs are another known
robustness limitation. Neither is claimed fixed here.

# PCP MoE ReduceScatter overlap

This branch builds on `e316b4e` (the measured GDN layout optimization).
It changes output ownership around the audited BF16 Triton MoE path so that
ReduceScatter of one chunk can run concurrently with the next chunk's experts.
The default local chunk size stays at 8192 rows.

## Output ownership and stream ordering

Native MoE output shares storage with expert workspace. The previous pipeline
therefore waited for RS(i) before calling experts(i+1). Removing that wait alone
is incorrect: the next call can overwrite data still being read by NCCL.

`try_prefill_moe(..., out=destination)` now writes the fused top-8 sum, shared
gate and shared add directly into the supplied contiguous output. Its scoped
finalize hook returns that destination through the native runner/custom op.
The native routing, expert kernels and BF16 rounding order are unchanged.
If sum fusion cannot handle a layout, the native sum plus gate/add still writes
and returns the supplied destination. Calls without `out` retain their existing
borrowed-output contract.

`_overlap_pr_moe` alternates two independently allocated partial buffers:

- Compute waits for AG(i), then computes chunk i into buffer i modulo 2.
- The communication stream prefetches AG(i+1), then waits for compute(i) and
  runs RS(i), directly into the appropriate final local output slice.
- Compute(i+1) does not wait for RS(i). Before reusing a buffer at i+2,
  compute waits for RS(i).
- All allocations occur on the compute stream before communication starts.
  A final stream join orders downstream reads and allocator reuse, including
  exceptional exits. The buffers remain referenced through this join.

All gathered chunk shapes are checked for eligibility before any collective
launches. A small tail that switches native shared experts to their auxiliary
stream now falls back to the general PCP path before entering the pipeline.

For 64K total tokens, PCP4 and hidden width 2048, each partial buffer is
32768 × 2048 BF16 = 128 MiB per rank; two buffers total 256 MiB per rank.
The old temporary RS results and their 80 per-forward copy kernels are removed.
The buffers are per layer invocation, not multiplied by the 40 layers.

The last RS still has to finish before the next layer. Overlap also shares SM
and memory bandwidth with computation, so summed RS time is not a prediction
of the end-to-end saving.

## Validation on 2026-10-13

Artifacts: [`../validation/qwen36_pcp_moe_rs_20261013/REPORT.md`](../validation/qwen36_pcp_moe_rs_20261013/REPORT.md).

- All 31 focused MoE tests pass. Delayed-read tests cover 2–5 chunks,
  noncontiguous local input, uneven tails, buffer reuse and a borrowed-workspace
  negative control that reproduces corruption. Other checks exercise exact
  arithmetic, fusion fallback, output validation and scoped hook restoration.
- 13 real Qwen3.6 scenarios compare against the saved GDN baseline: 64K,
  8K×8, uneven chunks, three chunks, short/ragged/mixed inputs, mixed decode,
  chunked prefill, 65535 tokens, tiny-tail fallback and delayed real NCCL reads.
- 1120 same-input MoE partial comparisons and 844 complete-logit comparisons
  are bitwise equal. All 290 generated tokens per variant and saved top-5
  logprobs match.
- 40 delayed real-NCCL RS comparisons pass across baseline/candidate, four
  ranks and 2/3 chunks. CUDA events confirm candidate chunk 1 completes while
  RS(0) is still pending. These intentionally delayed timings are not used as
  performance estimates.
- Full CP suite: 135 passed, 3 inherited failures in
  `test_moe_preserves_local_tokens_and_single_reduction[True-...]`.
  Those failures are in the already-reduced path, outside this optimization's
  eligible configuration, and were reproduced on the earlier baseline.

## Matched performance

Same GPUs 0–3, H100 80GB, `/data/model`, BF16/eager/TP4/FA3/Triton MoE,
65536-token budget, 66560 model length, 64 maximum sequences, 4 GiB KV cache per
GPU and no prefix cache. ABBA order with reversed case order in the second
round; each segment has 3 warmups and 10 formal samples per shape, giving
20 formal samples per version/shape. All samples are retained.

| Prompt shape | GDN baseline | With MoE RS overlap | Saving | Reduction |
|---|---:|---:|---:|---:|
| 65536 × 1 | 686.772 ms | 683.035 ms | 3.737 ms | 0.54% |
| 8192 × 8 | 505.820 ms | 502.160 ms | 3.660 ms | 0.72% |

These are unprofiled `LLM.generate` wall times for full prefill plus one real
output token per request. Both rounds improve, but the size of the improvement
varies: 64K round medians are 687.706 → 679.580 and 686.538 → 684.678 ms.
This is a small observed gain, not evidence for a large or universal speedup.

Path checks and PyTorch peak-memory checks run separately after timing.
For 64K the maximum observed allocated peak changes from 27482.7 to
27491.3 MiB per rank; other operations determine the full-forward peak.
Maximum reserved memory changes from 39780 to 29860 MiB in this campaign,
consistent with moving gather allocation to the caller's stream; allocator
reservation is not a general capacity guarantee.

Using the historical native TP time of 889.328 ms would give an estimated
23.20% latency reduction, not 25%. Native TP was not rerun in this campaign.

GPU work is guarded by the shared `.gpu.lock` and an all-GPU process, memory
and utilization check before each phase. GPUs were free before launch and
released after validation. No other user's processes were terminated.

This does not change mixed prefill/decode TP fallback or introduce a new
continuous-batching scheduler. Actual FlagCX deployment, other hardware,
quantization and other models have not been validated here.

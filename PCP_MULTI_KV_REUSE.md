# Reuse complete KV for multi-request PCP prefill

This branch extends `6a27bfb`, preserving the GDN layout optimization and MoE
ReduceScatter overlap. It removes the second KV all-gather for complete,
page-aligned multi-request prefill batches. The single-request fast path is
preserved.

## What changes

The first sequence all-gather already contains every K/V head for every input
token. Previously, only single-request prefill reused that tensor directly;
multiple requests wrote native KV cache and then gathered it again by head.

`ZigZagPlan.full_prefill_lengths` now records all request lengths when every
request has query length equal to KV sequence length. `PagedKVPlan` builds a
virtual table with a separate contiguous range of pages for each request.
The table cache includes the full tuple of request lengths, so requests with
the same total length but different boundaries do not reuse an incorrect table.

For example, request lengths 16, 48 and 32 at block size 16 have virtual pages
`[0]`, `[1, 2, 3]` and `[4, 5]`. Table rows pad to the longest row, and the
virtual query segment's sequence length bounds attention reads. The existing
zigzag metadata still selects the correct request table and applies causal
alignment to each segment.

The gathered full-head K/V tensors are passed directly to FA3. Each rank's
native head view still writes to the scheduler's actual KV slots. Decode and
KV transfer therefore continue to see native cache data.

All of these conditions must hold:

- Every request has no prior KV history and a positive, page-aligned length.
- Token rows are distributed uniformly across PCP ranks.
- K, V and native cache are BF16, and the attention backend is FA3.

Alignment of the batch's total length alone is insufficient. A history-bearing
request, partial page, nonuniform row layout, another dtype or FA2 retains the
previous cache-gather path. Mixed prefill/decode keeps the existing native TP
fallback. Prefix caching remains unsupported by the overall PCP configuration.

No GDN, MoE, attention arithmetic, chunk size or scheduling behavior changes.

## Validation on 2026-10-13

Artifacts: [validation report](../validation/qwen36_pcp_multi_kv_20261013/REPORT.md).

- 39 KV/layout unit tests pass. Tests cover separate request page ranges,
  unequal request lengths, different boundaries with the same total length,
  total-only alignment, history, dtype/backend fallback, nonmonotonic physical
  pages and native KV head counts below/equal to/above the TP world size.
- Full CP suite: 161 passed, 3 inherited failures in the already-reduced MoE
  path (`test_moe_preserves_local_tokens_and_single_reduction[True-...]`).
  These were already present before this branch; no new failures.
- 12 real Qwen3.6 scenarios use distinct prompts within each request batch:
  64K×1, 8K×8, 4K×16, 2K×32, unequal aligned lengths, reordered lengths,
  total-only alignment, ragged lengths, mixed decode, chunked prefill,
  mixed history/new requests, and 65535 tokens.
- 320 comparisons of reused K/V, virtual page tables and FA3 output against
  reconstruction from the native cache are bitwise equal.
- 872 complete-logit comparisons are bitwise equal. All 728 generated tokens
  per variant and saved top-5 logprobs match.
- Post-timing checks confirm that all 10 Full Attention layers use one KV
  all-gather per multi-request batch, down from two per layer on the baseline.
  Both versions use one per layer for 64K×1. GDN/MoE optimized paths remain
  active in both versions.

## Performance protocol and scope

The baseline is the completed second optimization, `6a27bfb`. All measurements
use the same GPUs 0–3, four H100 80GB GPUs, `/data/model`, BF16/eager/TP4/FA3,
Triton MoE, 65536 maximum batched tokens, 66560 model length, 64 sequences,
4 GiB KV cache per GPU, no prefix cache and seed 42.

The order is baseline → candidate → candidate → baseline, with shape order
reversed in the second round. Each segment/shape has 3 warmups and 10 formal
samples, totaling 20 formal samples per version/shape. No samples are removed.
Path instrumentation and PyTorch peak-memory checks run after all timed work.

The metric is unprofiled `LLM.generate` wall time for complete prefill plus one
real token per request. Each shape totals 65536 input tokens. This is not HTTP
TTFT, continuous mixed-serving throughput or multiple requests of 64K each.
Since 64K×1 already used KV reuse, its timing difference is a control rather
than evidence of a newly removed collective.

Measured medians for the two-round campaign:

| Shape | First two optimizations | With multi-request KV reuse | Saving | Latency reduction |
|---|---:|---:|---:|---:|
| 65536 × 1 | 679.769 ms | 680.503 ms | -0.734 ms | -0.11% |
| 8192 × 8 | 498.904 ms | 488.224 ms | 10.679 ms | 2.14% |
| 4096 × 16 | 489.892 ms | 478.403 ms | 11.489 ms | 2.35% |
| 2048 × 32 | 490.423 ms | 477.190 ms | 13.233 ms | 2.70% |

All three multi-request shapes improve in both rounds. The 64K×1 direction
changes between rounds and its pooled difference is small; no single-request
improvement is established. Full distributions and allocator observations are
in the validation report, with all 160 formal samples in `samples.csv`.

Before each GPU phase the scripts acquire the shared `.gpu.lock` and check
all GPUs for processes, memory use and utilization. Busy cards prevent launch.
The tests do not modify installed vLLM or terminate other users' processes.

Actual P/D serving, FlagCX deployment, quantization, other hardware/models and
non-page-aligned multi-request reuse have not been tested or added here.

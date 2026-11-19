# Reuse full K/V during PCP FA3 prefill

Branch `prstyle-pcp-kv-reuse` extends the preserved FA3 experiment `caa0fc5`.
For one complete, page-aligned request with uniformly distributed rows and
BF16 K/V/cache, the first sequence all-gather restores every KV head. Native
head views still update each rank's KV cache, preserving subsequent decode and
KV transfer data. Attention reads the restored full K/V through contiguous
virtual pages, eliminating the second cache/head all-gather and deduplication.

The fast path requires FA3. Prefix-bearing chunks, multiple requests, unequal
row counts, incomplete pages and other cache dtypes retain the previous path.
The existing configuration restrictions, including unsupported prefix caching,
remain in force. The new virtual-page cache is bounded to eight lengths and
does not depend on physical block IDs; the native cache update still uses the
scheduler's actual slot mapping. No attention arithmetic or native TP/decode
implementation changes.

Validation on Qwen3.6, 4 H100s, BF16, eager, FA3, Triton MoE, 65536 input and
128 output tokens at concurrency 1: two warmups plus seven unprofiled samples
per version. Median TTFT decreased from 725.637 to 715.145 ms (10.493 ms, 1.45%).
All 128-token outputs matched. Six short/ragged/mixed/chunked probes matched
tokens and logprobs exactly. A separate audit reconstructed K/V through the
native cache on all four ranks: 40 layer checks had bitwise-identical K, V and
FA3 attention output. 81 CP unit tests passed, including native cache writes
from strided full-head views to non-monotonic physical pages.

A new warmed prefill trace confirms 83 rather than 93 all-gathers on every
rank, zero `pages.gather` calls, and unchanged counts of attention/GDN/expert
kernels. GPU0 span is 671.730 ms versus 682.727 ms in the preserved trace;
attention core time remains 111.881 versus 111.843 ms. Trace device time is
diagnostic and is distinct from the unprofiled HTTP TTFT measurements.

Results, raw requests, source snapshots and trace verification:
`/vllm-workspace/validation/qwen36_pcp_kv_reuse_64k_20260930/`.
These checks do not add broad FA3 support or extend the existing supported
configurations. The optimization has not been separately benchmarked at 128K
or in PD serving.

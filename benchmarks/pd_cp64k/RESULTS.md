# PD 64K / N256 results — 2026-10-19

The implementation and all performance/quality results on this page target
**Qwen3.6-35B-A3B**.

Mean TTFT fell **30.27%** with a matched P token budget of
**65535** (D budget 65536), meeting the 25% target. This is real **4P + 4D**
on eight H100 80GB GPUs: native TP4 → TP4 versus optimized CP4 → TP4.
Each mode ran two repeats of 256 requests, C64, exact 65536 input / 1024 output
tokens using the unchanged official `vllm bench serve` client.

| Metric | Native TP | Optimized CP | Benefit |
|---|---:|---:|---:|
| Mean TTFT (s) | 27.895 | 19.452 | +30.27% |
| Output tokens/s | 571.546 | 618.623 | +8.24% |
| Mean E2E (s) | 106.781 | 98.455 | +7.80% |
| P99 TTFT (s) | 64.344 | 48.647 | +24.40% |
| P99 E2E (s) | 143.273 | 127.217 | +11.21% |
| P99 ITL (ms) | 117.779 | 132.433 | -12.44% |

Benefits use `1 - CP/TP` for latency and `CP/TP - 1` for throughput. Each cell
averages two runs; P99 cells average per-run percentiles. The ITL tail tradeoff
must be considered separately from the mean-TTFT acceptance criterion.

P budget 65535 mean TTFT: TP 27.895 s,
CP 19.452 s.
P budget 65536 mean TTFT: TP 27.893 s,
CP 24.051 s
(13.78% reduction).
Use the stated budget when reproducing the recommendation. Two repeats do not
establish long-term or cross-workload stability. The earlier single-run result
with the old router was 24.50%, below target; it is not combined with these runs.

All eight final runs completed 256/256 with zero preemptions, exact token counts,
and 256 matching KV/SSM transfers per rank. Optimized P had 100% CP token
coverage and zero fallback/decode tokens; D stayed on native TP. Both modes
used the identical repaired router. Native TP code stayed fixed at `ce22f5c`.

The existing CP configuration also enables fused Gemma norm and Triton GDN;
TP/D retain native norm and FlashInfer GDN. This measures that full configuration.
188 CPU/GPU tests and 5 router lifecycle tests passed. Real same-input audits
were exact for attention and GDN output norm; MoE maximum relative RMS was
0.00035198 and maximum absolute difference 0.046875. General model quality and
whole-model bitwise equivalence were not established.

See [README](README.md) for commands and infrastructure requirements. Full
per-run data, source hashes, failed-attempt exclusions and the detailed report
are retained at `/vllm-workspace/validation/pd_cp64k_n256_20261019/RESULTS.md`.

## PD 1K–64K with request-aligned CP prefill budgets — 2026-11-16

Qwen3.6-35B-A3B BF16, disaggregated prefill/decode, 4P + 4D H100,
1,024 output tokens, client concurrency 64, 256 measured requests per case,
64 separate warmup requests, and the default MoE configuration. Both stages
use TP4, FA3 and 44 GiB KV cache per GPU. P runs eagerly; D uses FULL CUDA
graphs for decode. Each cell represents one measured run.

Enable CP on P with `VLLM_PLUGINS=fl_qwen_cp VLLM_FL_QWEN_CP=1` and
`--additional-config '{"gdn_prefill_backend":"flashinfer"}'`. The measured
configuration also sets `VLLM_FL_GDN_SP=0` to retain GDN's TP-head path.
D loads `VLLM_PLUGINS=fl_qwen_cp` with `VLLM_FL_QWEN_CP=0`.

For prompt length `L`, P handles `L - 1` tokens and D replays the last token.
CP uses `floor(65535 / (L - 1)) * (L - 1)` as P's
`--max-num-batched-tokens`; TP keeps 65535 for every length. The CP budgets
for 1K / 4K / 16K / 32K / 64K are 65472 / 65520 / 65532 / 65534 / 65535.
D's budget is 65536, and both stages use `--max-num-seqs 64`.
This comparison includes the CP-specific budget adjustment.

| Length | TP TTFT (ms) | CP TTFT (ms) | Δ TTFT | TP duration (s) | CP duration (s) | Δ duration | TP total tok/s | CP total tok/s | Δ throughput | TPOT TP / CP (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 583 | 497 | -14.82% | 41.97 | 41.40 | -1.36% | 12492 | 12664 | 1.38% | 9.5 / 9.5 |
| 4K | 1112 | 899 | -19.09% | 47.19 | 46.27 | -1.96% | 27775 | 28329 | 1.99% | 10.1 / 10.1 |
| 16K | 2745 | 1762 | -35.81% | 65.53 | 61.85 | -5.62% | 68005 | 72058 | 5.96% | 12.3 / 12.5 |
| 32K | 15982 | 4153 | -74.02% | 112.84 | 84.85 | -24.80% | 76667 | 101957 | 32.99% | 9.3 / 15.4 |
| 64K | 42878 | 26570 | -38.03% | 226.09 | 163.70 | -27.59% | 75365 | 104086 | 38.11% | 8.2 / 11.1 |

TTFT and TPOT are means. Duration is the complete benchmark run's wall time;
total throughput counts input plus output tokens. Deltas are `(CP / TP - 1)`
and use unrounded measurements. All ten runs completed 256/256 requests.
At 32K and 64K, lower TTFT and higher total throughput accompany higher TPOT.

The raw results, launch configurations, validation records and source hashes
were checked against the original local November 16 campaign artifacts.
The November campaign uses its recorded support scripts and launch overrides;
the October commands above do not reproduce its D graph configuration.

**Known limitation:** aligning the budget to `L - 1` applies to fixed-length
requests. Variable-length traffic needs additional scheduling logic to obtain
the same alignment; these results do not establish its performance.

### GPQA Diamond — 2026-11-19

Two runs each scored **86.87% (172/198)**, with temperature 1.0, top_p 0.95,
top_k 20, presence_penalty 1.5 and max_tokens 81920. Thinking was enabled;
both rounds used seed 42. The published model reference is 86.0%.
This is a reference comparison, not a matched evaluation of the official
serving stack.

The two evaluation reports are retained under
`/vllm-workspace/validation/pd_cp_gpqa_rerun_20261119_gWuL/`, in
`gpqa_official_round1` and `gpqa_official_round2`.

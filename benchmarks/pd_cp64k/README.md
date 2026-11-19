# PD 64K / N256 validation

This implementation and its reported performance/quality validation target
**Qwen3.6-35B-A3B**.

Target: real FlagCX **4P + 4D**, 65,536 input tokens, 1,024 output tokens,
256 requests, client concurrency 64. Prefer a 25% reduction in mean TTFT;
also report throughput, mean E2E and latency tails. A throughput increase is
`candidate / TP - 1`; a latency reduction is `1 - candidate / TP`.

See [measured results and tradeoffs](RESULTS.md) for the October 19 validation
and the November 16 request-aligned 1K–64K comparison. The configuration and
commands below describe the October campaign; the November configuration is
documented separately at the end of this page.

P uses CP4 in the candidate and native TP4 in the baseline. D always uses
native TP4. Both use FA3, BF16, eager execution and matching scheduler limits.
By default vLLM profiles memory at `--gpu-memory-utilization 0.90` and allocates
KV from the remaining budget; actual P/D capacities are recorded separately.
`--kv-gib` optionally fixes KV memory per rank and overrides automatic sizing.
The hybrid connector computes 65,535 prompt tokens on P
and replays the final token on D. All requests go through the PD router.

```bash
python3 benchmarks/pd_cp64k/run.py --mode tp --source /path/to/baseline --out /path/to/tp --batch-tokens 65535 --kv-gib 44 --repeats 2
python3 benchmarks/pd_cp64k/run.py --mode cp --out /path/to/new_cp --batch-tokens 65535 --kv-gib 44 --repeats 2
```

The recommended commands align the P budget to the hybrid connector's
65,535-token prefill in **both** modes; D remains at 65,536. The driver's
default P and D budgets are 65,536; omit `--batch-tokens 65535` from both
commands to reproduce that control. Use a fresh output directory for each
deployment. The baseline reference for this change is `ce22f5c`.

The commands above retain the historical 44 GiB KV budget for reproduction.
Omit `--kv-gib 44` from both commands to use automatic memory sizing. This
compares the same utilization policy rather than identical resolved KV
capacity: different prefill implementations can have different activation
peaks. Removing the fixed budget does not guarantee 64 resident 128K requests;
model weights, activations and workspaces also consume GPU memory.

The driver uses the installed, unchanged official `vllm bench serve` client.
It depends on the audited validation support directory supplied by
`--support-dir` (default: the October 16 campaign). That directory provides
launch templates, worker selectors, transfer inspection and client dependencies.
The candidate uses fused Gemma norm and the Triton GDN prefill backend, as did
the existing CP configuration. The baseline and D retain their native norm and
FlashInfer GDN dispatch. This measures the complete existing CP configuration
plus the changes in this branch, not an isolated CP attention algorithm.

Each deployment warms up with 64 requests containing 65,536 input tokens and
two output tokens before measurement; these are excluded from the results.
Raw results, exact commands, source/dependency hashes, rank identities, memory,
preemptions and successful matching KV/SSM transfer IDs are retained. New CP
counters report both calls and tokens; one CP call is not evidence that all
prefill tokens used CP. Startup failures and smoke tests must not be included
in performance comparisons.

TP and CP must use the same router. `--router-source` defaults to the router
in this checkout, independently of `--source`, so an immutable TP reference
also gets the common router repair. The driver records its hash and checks
that it stays unchanged during the deployment. The repaired router propagates
P errors and cancels the peer stream instead of leaving D waiting for missing
KV. It disables P HTTP keep-alive reuse; D keeps its existing connection limits.

All measured deployments use the same `router_trace.py` harness. It logs P
request lifecycles and exposes readiness/pending-request diagnostics. The driver
waits for actual router readiness and captures pending transfers when both
engines are idle while D still waits for requests. Native TP and connector
implementations remain fixed.

Compare validated deployments with:

```bash
python3 benchmarks/pd_cp64k/compare.py /path/to/tp /path/to/new_cp
```

For an untimed same-input layer audit, use a small run:

```bash
python3 benchmarks/pd_cp64k/run.py --mode cp --out /path/to/audit \
  --num-prompts 4 --concurrency 4 --output-tokens 16 --warmup-requests 4 \
  --worker-extension audit_worker.PDAuditWorker --audit-reference /path/to/baseline
```

The audit compares actual MoE and attention outputs against the baseline
implementation and checks GDN output norm against the original sequence order.
It does not execute GDN's stateful convolution/recurrence twice. All audit
hooks are removed before the timed client runs. These checks cover layout and
numerical correctness, not general model quality.

## Changes

* Admit short chunks in a substantial prefill batch on a producer-only PD
  engine. Standalone, consumer and `kv_both` retain the previous admission rule;
  all-single-token batches use native TP.
* Keep ragged MoE inputs in collective order. Its stateless padded rows are
  discarded after reduce-scatter; no padding enters attention or GDN state.
* Emit GDN's token-wise output norm/projection directly in padded collective
  order, preserving native ordered state updates.
* Reuse gathered full K/V for complete, non-page-aligned FA3 prefills. Allocate
  separate virtual page ranges per request and mask padding with true lengths;
  native KV writes remain available for transfer and decode.
* Report CP token coverage and TP fallback tokens explicitly.

## Request-aligned 1K–64K configuration

The November results use Qwen3.6-35B-A3B BF16 on 4P + 4D H100 with TP4
on both stages, FA3, 44 GiB KV cache per GPU, 1,024 output tokens, concurrency
64, and 256 measured requests after 64 warmup requests.

* P: `VLLM_PLUGINS=fl_qwen_cp VLLM_FL_QWEN_CP=1 VLLM_FL_GDN_SP=0`,
  `--tensor-parallel-size 4 --enforce-eager`,
  `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":3}'`,
  and `--additional-config '{"gdn_prefill_backend":"flashinfer"}'`.
* D: `VLLM_PLUGINS=fl_qwen_cp VLLM_FL_QWEN_CP=0`, TP4, FA3, and FULL
  CUDA graphs for decode. Keep P's eager setting separate from D's configuration.
* Both stages: `--no-enable-prefix-caching --max-num-seqs 64` and
  `--kv-cache-memory-bytes 47244640256`. Keep the default MoE configuration.
  Use the FlagCX producer/consumer connector and the PD router.

Set P's CP token budget to `floor(65535 / (L - 1)) * (L - 1)`, where `L` is
the prompt length. The hybrid connector runs the last prompt token on D.

| Prompt length | CP P budget | TP P budget | D budget |
|---|---:|---:|---:|
| 1K | 65472 | 65535 | 65536 |
| 4K | 65520 | 65535 | 65536 |
| 16K | 65532 | 65535 | 65536 |
| 32K | 65534 | 65535 | 65536 |
| 64K | 65535 | 65535 | 65536 |

`VLLM_FL_GDN_SP=0` is required to reproduce this table: the tree also contains
a GDN sequence-parallel path that is enabled by default. The recorded campaign
uses external support scripts and launch overrides for D graph execution.
`run.py` and `compare.py` above retain the historical eager, matched-budget
protocol; its strict budget check intentionally rejects unequal-budget pairs.

**Known limitation:** this budget alignment applies to fixed-length requests.
Variable-length requests need additional scheduling logic for equivalent
alignment. See [RESULTS.md](RESULTS.md) for all five lengths, the TTFT/TPOT
tradeoff, and the two GPQA Diamond evaluations.

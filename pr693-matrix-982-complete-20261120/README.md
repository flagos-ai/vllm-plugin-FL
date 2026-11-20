# MetaX PR #693: merged-head two-host matrix complete

This delta records source `982739fd5d68ff19955c7ff975756960abe1cce7` only. All four official Qwen3.6-27B / Qwen3.6-35B-A3B eager and graph groups completed on two physical nodes, TP4/PP1: **104/104 requests and 1040/1040 checks passed**. All four groups exited naturally on both nodes within the original worker shutdown grace, with no forced worker termination or GPU remnants. Server tracebacks and resource-tracker leak warnings were absent.

| Case | Original requests/checks | Cleanup |
|---|---|---|
| 35B graph | 26/26;260/260 | Both nodes clean |
| 35B eager | 26/26;260/260 | Both nodes clean |
| 27B graph | 26/26;260/260 | Both nodes clean |
| 27B eager | 26/26;260/260 | Both nodes clean |

[Matrix](matrix.json) aggregates the four results. Each case has a receipt, exact canonical manifest comparison and per-rank lifecycle witness. NODE_A and NODE_B preserve the prior snapshot's anonymous node identities: ranks0/1 and ranks2/3 respectively. The matrix cleanup list has that same NODE_A/NODE_B order. Graph ranks each completed exactly one explicit native communicator Destroy. Eager runs had no graph capture and used normal MCCL cleanup; they did not call that graph-only close path.

The unchanged protocol is official8192 context/.90 memory, platform-default PIECEWISE/9 capture sizes, original prompts, token budgets, checkers, thresholds, SDK300s/2 retries and60s shutdown grace. Native Runner V2 remained the default; there was no forced V2 environment override or operator whitelist. Actual canonical manifest hashes and source hashes are preserved in the receipts/comparisons.

**Remaining acceptance:** the required single-physical-node27B graph matching regression with an independent initially empty cache, the original two text scenarios9/90 and60s shutdown grace is running and remains unqualified in this snapshot. This cached two-host matrix does not substitute for that cold-cache regression. [Complete CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/44678172822) is also unfinished; its separate device-preflight timeout still needs a successful retry. No merge-readiness claim is made here.

Immutable prior evidence retains [source / merge-tree and174 CPU contracts](https://github.com/physics31415926/vllm-plugin-FL/blob/e005ef29bb7e6e6a7bccd344dac13b3831773119/pr693-lifecycle-982-20261120/source/provenance.json), [checkpoint identities and all41 shard hashes](https://github.com/physics31415926/vllm-plugin-FL/blob/e005ef29bb7e6e6a7bccd344dac13b3831773119/pr693-lifecycle-982-20261120/model-identity.json), and the [official protocol and source freeze](https://github.com/physics31415926/vllm-plugin-FL/blob/e005ef29bb7e6e6a7bccd344dac13b3831773119/pr693-lifecycle-982-20261120/protocol/canonical-freeze-982.json). Its [initial merged-head35B graph evidence](https://github.com/physics31415926/vllm-plugin-FL/blob/e005ef29bb7e6e6a7bccd344dac13b3831773119/pr693-lifecycle-982-20261120/head982/official-982-35b-graph-receipt.json) is one of the four groups completed here. The same [prior snapshot](https://github.com/physics31415926/vllm-plugin-FL/tree/e005ef29bb7e6e6a7bccd344dac13b3831773119/pr693-lifecycle-982-20261120) also includes the R3 A/B and native stacks linked from [FlagCX#730](https://github.com/flagos-ai/FlagCX/issues/730).

[MANIFEST](MANIFEST.json) records original and sanitized SHA256 for each exported document. Embedded original config/manifest hashes remain original provenance; sanitized file hashes are authoritative for this delta. Raw archives, private IP/SSH/proxy details, GPU UUIDs and unrelated workloads are excluded. `SHA256SUMS` covers every package file except itself.

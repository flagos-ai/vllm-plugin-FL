# Coverage Deviation Analysis — Baselines @ 4856534d

Module-level coverage from the core suites (unit + functional scopes),
per platform, from the frozen revision `4856534d`. CPU-env platforms were
measured on the real-vllm WSL2 environment (vllm 0.32.0, torch
2.9.1+cpu); the thead row was measured on real zw810e PPU hardware with
the CI image and env.

## kunlunxin (p800, CPU measurement env)

| Module | Statements | Covered | Missing | Coverage | Supplementary focus |
|---|---:|---:|---:|---:|---|
| Worker | 3969 | 292 | 3677 | 7.4% | model_runner helpers, kl-budget |
| Dispatch (core) | 5793 | 2931 | 2862 | 50.6% | policy fallbacks |
| vendor.kunlunxin | 2131 | 495 | 1636 | 23.2% | backend class, register_ops, impls, patches |
| Attention | 106 | 38 | 68 | 35.8% | metadata builder |
| KV Cache | 179 | 150 | 29 | 83.8% | — |
| Distributed | 1426 | 271 | 1155 | 19.0% | flagcx loader |
| Ops | 840 | 160 | 680 | 19.0% | fused_moe impls |
| Quantization | 702 | 113 | 589 | 16.1% | — |
| Compilation | 343 | 321 | 22 | 93.6% | — |
| Platform Config | 796 | 424 | 372 | 53.3% | vendor branches |

## hygon (bw1000, CPU measurement env)

| Module | Statements | Covered | Missing | Coverage | Supplementary focus |
|---|---:|---:|---:|---:|---|
| Worker | 3969 | 299 | 3670 | 7.5% | determine_available_memory headroom |
| Dispatch (core) | 3749 | 2614 | 1135 | 69.7% | — |
| vendor.hygon | 87 | 66 | 21 | 75.9% | worker.py headroom fn (0% — the 66 is patch.py) |
| Attention | 106 | 38 | 68 | 35.8% | — |
| Distributed | 1426 | 267 | 1159 | 18.7% | — |
| Ops | 840 | 160 | 680 | 19.0% | _get_priority_backends hygon arm |
| Quantization | 702 | 91 | 611 | 13.0% | — |

## iluvatar (bi_v150, CPU measurement env — final numbers from v5)

(Measured in the same environment; table recorded from the v5 run.)

## thead (zw810e, REAL PPU hardware)

| Module | Statements | Covered | Missing | Coverage | Supplementary focus |
|---|---:|---:|---:|---:|---|
| Worker | 3969 | 299 | 3670 | 7.5% | v2_runner_compat install |
| Dispatch (core) | 3852 | 2407 | 1445 | 62.5% | — |
| vendor.thead | 190 | 0 | 190 | 0.0% | entire vendor surface (backend, register_ops, patches) |
| Attention | 106 | 33 | 73 | 31.1% | — |
| Ops | 840 | 126 | 714 | 15.0% | cache.py validation ladder |
| Distributed | 1426 | 210 | 1216 | 14.7% | flagcx arm |

## Deviation classification

1. **Vendor-exclusive code uncovered** — vendor.thead is 0% on real
   hardware; vendor.kunlunxin at 23% (impls + patch internals missing);
   hygon's `worker.py` headroom function has zero references in any test.
2. **Shared-code vendor branches** — platform.py vendor conditionals,
   `fused_moe_utils._get_priority_backends` vendor arm, worker
   `determine_available_memory` vendor headroom integration.
3. **Worker surface** — ~7.4-7.5% across all platforms (model_runner is
   import-gated behind real devices on the CPU env; the real-hw thead run
   shows the same low number, meaning core tests don't exercise the worker
   integration paths either).
4. **Distributed/ops/quantization** — wide but shared across platforms;
   the supplementary suite targets the vendor-keyed entry points
   (flagcx loader branches, fused_moe vendor impls) rather than
   duplicating the core suites' shared coverage.

## GPU-only gaps (Hardware Validation Pending on CPU platforms)

Real-kernel numerics (xtorch_ops on P800, flash_attn_3 on PPU, flaggems
kernels), graph capture/replay, and multi-rank communication. These are
marked `gpu`/`multi_gpu` + `supplementary` and run on the CI runners via
the workflow; the thead legs can be validated on the provided real-hw
host.

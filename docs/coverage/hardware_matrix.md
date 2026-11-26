# Hardware Validation Matrix — Chip × Test Suite × Backend

> Status values: **Pass** (ran and passed on real hardware) · **Fail** (ran and
> failed; see notes) · **Skip** (collected but skipped, reason stated) ·
> **N/A** (not applicable, reason stated) · **Not Run** (pending hardware).
>
> Every N/A carries a reason. Nothing is silently skipped. Last updated: 2026-11-26.

Backend semantics (matches `vllm_fl/dispatch/policy.py`): `flagos` = FlagGems/FlagOS
runtime implementations; `vendor` = vendor-native SDK (kunlunxin xtorch_ops /
thead FA3 wheel + Triton on PPU / iluvatar corex-patched torch / hygon DTK torch);
`reference` = pure-torch implementations.

## kunlunxin P800

| Suite | flagos | vendor (xtorch_ops) | reference | Notes |
|---|---|---|---|---|
| backend_class (25 cases) | N/A — no kernel execution (forwarding contract) | N/A — same | N/A — same | Runs on CPU and P800 identically; on P800 ~5 availability tests skip (real torch_xmlir present) |
| register_ops (15) | N/A — registration contract | N/A | N/A | CPU |
| attention_metadata (47) | N/A — CPU metadata contract | **Not Run** — on P800 re-runs the profiling path with real kernels (smoke target) | N/A | CPU-verified |
| impls_cpu silu/rms/rotary/gdn (17) | N/A (tests vendor wrappers) | **Not Run** — real-kernel vs torch oracle is the P800 request item #9 | N/A | stub oracle functions reusable |
| moe_fla_conv (42) | N/A | **Not Run** — fused_experts pipeline / FLA chunk / conv1d real kernels (items #3/#8) | N/A | |
| input_batch_launchers (25) | Pass-equivalent — pure torch, runs on any backend | same | same | Backend-agnostic by design |
| patch_collectives (19) | Not Run (real FlagCX under graph) | **Not Run** — real capture collective (item #7) | N/A | CPU-simulated capture verified |
| patch_graph_adaptations (36) | Not Run | **Not Run** — FULL graph capture/replay (item #6) | N/A | |
| patch_ops_adapters (21) | Not Run | **Not Run** — NCW↔NWC + slot mapping on device (item #11) | N/A | |
| patches_misc (16 after trim) | Not Run | Not Run | Not Run | CPU-verified backend-agnostic semantics |
| patches_compat (16) | **Not Run** — FlagTree rope/block_table/mamba kernels compile+execute (item #5) | same | same | Source-level replacement asserted on CPU |
| functional e2e | Not Run | **Not Run** — attention e2e + FLA chunk↔recurrent cross-check (items #1/#3) | oracle role | **Missing suite — to add on hardware run** |
| platform defaults (new) | CPU | CPU | CPU | |

## hygon BW1000

| Suite | flagos | vendor (DTK torch) | reference | Notes |
|---|---|---|---|---|
| platform_branches (14) | CPU-verified | CPU-verified (monkeypatched probes) | N/A | Real BW1000 probe-chain test is P1 request |
| worker_headroom (5+) | N/A — arithmetic | **Not Run** — real `get_device_total_memory` (64 GiB) + headroom end-to-end (P1) | N/A | Formula CPU-verified |
| dispatch_config (new, P0) | CPU | CPU | CPU | yaml contract |
| hygon_worker_integration (3) | Not Run | **Not Run** — `apply_hygon_patches` shim effect in DTK container | N/A | |
| flagos_blacklist real effect | **Not Run** — zeros/cat int32 overflow on >2^31 elements (P2) | N/A | N/A | blacklist yaml contract CPU-pinned |
| SDPA D72 pad motivation | Not Run — verify HCU kernel requires pad (P2) | same | N/A | numerical identity CPU-verified |
| RCCL e2e | Not Run | **Not Run** | Not Run | communicator behavior shared with #799 |

## iluvatar BI-V150

| Suite | flagos | vendor (corex torch) | reference | Notes |
|---|---|---|---|---|
| backend_class | CPU | CPU | CPU | |
| gpu_activation (P0) | **Not Run** — real flaggems_vllm silu vs torch oracle | N/A | oracle role | |
| gpu_attention_e2e (P0) | **Not Run** | N/A | oracle (SDPA) | 64.6% impl coverage; real-card re-run requested |
| gpu_flagcx_allreduce (P0, 2-rank) | **Not Run** | N/A | gloo-channel oracle | 2 devices requested |
| gpu_communicator_allreduce (P0) | **Not Run** | N/A | oracle | |
| gpu_graph_weakrefs (P0) | **Not Run** — real `torch._C` construction primitives | N/A | | |
| gpu_model_runner_v2 (P0) | **Not Run** | N/A | | 8 gpu files need 1 card |
| gpu_moe_align (P0) | **Not Run** — real flaggems no-TLE path | N/A | Counter-bucket oracle | |
| patch_functions (new) | CPU (fake triton) | Not Run — real triton/corex chain | N/A | new CPU suite covers launch-critical paths |
| worker_fl_surface (new) | CPU (object.__new__) | Not Run — real sleep/wake device segments | N/A | |
| moe_router / mrope / shutdown / vision | CPU | CPU | CPU | |
| pipeline PP | Not Run | **Not Run** — real FlagCX send/recv ≥2 ranks (P2) | N/A | |

## thead ZW810E (real PPU already validated once)

| Suite | flagos | vendor (FA3 wheel) | reference | Notes |
|---|---|---|---|---|
| attention_wrapper_cpu (16) | CPU | CPU | CPU | double schema pin |
| gpu_attention (13 cases) | N/A | **Pass** (92-passed batch) | oracle (SDPA) | coverage data not merged into combined JSON — documented |
| gpu_cache (after trim) | N/A | **Pass** (real Triton kernel scatter) | oracle | same data caveat |
| gpu_attention_pipeline (4) | N/A | **Pass** | oracle | write+read stride consistency |
| gpu_platform (11) | Pass (live env) | Pass | Pass | FLAGCX_PATH assert→skip fix pending (R4) |
| capability_contracts (new) | CPU | CPU | CPU | supports_combination / CC range / version pin |
| dispatch_config (new) | CPU | CPU | CPU | thead.yaml routing contract |
| flaggems_attn routing (4) | CPU after fixture fix | N/A | N/A | was cuda-gated, R4 fixes |
| flaggems_attn triton small | **Pass** | N/A | oracle | |
| v2_runner_compat (after trim) | CPU | CPU | CPU | |

## Real-hardware request summary (Phase 4)

| # | Hardware | What | Tests/cases | Priority |
|---|---|---|---|---|
| 1 | P800 ×1 | attention forward chain, silu/rms/rotary/gdn oracle, fused_experts, FLA chunk↔recurrent, FlagTree kernels, graph capture | kunlunxin items #1-#10 | P0 |
| 2 | P800 ×2+ | FlagCX collectives under real capture | patch_collectives real path | P1 |
| 3 | BI-V150 ×1 | 8 gpu-marked files + attention e2e + moe_align + graph weakrefs | iluvatar P0 list | P0 |
| 4 | BI-V150 ×2 | flagcx allreduce 2-rank | functional allreduce | P0 |
| 5 | ZW810E ×1 | re-run GPU suites with coverage capture ON (merge into combined) | 34+13+4 cases | P0 |
| 6 | ZW810E ×2 | UUID uniqueness | 1 test | P2 |
| 7 | BW1000 ×1 | platform probe chain + headroom end-to-end + SDPA D72 motivation | hygon P1 | P1 |
| 8 | BW1000 ×2 | RCCL/communicator e2e | shared with #799 | P2 |

Product bugs found by tests (to report upstream, NOT fixed in this PR — per meeting
rule the coverage PR exposes, product fixes are customer-planned):

1. `vllm_fl/ops/custom_ops.py` `_patch_fused_moe_factory` walks the whole
   `sys.modules` table and calls `vars()` on every entry — stub modules without
   `__dict__` and real C-extension modules raise TypeError from inside
   `register_oot_ops` (real-hardware hazard).
2. `vllm_fl/ops/fused_moe/fused_moe.py` `cpu_invoke` L146-147: the two branches are
   code-identical; the comment claims opposite two-GEMM flag semantics that the code
   does not implement (test now locks the call contract).
3. `vllm_fl/ops/fla/chunk.py` L67-75: state-layout transposition silent-wrong when
   K==V (documented in test as regression guard; CPU asserts layout only).

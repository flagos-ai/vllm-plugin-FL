# PR675 review remediation record

## Integration and verification scope

Review baseline: `05edf01bbb2bb62d4fe1c56f20018576459cd80a`, independently
verified against the PR branch. It includes target `0.5.0-dev@f0e89ad`
through merge commit `dc03bf2`. The shared `vllm_fl/worker/model_runner.py`
and `vllm_fl/worker/worker.py` have no diff against that target. This is not
a claim that all required CI has passed.

The Enflame PR matrix preserves three inference cases and four separate
adaptation-gate jobs (27B/35B-A3B × eager/graph). Independently inspected
archived artifacts `14790260671`, `14793802294`, `14796392600`, and
`14809741567` each contain five scenarios, 26 requests and 260 passing
quality checks. These archived runs do not establish hardware coverage of
subsequent integration commits. Local focused regressions on the review
baseline passed: **55 Enflame/provider/config tests** and **29 Hygon cleanup
tests**. They do not replace hardware or cross-platform CI validation.

The official FlagGems-vllm main branch at
`23e7747dbb99fef74f23480adf69133739c32141` still diverges from required
Iluvatar fork commit `90dd13cd721f2ff139e6f42ef36854518c0b2b95`
(GitHub compare: ahead 113, behind 1). Merely changing the installer URL
would therefore not prove the required changes landed upstream. Direct source
inspection also found that main's four-stage MoE alignment still lacks the
fork's negative/out-of-range expert-ID masks in stages 1 and 4.
Upstream follow-ups already exist, so creating duplicate PRs is unnecessary:

- [FlagGems-vllm #1317](https://github.com/flagos-ai/FlagGems-vllm/pull/1317),
  head `2eb42fd2c5471446bea457f5f7a1ca65ae44ba09`: invalid-expert-ID
  filtering. At inspection it was open/unmerged, with `unittest-required`
  failing because the required NVIDIA operator lane was cancelled (job
  `136455749312`: `python-op was expected to succeed, got: cancelled`).
  Its reported H100 validation is author-provided evidence, not a validation
  run performed for this record.
- [FlagGems-vllm #1231](https://github.com/flagos-ai/FlagGems-vllm/pull/1231),
  head `40856d9224183ef2182c017705ab1502e45a167b`: Enflame migration.
  At inspection it was open/unmerged, with `rule-check-required` failing and
  NVIDIA/backend tests skipped. The downloaded job `132381462791` reports
  47 forbidden Torch compute calls across 158 checked files, including FLA
  normalization/gates and MoE activations. These require operator-library
  fixes; bypassing the policy would not resolve R1. Its existence alone does
  not establish full coverage of R1's numerical implementations.

R1/R2 remain open until the relevant operators/fixes are accepted upstream,
validated and pinned to immutable official commits. The authenticated account
has no push/merge permission on the official provider repository.

The dtype-codegen workaround now retains the vendor helper and removes only
four recognized integer rewrite statements. Unknown helper ABI/source emits a
warning and skips adaptation. `FL_GCU_PRESERVE_DTYPE_CODEGEN=0` selects the
vendor path; changing this mode changes the compilation cache salt. This is
a downstream workaround, not evidence of a vendor upstream fix.

## Review disposition

| Review | Status | Outstanding evidence |
|---|---|---|
| R1 numerical implementations | OPEN | Official operator integration, independent tests and immutable official SHA. Private/Torch/CPU implementations remain. |
| R2 dependency provenance | OPEN | Shared installer uses official sources and Iluvatar reuses its image, but preinstalled image provenance and official landing of old fork changes remain unproved. |
| R3 full gate | Current-code hardware rerun passed | Fresh-cache RMS fix `e394a2a`: all four groups completed five scenarios, 26 requests and 260/260 checks each; zero skipped scenarios, all test/server exits 0. [Bundled raw JSON, complete logs, commands, environment and portable audit](../validation/pr675-e394a2a/README.md) cover 104 requests and 1040 checks. All 15/26 weight shards match immutable Qwen revisions `1cda14dd71a3f47115850eb790f416f165da1cb5` / `345d467b117d1fd5eda9b879a5988de9ae5f47a7`; tokenizer/chat-template and 35B configuration overrides are preserved with hashes. These are not exact upstream model snapshots. Required hosted/cross-platform CI remains a separate validation scope. |
| R4 image BOM | PARTIAL | [Digest-verified layer package evidence](../validation/pr675-image-732e3d5/README.md) now establishes TOPS SDK 3.8.20260714, ECCL 3.7.1.2, FlagCX 0.13.0 and the compiler/operator package sequence. All layers after the SDK were inspected, including the Triton GCU July 22 replacement. Public source SHAs for local/binary builds and hashes/paths of actually loaded communication libraries remain missing. |
| R5 runner docs | Corrected | Actual runner still must be recorded per serving log, not inferred from an unset environment variable. |
| R6 RMS contract | Execution-mode consistency fixed; independent CUDA validation outstanding | Eligible GCU inference now uses the same RMS custom operator in eager and compiled execution. The real 12×1024 BF16 diagnostic changed from one mismatch (max abs 0.0009765625) to zero across eager, compile and graph replay. Eleven focused tests pass, including masked-width replay coverage; restoring the old compile-only guard makes the new regression fail. The public FlagGems implementation differs on 4231 elements (max abs 0.0625) on this input, so it cannot yet replace the CUDA-contract operator. The NVIDIA reference endpoint refused connections during this verification; independent provider parity remains unverified. |
| R7 atomic install | Regression passed | Import/ABI failure, assignment rollback, same-process retry and idempotence covered. |
| R8 CPU top-k | PARTIAL | Follow-up evidence contains 24 warmed ABBA service trials across both vocabularies and concurrency 1/4/16: mean ITL overhead 0.44–3.79%; 27B throughput has explicit wall-time outliers. Public FlagGems batch-1/vocab-151936 top-k completed exactly after 590.652 s cold compilation, 2.456 ms warmed. A separate hierarchical composition of public FlagGems top-k/gather passed exact value/index checks for batch 1/vocabulary 151936 and batches 4/16/vocabulary 248320. Warmed latency was 4.89/22.20/82.86 ms versus 0.63/2.63/8.19 ms for CPU top-k including transfers. This measured composition is 7.7–10.1× slower and was not adopted; it does not establish a lower bound for other device implementations. These are random-input operator microbenchmarks, not model-quality or end-to-end service results. A faster validated public provider remains outstanding. |

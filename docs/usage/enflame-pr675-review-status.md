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
  failing and NVIDIA operator CI cancelled. Its reported H100 validation is
  author-provided evidence, not a validation run performed for this record.
- [FlagGems-vllm #1231](https://github.com/flagos-ai/FlagGems-vllm/pull/1231),
  head `40856d9224183ef2182c017705ab1502e45a167b`: Enflame migration.
  At inspection it was open/unmerged, with `rule-check-required` failing and
  NVIDIA/backend tests skipped. Its existence alone does not establish full
  coverage of R1's numerical implementations.

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
| R3 full gate | Execution passed; provenance partial | All four groups completed 26 requests and 260/260 checks each, with zero skipped scenarios. Raw JSON, logs, launch commands and environment records are retained outside this PR to avoid shipping generated experiment artifacts. Local model metadata hashes are recorded, but immutable upstream model revisions remain unknown. |
| R4 image BOM | OPEN | Published digest not tied to verified SDK/FlagCX/ECCL package/source manifest. Development container is not a substitute. |
| R5 runner docs | Corrected | Actual runner still must be recorded per serving log, not inferred from an unset environment variable. |
| R6 RMS contract | PARTIAL; strict consistency diagnostic failed | Exact CUDA source supports intended rounding. A real BF16 input diagnostic (raw records retained outside this PR) finds one normalized-output mismatch (max abs 0.0009765625) between GCU eager and compile/replay with residual; no-residual output and residual sum match exactly. This is not independent NVIDIA provider validation, which remains outstanding. |
| R7 atomic install | Regression passed | Import/ABI failure, assignment rollback, same-process retry and idempotence covered. |
| R8 CPU top-k | PARTIAL | Follow-up evidence contains 24 warmed ABBA service trials across both vocabularies and concurrency 1/4/16: mean ITL overhead 0.44–3.79%; 27B throughput has explicit wall-time outliers. Public FlagGems batch-1/vocab-151936 top-k completed exactly after 590.652 s cold compilation, 2.456 ms warmed. Larger-shape validation and controlled provider comparison remain required before replacing the CPU fallback. |

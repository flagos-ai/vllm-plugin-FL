# PR675 review remediation record

## Integration and verification scope

Integrated target `0.5.0-dev@77f141a961d75e54c0cf209de80c99a7c7eec04b`
as merge commit `262f814db849e4687bdabd5417832ed9cf83265a`.
Exact remote SHA independently verified; GitHub reports `mergeable=true`.
This is not a CI success claim.

The generated Enflame PR matrix preserves three inference cases and four
separate adaptation-gate jobs (27B/35B-A3B × eager/graph). Matrix generation
proves routing, not execution. Local host regressions: **52 passed**, including
gate validation, compilation policy, provider installation and V2 buffer retry.
Ruff checks/formatting and shell syntax checks passed.

## Review disposition

| Review | Status | Outstanding evidence |
|---|---|---|
| R1 numerical implementations | OPEN | Official operator integration, independent tests and immutable official SHA. Private/Torch/CPU implementations remain. |
| R2 dependency provenance | OPEN | Shared installer uses official sources and Iluvatar reuses its image, but preinstalled image provenance and official landing of old fork changes remain unproved. |
| R3 full gate | Execution passed; provenance partial | All four groups completed 26 requests and 260/260 checks each, with zero skipped scenarios. JSON, logs, launch commands and environment are in `docs/validation/pr675-262f814/`. Local model metadata hashes are recorded, but immutable upstream model revisions remain unknown. |
| R4 image BOM | OPEN | Published digest not tied to verified SDK/FlagCX/ECCL package/source manifest. Development container is not a substitute. |
| R5 runner docs | Corrected | Actual runner still must be recorded per serving log, not inferred from an unset environment variable. |
| R6 RMS contract | PARTIAL | Exact CUDA source supports intended rounding; independent provider execution plus S60 eager/compile/capture/replay evidence still required. |
| R7 atomic install | Regression passed | Import/ABI failure, assignment rollback, same-process retry and idempotence covered. |
| R8 CPU top-k | PARTIAL | Unused values no longer return to GCU; full logits still transfer to CPU. Historical one-vocabulary measurements do not close current multi-vocabulary acceptance. |

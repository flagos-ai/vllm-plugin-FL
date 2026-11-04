# Enflame Runtime CI Implementation Plan

> Historical implementation record. On 2026-11-05 the user requested that no
> further superpowers skills be used, no additional nightly run be made, and
> the files be shown for confirmation before PR creation.

**Goal:** Build, validate and publish the requested Enflame vLLM 0.28 inference CI image and submit its recipe as a PR.

**Architecture:** Rebase the existing staged Dockerfile onto the digest-pinned runtime 2.3.0. Preserve the vendor torch/compiler stack and install the plugin revision under test during CI setup. Validate in a private container on the supplied S60 host.

**Tech Stack:** Docker, bash, Python 3.12, torch-gcu, FlagTree, FlagGems, vLLM 0.28, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-enflame-runtime-ci-design.md` (approved 2026-11-04).

## Global Constraints

- Base: runtime tops1.10.6 2.3.0, manifest `sha256:6356cd26647a4cbd27f2ad9ece7281df888df7ec9e9c2dc3bb5aaa262f9097c6`.
- Final tag: `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-enflame-ci`.
- Branch starts at `0.4.0-dev`; PR targets `BrianPei:ci/thead-0.28`.
- No unrelated training packages; no test weakening or hidden exclusions.
- User authorized direct implementation and routine decisions; no additional plan approval needed.

## Review Focus

- Build helper defaults must use the same base digest as direct docker builds.
- /flagos Python must remain active in noninteractive CI containers.
- Transitive installs must preserve vendor torch and compiler file hashes.
- Test skips/missing weights must not become a full-CI success claim.
- CI and docs must reference the actually published new digest.

### Task 1: Runtime-based candidate recipe

**Files:** `docker/enflame/Dockerfile.v0.28.0`, `docker/build.sh`, optional Enflame dependency inputs.
**Interfaces:** Existing `docker/build.sh --platform enflame --target ci`; produces a candidate dependency image.

- [x] Inspect baseline compiler/package metadata and confirm base layer digest.
- [x] Change both base defaults, install missing runtime/build/test dependencies, preserve compiler guards.
- [x] Run bash syntax and build-helper argument checks; build on S60 with logs in a private tmux session.
- [x] Confirm candidate package metadata and compiler integrity; commit recipe.

### Task 2: Full configured Enflame validation

**Files:** `docker/enflame/` validation script/report as needed; dependency fixes belong in Task 1 recipe.
**Interfaces:** Candidate image and selected plugin checkout produce raw logs, exit codes and reports.

- [x] Run check.sh/setup.sh from a clean private container with read-only models and free devices.
- [x] Run all unit and functional tests, retaining output and skip reasons.
- [x] Generate PR/nightly/weekly matrices; run every unique Enflame E2E and benchmark selection and available documented graph selections.
- [x] Diagnose actual failures, repair necessary image dependencies, rebuild and rerun affected checks.
- [x] Record exact source, environment, results, failures and limitations; acceptance requires passing results.

### Task 3: Publish and PR

**Files:** `.github/configs/enflame.yml`, `docker/enflame/README.md`, `docs/usage/enflame.md`, validation report.
**Interfaces:** Validated local image produces registry manifest digest and reviewable PR.

- [x] Push final requested tag after validation, verify remote digest and ancestry.
- [x] Update Enflame image references/build instructions/results using verified new digest.
- [x] Run relevant lint/build checks and fresh whole-branch review, resolve important findings (reviewed through `48b11a2`; later filename/docs adjustments checked separately).
- [ ] Push working branch and create PR to the requested base, with honest validation evidence.

## Verified execution record

- Task 1 complete: recipe `320bbe7`; build 06 passed, required runtime's 16-layer prefix and CI metadata verified, compiler guards and separate clean-image wheel install passed. Shell syntax, ShellCheck and repository typos checks passed.
- Task 2 complete: source `320bbe7` in a fresh corrected-image container, native nightly `--scope all` exited 0 with 10/10 groups passed (2876.46s). UT 1959+2 passed, FT 71 passed, both models' eager inference/serving/eight-request concurrent and benchmark passed. Original 35B graph serving (2 tests) and 27B graph inference (1 test) also passed on a separate verified device pair.
- Task 3: final requested tag published and pulled by manifest `sha256:aabdf954a2f4cb60c80e972d1e105a07b80bed321efed887b24997f25195ea5e`, resolving to tested image `c7efd76...`. CI/usage references updated. Independent whole-branch review through `48b11a2` found no actionable issues and independently verified the raw evidence archive. PR creation now waits for user confirmation.

## Rulings and practical limits

- Verify the recipe/configuration migration with actual builds and existing hardware tests; file-content unit tests would mirror the configuration. Cost: hardware verification is slower.
- Invoke unregistered graph examples through original pytest files; the native graph filter selects zero cases. Cost: supplemental validation needs separate commands.
- Match NumPy 2.3.5 to required OpenCV and the reference stack after reproducing the 1.26 resolver conflict. Cost: optional vendor metadata remains incompatible.
- Use SHA256-checked source bundles and temporary Git mirrors after direct network failures, preserving upstream identity and commit pins. Cost: temporary task daemons require cleanup.
- Bake the CI-pinned FlagGems-vllm dependency to avoid setup's failing GitHub fetch. Cost: the CI image carries this necessary fixed dependency.
- Increase S60 concurrent output allowance to 256 after actual 64-token truncation; eight requests and all answer checks remain intact. Cost: additional allowed generation work.
- Extend 35B graph startup to 40 minutes after an actual 20-minute timeout during capture. Cost: other platforms sharing that YAML also wait longer before reporting failure.
- Clear the inherited all-ops CPU debug fallback through a late profile: controlled exit134 versus unset exit0, then all 71 original FT cases passed. Cost: untested unsupported native ops cannot depend on that debug fallback.
- Supersede old-image attempt03 after confirming the environment defect; its interrupted 35B concurrent result is not counted as passing. Cost: no complete old-image aggregate.
- Reuse only task compiled caches in final validation and retain separate cold-run evidence. Cost: final timings are warm-cache rather than cold capacity claims.
- Run supplemental graph tests in a separate container, checkout and copied caches on verified idle devices4,5 while nightly uses2,3. Cost: two additional free devices are occupied temporarily.
- Preserve actual vendor pip-check conflicts instead of replacing vendor Torch/compiler to make metadata superficially green. Cost: ModelOpt and generic dependency re-resolution remain outside validated inference CI.

Detailed sources, failures, corrections and raw evidence locations are in
`docker/enflame/validation-20261104.md`. No new test exclusions, precision
changes, weaker assertions or adaptation Gate capacity claims were introduced.

## User review handoff (2026-11-05)

- Merged the latest requested target `ci/thead-0.28` at `0a5fecc`; retained
  the own branch `codex/enflame-runtime-ci-0.28`.
- Moved the tested recipe bytes to `docker/enflame/Dockerfile.v0.28.0`,
  restored the target's unversioned Dockerfile, and selected the versioned
  recipe in `docker/build.sh`.
- PR validation plus supplemental graph commands are the current handoff
  instructions; completed nightly evidence remains recorded without a rerun.
- No PR has been created; show the changed files before the user confirms.

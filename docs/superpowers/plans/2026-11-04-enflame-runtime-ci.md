# Enflame Runtime CI Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Track verified results below.

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

**Files:** `docker/enflame/Dockerfile`, `docker/build.sh`, optional Enflame dependency inputs.
**Interfaces:** Existing `docker/build.sh --platform enflame --target ci`; produces a candidate dependency image.

- [ ] Inspect baseline compiler/package metadata and confirm base layer digest.
- [ ] Change both base defaults, install missing runtime/build/test dependencies, preserve compiler guards.
- [ ] Run bash syntax and build-helper argument checks; build on S60 with logs in a private tmux session.
- [ ] Confirm candidate package metadata and compiler integrity; commit recipe.

### Task 2: Full configured Enflame validation

**Files:** `docker/enflame/` validation script/report as needed; dependency fixes belong in Task 1 recipe.
**Interfaces:** Candidate image and selected plugin checkout produce raw logs, exit codes and reports.

- [ ] Run check.sh/setup.sh from a clean private container with read-only models and free devices.
- [ ] Run all unit and functional tests, retaining output and skip reasons.
- [ ] Generate PR/nightly/weekly matrices; run every unique Enflame E2E and benchmark selection and available documented graph selections.
- [ ] Diagnose actual failures, repair necessary image dependencies, rebuild and rerun affected checks.
- [ ] Record exact source, environment, results, failures and limitations; acceptance requires passing results.

### Task 3: Publish and PR

**Files:** `.github/configs/enflame.yml`, `docker/enflame/README.md`, `docs/usage/enflame.md`, validation report.
**Interfaces:** Validated local image produces registry manifest digest and reviewable PR.

- [ ] Push final requested tag after validation, verify remote digest and ancestry.
- [ ] Update Enflame image references/build instructions/results using verified new digest.
- [ ] Run relevant lint/build checks and fresh whole-branch review, resolve important findings.
- [ ] Push working branch and create PR to the requested base, with honest validation evidence.

## Progress

- Recon complete: private clone/branch, host accessible, base pulled, models present, no active device processes at initial inspection.
- Design builder check: no material unanswered questions. User approved direct execution.
- Configuration changes use build/runtime verification rather than tautological file-content unit tests.

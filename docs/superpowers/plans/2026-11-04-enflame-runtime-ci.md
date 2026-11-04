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

- Task 1: recipe defaults and CI packages committed at df5eb80; bash syntax and build-helper requested-base/output-tag checks passed. Candidate build-01 running in task tmux. Buildx 0.20.1 was extracted from Ubuntu package into task-local Docker config; host installation unchanged.
- Ruling: configuration migration is verified by actual Docker build, compiler guard, and existing hardware CI suites rather than file-content tests.
- Ruling: unregistered graph cases require direct original pytest invocation; tests/run.py filters these out and returns zero without executing them.

- Dependency probe: numpy==1.26.4 plus opencv-python-headless==4.13.0.92 failed with ResolutionImpossible; numpy==2.3.5 passed. Matched reference image NumPy 2.3.5, transformers 5.6.0, huggingface-hub 1.33.0, xgrammar 0.2.8, opencv-python-headless 4.13.0.92; recipe committed at 5ee9676dd6eca0be9dda1c0a0d48c53651ca156e.
- Network ruling: source clone stalled on host. Fetch official v0.28.0 locally, verify 2cf0a6915ce544dc493a0990f2ea38d81601128a, transfer SHA256-checked Git bundle via supplied bastion and expose a temporary Git cache only on Docker bridge. Build still verifies the exact official commit.

- Build 04 succeeded; base layer prefix proved (16 runtime layers retained), compiler guard and device check passed. Setup failed on GitHub TLS while installing the required pinned FlagGems-vllm dependency.
- Ruling: bake CI-pinned FlagGems-vllm into CI stage, preserving direct_url identity; optional task-local Git transport mirror avoids changing package identity. Build 05 at 1af55e7 running.
- Host terminal long input was interrupted by bastion confirmation; verified no duplicate build/test process, rewrote task script as readable file and restarted normally.
- Actual dependency metadata conflicts were retained in validation report; do not overwrite FlagTree or vendor Torch to make pip check superficially green.

- Baseline hardware validation: UT/FT, 27B eager inference (1506s), 35B eager inference (1169.5s) passed. Candidate manifest a10eb502e815568815730684dbf4dd52bac1f4afda0b98679445ffaf252b3c4f published; final tag unchanged.
- PR target advanced to ad8ebc972a7a2b8666796891f14ca81d674618b0 and reverted its earlier Enflame graph/Gate skips. Synchronize existing target changes to avoid duplicate patches/conflicts; preserve baseline test evidence and source transitions.

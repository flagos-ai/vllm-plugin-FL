# Enflame runtime-based vLLM 0.28 inference CI image

## Intent and scope

Build a reproducible Enflame S60 inference development/CI dependency image from
`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-enflame-tops1.10.6:2.3.0@sha256:6356cd26647a4cbd27f2ad9ece7281df888df7ec9e9c2dc3bb5aaa262f9097c6`.
Publish the validated result as
`harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-enflame-ci` and submit a PR
targeting `BrianPei/vllm-plugin-FL:ci/thead-0.28` from a new branch based on
`0.4.0-dev` (`ed8c4801dc1f62b2fc7c5bf01a125a2bd52d2c1b`).
The user clarified that this is inference development/CI, not FlagScale training.
Do not install FlagScale or unrelated training frameworks.

## Approach

Modify the existing Enflame Dockerfile and build helper in place, preserving its
base/dev/ci/release stages and exact vendor version/compiler guards. This is the
single approach consistent with the requested base and existing build interface.
Reusing the old inference image as FROM would violate the requested provenance.
Copying its entire filesystem would hide dependencies and undermine reproducibility;
its documented versions and available package artifacts may inform the recipe.

The runtime was inspected on the supplied S60 host: Python 3.12.13, torch
2.11.0+cpu, torch-gcu 2.11.0+3.8.20260713, FlagGems 5.4.0, FlagTree
0.7.0+enflame3.6, flagcx 0.14.0+tops1.10.6, and pandas 3.0.6.
Keep /flagos as the active interpreter environment. Install official empty-device
vLLM 0.28.0 from its pinned source commit, its runtime dependencies, plugin build
tools, and all dependencies exercised by this repository's CI. Keep the plugin
checkout editable-install step in setup.sh, matching existing CI behavior.
Protect vendor torch and the FlagTree triton namespace against dependency resolution.
Record exact image ancestry, source revision, build logs, package inventory and digest.

## Validation and acceptance

Use the actual checked-out workflows and platform YAML as the CI source of truth.
The supplied usage document differs from BrianPei's current branch configuration.
Run check.sh and setup.sh, full unit and functional suites, then all configured
nightly E2E inference/serving/concurrent cases for both Qwen3.6-27B and
Qwen3.6-35B-A3B (TP2 eager), plus configured serve benchmark smoke. PR and weekly
matrix coverage must be audited against nightly; run additional unique selections.
Also run documented eager/graph cases where available, recording these separately
from configured CI. Run relevant lint/build checks and retain raw reports.
Do not suppress failures, weaken expected outputs, or add exclusions to manufacture
acceptance. Missing models and skipped tests must be explicit. A dependency image
build alone does not establish a successful CI run.

Use a private container, read-only model mounts, available device IDs verified
with efsmi, and a unique tmux session for long commands. Keep the private PID
namespace because the test runner performs cleanup. Do not disturb other services.

## Implementation decisions delegated to the builder

Resolve missing system/Python packages from actual build and test failures; pin
vendor-sensitive packages and verify the compiler after installation. Use a
candidate tag until validation succeeds, then push the requested final tag and
pin its new digest in the Enflame CI config and updated documentation. If a test
exposes a repository/backend bug rather than a missing dependency, preserve and
report the evidence, fix only changes necessary for this Enflame image, and rerun
the affected checks. Do not claim completion or a fully passing CI without results.
Credentials are used only for authenticated operations and never committed.

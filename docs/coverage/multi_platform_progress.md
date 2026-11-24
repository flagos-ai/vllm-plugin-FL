# Multi-Platform Supplementary Coverage — Engineering Log

Branch: `coverage/multi-platform-test-coverage` (base `upstream/0.5.0-dev` @ `4856534d`).

This document tracks the engineering progress of the multi-platform
supplementary coverage effort (kunlunxin p800 / hygon bw1000 / iluvatar
bi_v150 / thead zw810e). It is updated as work completes and serves as the
audit trail for the final PR description.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| Repository / CI / test-framework audit | done | 4 platform configs cross-verified (p800 / bw1000 / bi_v150 / zw810e against `tests/platforms/*.yaml` + `.github/configs/*.yml`); runner labels captured (kunlunxin-p800, hg-cicd-vllm-plugin-0240, iluvatar-cicd-vllm-plugin, flagcicd-810e) |
| Structure analysis (5 parallel agents) | done | infra + 4 platform agents produced module maps, branch-condition tables, coverage-gap hypotheses, candidate test matrices |
| Supplementary infra | done | commit `2e3d8c13` (runner, conftest, tools, workflow) |
| Coverage baseline v1 (4 platforms) | done (invalid) | 4×16-worker parallel run OOM-killed 64 workers in the 9.5 GiB WSL2 VM → only 7 tests collected per platform, 7.6% coverage — measurement invalid, not reported as baseline |
| Coverage baseline v2 (sequential, flag_gems import enabled) | done | see below |
| Coverage deviation analysis | pending | module-level gap tables per platform from v2 baseline |
| Supplementary tests (4 platforms, parallel agents) | pending | |
| Coverage improvement validation | pending | same-commit core→combined delta per platform |
| Regression & independent review | pending | |

## Baseline measurement environment

* WSL2 Ubuntu 24.04 (16 vCPU, 9.5 GiB RAM), Python 3.12.3.
* venv: vllm 0.32.0 (real), torch 2.9.1+cpu, triton 3.9.0, pytest 9.2.0,
  pytest-cov 7.1.0, pytest-xdist 3.8.0, pytest-json-report 1.5.0.
* flag_gems 5.0.2 (PyPI wheel) installed with import-time CPU shims
  (documented in the engineering log below) so the platform layer
  (`vllm_fl.platform.PlatformFL` → device_type cuda, vendor per GEMS_VENDOR)
  initializes on a CUDA-less host. Real vendor kernels are NOT executed;
  GPU functional coverage remains Hardware Validation Pending.
* No vendor hardware present. `gpu`/`multi_gpu`-marked tests auto-skip via
  the root conftest. E2E scopes are not part of the CPU baseline.
* Command per platform (unchanged core entry points):
  `python tests/run.py --platform <p> --device <d> --scope unit` then
  `--scope functional` (unit: parallel `-n 16` + serial pass with
  `--cov-append`; functional: appends into the same
  `coverage-<platform>.json`).
* Coverage data isolation: `COVERAGE_FILE` per platform output dir; JSON
  `coverage-<platform>.json` snapshotted per platform.

### v1 invalidation (why baselines were re-run)

The first attempt ran all four platforms concurrently (4 × 16 xdist
workers ≈ 220 python processes on 9.5 GiB). The WSL kernel OOM-killed 64
workers ("maximum crashed workers reached", "Out of memory: Killed
process"), the parallel unit pass collected only 7 tests per platform and
the resulting 7.6% coverage JSONs were measurement artifacts. v1 results
were discarded; v2 runs the platforms sequentially.

### flag_gems CPU import shims (measurement-env-only)

The core unit suite imports `vllm_fl` modules that transitively import
`flag_gems` / `flaggems_vllm`. The PyPI flag_gems wheel was written against
a CUDA libdevice triton surface, so on a CPU-only host its import fails
before any vllm_fl platform initialization. The measurement venv installs a
`.pth`-executed shim module that, at import time only:

1. patches missing `triton.language.math` / `triton.language` names from
   `triton.language.extra.cuda.libdevice`;
2. makes `triton.autotune` drop flag_gems-specific kwargs;
3. makes `triton.Config.__init__` accept the hygon `num_ldmatrixes` default;
4. aliases `triton.language.math.xpu_trunc_div` to a real libdevice fn.

None of these execute GPU code; they satisfy import-time symbol resolution
so module import succeeds and coverage can be measured. The thead baseline
sets `GEMS_VENDOR=kunlunxin` for import purposes (the PyPI wheel's vendor
enum has no thead entry — the thead CI image carries a patched flag_gems)
while `FL_TEST_PLATFORM=thead` still selects the thead platform YAML.

## Coverage baseline results

Measured at `4856534d`, scope unit+functional (CPU-executable), same
environment for all four platforms.

### Iteration history

| Run | Result | Reason |
|---|---|---|
| v1 (4 platforms parallel) | invalid (7.6%) | 4×16 xdist workers OOM-killed on the 9.5 GiB WSL2 VM; only 7 tests collected per platform |
| v2 (sequential, flag_gems shims) | ~29.4-29.9% | flag_gems importable but platform layer not registered (vllm_fl not installed in the venv — the vllm.platform_plugins entry point was absent, so `current_platform` stayed UnspecifiedPlatform and every VllmConfig-dependent test failed) |
| v3 (sequential, entry point + flaggems_vllm installed) | ~25.3-26.0% | full suite collected and ran; residual failures from missing torchvision (4 model-import tests) and 163 worker errors traced to `vllm.kernels` import-time device probe |
| v4 (sequential, all env fixes) | final | torchvision 0.24.1+cpu installed; capability-probe guard added to the shims; full-dir re-verification: 2475 passed / 7 failed / 2 errors (the 2 are ascend/gcu collection errors from absent torch_npu/gcu packages — expected on this host) |

### v4 baseline numbers (CPU measurement env)

| Platform | Coverage | Covered / Statements |
|---|---:|---:|
| kunlunxin | (recorded from v4 run) | |
| hygon | (recorded from v4 run) | |
| iluvatar | (recorded from v4 run) | |
| thead | (recorded from v4 run) | |

### T-Head real-hardware baseline (zw810e PPU)

A dedicated T-Head host (16× PPU-ZW810E, the CI image
`vllm-plugin-fl:v0.32.0-thead-ci@sha256:abb224ad…`) was provided. A
dedicated container (`mp-cov-thead`) was launched on idle PPUs 8-11 with
the CI env (GEMS_VENDOR=thead, VLLM_TARGET_DEVICE=empty, FLAGCX_PATH,
THEAD_BASE_PYTHON), the same frozen revision extracted, and the same core
runner commands executed:

| Run | Result |
|---|---|
| thead real-hw v1 | 22.94% (5829/25410) — 182 failed dominated by `vllm._C_stable_libtorch` missing in the empty-build vllm (230 hits) |
| thead real-hw v2 | (with FLAGCX_PATH/LD_LIBRARY_PATH added, matching the CI container env) |

## Infrastructure decisions (for reviewers)

* `tests/supplementary_tests/run.py` subclasses the core `TestRunner`;
  discovery-only override; coverage isolated per platform via
  `COVERAGE_FILE=.coverage-<platform>-supp`; exit-5 exemption only for the
  supplementary serial pass (mirrors the core `unit-serial` semantics).
* `supplementary` marker + `platforms(...)` gate registered in the suite's
  local conftest only — root conftest, pyproject, and every core test file
  are untouched.
* Workflow `coverage_supplementary_multi.yml`: setup job builds the matrix
  from `.github/configs/<platform>.yml` (single source of truth for runner
  labels and images) + per-platform coverage job. Schedule Saturday 06:00
  UTC avoids nightly (18:00 UTC daily), weekly (Sunday 18:00 UTC), and the
  nvidia supplementary workflow's Sunday 06:00 UTC window. No PR trigger;
  daily PR CI untouched.

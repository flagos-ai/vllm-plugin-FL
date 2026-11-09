# CUDA Runtime 2.3.0 CI Image Implementation Plan

> Execution: superpowers:executing-plans, in the user-specified local checkout and branch.

Goal: build the NVIDIA vLLM 0.32.0 inference/CI image from FlagOS Runtime CUDA 13.3:2.3.0, preserving its core stack and validating the repository's NVIDIA tests.

Architecture: install the native `vllm[audio]==0.32.0+flagos` closure using the vendor and ordinary indexes, then the plugin/CI dependencies. Validate the vendor stack and genuine native extensions instead of old cu129 development-source pins. Keep the base image reference tagged, without an image digest.

Constraints: local branch `codex/cuda-runtime-2.3.0-0.5.0-dev`; H20 via existing authorized SSH; target `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.32.0-cuda-ci`; no credentials in files; report failures without weakening tests. Recipe filename is `Dockerfile.v0.32.0`, renamed at the user’s request on 2026-11-10 to match the actual vLLM target.

- [x] Install the authorized dependency closure in the isolated H20 trial container; record actual additional dependencies and failures.
- [x] Add failing regression checks for the native FlagOS wheel and protected Runtime versions; adapt stack.py without removing ELF/RECORD/origin checks.
- [x] Add a concise Runtime Dockerfile, constraints and CI requirements; update build.sh, setup.sh and NVIDIA image configuration.
- [x] Run checker regressions, shell/static checks, then build on H20; run stack checks and repository UT/FT/E2E/benchmark, recording per-stage results.
- [x] Review changes and dependency inventory; publish the requested tag only with accurate validation status, never describe failed stages as passing.

Review focus: reject empty wheels and Python shims; retain Torch/cu130 and FlagTree ownership; reconcile modelscope with pyproject; keep Git provenance for FlagGems-vllm; avoid accidental model downloads or GPU conflicts with other jobs.

Execution ledger:
- Baseline: local user branch is at 5725277; only existing untracked IDE/audit artifacts. H20 GPU 7 is reserved for the isolated trial, other jobs are not stopped.
- Authorized combined vendor/ordinary package source installation has started. Full image tests will use currently available GPUs after checking occupancy.

- Completed authorized install in trial: UT 1874 + 2 passed (separate serial pass), functional runner passed. Core seven preserved; setuptools 80.10.2, modelscope 1.35.3 and standby Triton 3.7.1 installed.
- Reviewer findings resolved: all 13 native payloads checked (six extra missing-library regressions red before fix); vendor-first compatible version resolution; explicit source version and revision in image.
- Build-time libcuda absence resolved by metadata/ELF-only build checks; driver loading remains mandatory at GPU runtime.
- First-index cannot select setuptools 80.10.2 because vendor serves only 70.2.0/84.2.1. Switched to unsafe-first-match: prefer compatible vendor versions, fall back only when none exists.
- Fresh-image test found XGBoost cu12 NCCL overwrites the same native file as cu13 NCCL. Verified repair by restoring original nvidia-nccl-cu13==2.29.7; Torch works and loaded NCCL reports 2.29.7. Added actual loaded-version regression (red then green), 28 checker tests pass.
- Final image built successfully as v0.32.0-cuda-ci-validated; fresh GPU container pip check and full stack check pass, 2 GPUs visible. Running UT/FT/E2E/benchmark in that image before publishing.
- H20 benchmark scope previously discovered zero cases, so explicitly enabled existing throughput/latency/serve smoke cases. Empty discovery was not recorded as a pass.

- Final Runtime package closure additionally needed CUDA_HOME/PATH and pinned nvcc/CRT/NVVM 13.0.88; verified NCCL file overlap repair and FlashInfer JIT compilation/linking. Added toolkit lib64/cudart symlinks, preserving actual runtime libraries.
- Final image new container pip check/full stack/FT/UT/benchmark pass. H20 matrix 4 cases / 6 pytest checks pass under final configuration; 35B eager serving runs in the final published image. Graph cases verified by JUnit after SSH stdout disconnect, not assumed from progress bars.
- Published requested Harbor tag successfully, PUSH_EXIT=0. Full dependency inventory and evidence saved in the audit report. CPU checker regressions 28 pass, ruff/shell/whitespace checks pass. No deferred reviewer findings.

# Enflame runtime-based vLLM 0.28 validation

Status: accepted and published. Completed 2026-11-05 (Asia/Shanghai).
The native nightly entrypoint passed all 10 groups (0 failures), and the
supplemental original graph tests passed. These are executions on the supplied
host; GitHub Actions PR status is separate.

## Image and source

- Required base: `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-enflame-tops1.10.6:2.3.0@sha256:6356cd26647a4cbd27f2ad9ece7281df888df7ec9e9c2dc3bb5aaa262f9097c6`.
- Published image: `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-enflame-ci@sha256:aabdf954a2f4cb60c80e972d1e105a07b80bed321efed887b24997f25195ea5e`.
- Image ID: `sha256:c7efd76edccb35241c65863b739efe2c6336b031c8b79a11460f379ac927b1a4`.
- Recipe and corrected-image test checkout: `320bbe7436d92553d5cbb4858a582141cf70e4c4`.
- Branch: `codex/enflame-runtime-ci-0.28`, started at `0.4.0-dev` commit `ed8c4801dc1f62b2fc7c5bf01a125a2bd52d2c1b`; PR targets `BrianPei:ci/thead-0.28`.
- vLLM: official v0.28.0 source commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`, installed as the empty-device build.
- All 16 required runtime layers are the exact prefix of the corrected image's RootFS layers. OCI metadata records the base digest, recipe revision and `ci` stage.
- Build 06, final push, pull by digest and anonymous Harbor artifact lookup passed. Harbor reports compressed size 8,829,706,738 bytes.

## Environment

S60/C200 host driver 1.10.6; private container with private PID namespace,
`TOPS_VISIBLE_DEVICES=2,3` for nightly, privileged device access and host IPC.
Supplemental graph validation uses a second private container on separately
verified idle devices `4,5`, with its own checkout and copied caches. Complete
27B weights (15 indexed shards, 55.56 GB) and 35B-A3B weights (26 shards,
71.90 GB) are mounted read-only.

Python 3.12.13 in `/flagos`; torch 2.11.0+cpu, torch-gcu
2.11.0+3.8.20260713, FlagGems 5.4.0, FlagTree 0.7.0+enflame3.6,
triton-gcu 3.6.0+1.0.20260722, NumPy 2.3.5, OpenAI 3.28.0,
pandas 3.0.6, transformers 5.6.0, huggingface-hub 1.33.0 and xgrammar 0.2.8.
The CI stage includes FlagGems-vllm commit
`f775b37f3066e2935742d26c7592a5dd55499159`; setup recognizes its original
GitHub source identity and reuses it without another download.

Attempt 05 uses a fresh container of the corrected image and the normal Bash
entrypoint. It reuses only this task's compiled caches from earlier runs;
package inventories match the earlier candidate except for editable plugin
checkout metadata. Cold timings below belong to the earlier candidate, not
this final warm-cache acceptance attempt.

## Coverage and results

The checked-out workflows and `tests/platforms/enflame.yaml` select ten native
nightly groups: parallel/serial UT, FT, serve benchmark, and inference,
serving and concurrent for both Qwen3.6 models in TP2 eager mode. PR is a
subset; generated weekly and nightly matrices are identical. Existing
platform-specific skips remain visible; no new exclusions were added.

| Check | Corrected-image result (attempt 05 unless specified) |
|---|---|
| Build, runtime ancestry, OCI metadata | Passed |
| Shell syntax, ShellCheck, repository typos check | Passed |
| Compiler and index-fill release guards, device check, CI setup | Passed |
| Fresh-image Bash GCU graph capture/replay and numerical check | Passed, exit 0 |
| Plugin wheel build / separate clean-image install | Passed: 551 installed files, policy YAML and entry points verified |
| Unit, parallel | 1959 passed / 125 console skips |
| Unit, serial | 2 passed / 13 collection skips; 2071 deselected |
| Functional | 71 passed / 29 skipped; all four previously crashing graph tests passed |
| Serve benchmark | Passed, 1 pytest test (86.30s) |
| 27B / 35B-A3B eager inference | Both passed: 189.57s / 353.91s |
| 27B / 35B-A3B eager serving | Both passed, 2 tests each: 339.31s / 260.06s |
| 27B / 35B-A3B eight-request eager concurrent | Both passed: 348.88s / 992.85s |
| Supplemental 35B-A3B graph serving / 27B graph inference | Passed: 35B 2 tests (1242.03s), 27B 1 test (298.92s); both exit 0 |
| Final requested tag and digest | Published and verified; digest above resolves to the tested image ID |

The full native nightly runner exited 0 after 2876.46s (47m 56s), with 10/10 groups passed. Benchmark completed 5 requests with 0 failures.

UT parallel/serial pytest durations were 81.31s / 22.27s; FT was 52.25s.
The parallel console's 125 skips include 112 test skips plus collection
skips. Repeated collection skips across phases must not be summed as distinct
tests. Native runner JSON counts groups, whereas pytest reports individual
tests.

Earlier candidate `fb1c4fe...` at checkout `1af55e7` passed all UT/FT, both eager
inference cases (1506.0s / 1169.5s including cold compilation), both eager
serving cases (2 tests each), benchmark (5 completed requests / 0 failed),
NumPy/Torch/GCU round trip and supplemental 27B graph inference (1 test,
1166.4s). These are retained as baseline evidence, not corrected-image acceptance.

## Necessary corrections and actual failed runs

### Runtime Bash debug fallback

The base's `/etc/bash_env.sh` sources `/etc/profile.d/*.sh`.
`runtime_env.sh` unconditionally sets
`ENFLAME_PT_OP_DEBUG_CONFIG=fallback_cpu=all_ops`. Attempt 02's Python-driven
tests did not source that profile; attempt 03's normal Bash entrypoint did.
The latter had four FT worker aborts with
`topsStreamSynchronize(stream) = topsErrorStreamCaptureUnsupported`.

A controlled identical minimal GCU graph operation on separately verified idle
device 4 reproduced exit 134 with that default. Changing only the variable
before TorchGCU import made capture, replay and numerical checks pass (exit 0).
The corrected image adds a late profile fragment that clears only the exact
inherited debug default, preserving vendor files and the other flags.
A fresh corrected-image probe and all 71 original FT tests then passed.

Attempt 03 also passed both eager inference/serving pairs, UT, benchmark and
27B concurrent. Its remaining 35B concurrent run was interrupted after the
image defect was confirmed, to proceed with corrected-image full CI. Its exit
137 and superseded-run note are retained; no passing 35B result is inferred.

### Concurrent output budget

Both baseline concurrent cases hit the generic 64-token limit while reasoning:
27B stopped before completing `51 + 170 = 221`; 35B stopped before its
Shakespeare answer. Retokenization confirmed the 27B response had exactly
64 tokens. The S60 override now allows 256 tokens. Eight requests,
temperature, prompts, expected strings and all assertions are unchanged.
The original 27B concurrent test passed with this allowance in attempt 03
(430.35s); both corrected-image concurrent cases then passed in the complete nightly run.

### Supplemental graph startup budget

The original 35B graph serving attempt exceeded 120 ten-second readiness
polls while still capturing graphs. It exited 1 after 1239.9s, reporting two
setup errors without reaching serving assertions. Its server log is retained.
The shared graph model YAML now allows 240 polls (40 minutes); successful
startup, engine parameters, prompts and answer checks are unchanged. Other
platforms using this YAML also receive that bounded startup allowance.

The timeout left workers in this task's private container. Nightly preflight
correctly refused occupied devices. The task container was restarted and the
preflight failure was retained separately. Shared containers were untouched.

## Reproduction and evidence

Build with `docker/build.sh --platform enflame --target ci`. This host used
Tencent Ubuntu/PyPI mirrors because direct connections stalled or failed TLS.
Pinned vLLM and FlagGems-vllm source bundles were SHA256 checked and served
through temporary Git mirrors on the Docker bridge. Source commit, wheel hash
and compiler-file checks remained enabled.

Run the native entrypoint under Bash after check/setup:

```bash
python tests/run.py --platform enflame --device s60 --scope all \
  --stage nightly --output-dir /logs/enflame-nightly
```

Documented graph cases are supplemental and use original pytest files with
`FL_TEST_MODEL`, `FL_TEST_CASE`, `FL_TEST_PLATFORM` and `FL_TEST_DEVICE`; see
the [usage guide](../../docs/usage/enflame.md). Filtering the native runner to
an unregistered graph case would select zero tests and return success.
These checks do not establish the separate adaptation Gate capacity scenarios.

On the supplied host, `/data/codex-enflame-vllm-20261104/` retains the build
script/log, `logs/image06-provenance.json`, package inventory, matrix audit,
controlled environment probe, baseline logs and current native JSON/XML reports.
Attempt 05 uses `logs/validation-05-console.log` and `logs/validation-05/`;
35B graph serving uses `logs/validation-08/` with its server log, and
27B graph inference uses `logs/validation-09/`.
Use native runner reports and console logs: the old task driver copied stale
unit-only pytest JSON reports into some other baseline stage directories.

## Dependency metadata limitations

`pip check` still reports these vendor metadata conflicts; they are retained
in `logs/image06-pip-check.log`, not hidden by replacing the vendor compiler:

- xgrammar and triton-gcu declare the upstream `triton` distribution. The
  recipe removes it and verifies the FlagTree-provided `triton` namespace
  against release RECORD hashes (390 files / 2,814,450,573 bytes).
- TorchGCU declares lark 1.1.4; vLLM 0.28 requires lark 1.2.2.
- Inherited enflame-modelopt declares compressed-tensors 0.13.0 and
  transformers 4.56–4.57.6; vLLM uses compressed-tensors 0.17.0 and this stack
  uses transformers 5.6.0. ModelOpt workflows are outside this inference CI
  validation.

The vendor MoE autotuner emits unregistered-dialect diagnostics for some
candidate configurations. These remain in the raw logs; graph capture and the
original serving assertions completed successfully. This does not establish
compatibility for other model/operator configurations.

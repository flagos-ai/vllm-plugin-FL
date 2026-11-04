# Enflame runtime-based vLLM 0.28 validation

Status: in progress; full CI acceptance is pending.

## Image and source

- Required base: `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-enflame-tops1.10.6:2.3.0@sha256:6356cd26647a4cbd27f2ad9ece7281df888df7ec9e9c2dc3bb5aaa262f9097c6`.
- Candidate: `harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.28.0-enflame-ci-candidate-1af55e7-20261104@sha256:a10eb502e815568815730684dbf4dd52bac1f4afda0b98679445ffaf252b3c4f`.
- Local image: `sha256:fb1c4fe912ea092857883706f6a694f11cec0ec76f804a9aea903bacd5d51b9e`.
- Recipe and tested checkout: `1af55e745d21368d0c6e92ce05b34ef32db531a9`, on `codex/enflame-runtime-ci-0.28`, based on `0.4.0-dev` commit `ed8c4801dc1f62b2fc7c5bf01a125a2bd52d2c1b`.
- vLLM source: official v0.28.0 commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`, installed as the empty-device build.
- Ancestry: all 16 required runtime layers were verified as the exact prefix of the candidate's RootFS layers. OCI metadata records the runtime digest, recipe revision and `ci` stage.
- Candidate push and Harbor manifest verification passed. The final requested tag remains unchanged until acceptance.

## Environment

S60/C200 host driver 1.10.6; private container with private PID namespace,
`TOPS_VISIBLE_DEVICES=2,3`, privileged device access and host IPC. Model weights
are mounted read-only. The 27B model has all 15 indexed shards (55.56 GB); the
35B-A3B model has all 26 indexed shards (71.90 GB).

Python 3.12.13 in `/flagos`; torch 2.11.0+cpu, torch-gcu
2.11.0+3.8.20260713, FlagGems 5.4.0, FlagTree 0.7.0+enflame3.6,
triton-gcu 3.6.0+1.0.20260722, NumPy 2.3.5, OpenAI 3.28.0,
pandas 3.0.6, transformers 5.6.0, huggingface-hub 1.33.0 and xgrammar 0.2.8.
The CI stage includes FlagGems-vllm source commit
`f775b37f3066e2935742d26c7592a5dd55499159`; setup recognizes its original
GitHub source identity and reuses it without another source download.

Validation attempt 02 invokes test commands through a Python driver. Its
`ENFLAME_PT_OP_DEBUG_CONFIG` is unset; the runtime's Bash profile would set
`fallback_cpu=all_ops`. Check/setup use Bash and source that profile. Record this
entrypoint difference when comparing any failure with a GitHub Actions Bash job.

## CI selection

The checked-out workflows and `tests/platforms/enflame.yaml` determine coverage.
PR selects inference and serving for both 27B and 35B-A3B, TP2 eager. Nightly
adds concurrent for both models. Generated weekly and nightly matrices are
identical; PR is a subset. All unit and functional tests are included. Benchmark
selects serve smoke using Qwen3-0.6B with dummy weights.

The documented graph cases are supplemental. They must use the original pytest
files directly: filtering `tests/run.py` to an unregistered graph case executes
zero cases and returns success. No new test exclusions or quality relaxations
were introduced for this image.

| Check | Current result |
|---|---|
| Docker build, ancestry, metadata | Passed |
| Shell syntax, ShellCheck, build helper arguments | Passed |
| Compiler guard | Passed: 390 release files, 2,814,450,573 bytes verified |
| Device check and CI setup | Passed |
| NumPy/Torch/GCU round trip | Passed |
| Plugin wheel build and separate clean-container install | Passed; 551 installed files, policy YAML and entry points verified |
| Unit, parallel | 1959 passed / 125 console skips (112 test skips plus collection skips) |
| Unit, serial | 2 passed / 13 collection skips; runner exit 0 for both phases |
| Functional | 71 passed / 29 skipped; runner exit 0 |
| 27B TP2 eager inference | Passed; runner exit 0, 1506s including cold compilation |
| 35B-A3B TP2 eager inference | Passed; runner exit 0, 1169.5s |
| 27B and 35B-A3B eager serving | 27B running; 35B pending |
| 27B and 35B-A3B eager concurrent | Pending |
| Serve benchmark | Pending |
| Supplemental 27B graph inference / 35B graph serving | Pending |
| Final requested tag and digest | Pending |

Unit runner duration was 137.6s; functional runner duration was 111.2s.
Existing skips include NVIDIA-specific migration comparisons and native
CUDA/PPU tests. Skip counts from collection and repeated phases must not be
summed as distinct tests.

## Reproduction and evidence

Build through `docker/build.sh --platform enflame --target ci`; see the recipe
README for optional public-package mirrors. This build used Tencent Ubuntu/PyPI
mirrors because direct external connections stalled or failed TLS. Both vLLM
and FlagGems-vllm came from SHA256-checked Git bundles of the pinned official
commits, served through temporary Git mirrors on the Docker bridge. Package
source identity and commit checks remained enabled.

On the supplied host, `/data/codex-enflame-vllm-20261104/` contains
`build-05.sh`, its complete build log, `logs/image-provenance-final.json`,
`logs/flaggems-vllm-source.json`, package inventory, generated matrices and
`logs/validation-02/<stage>/console.log` plus native runner JSON/XML reports.
The test driver retains separate output directories for each invocation.

## Actual dependency metadata limitations

`pip check` reports the following vendor metadata conflicts; runtime/compiler
validation is recorded separately and these are not hidden by dependency
replacement:

- xgrammar and triton-gcu declare the upstream `triton` distribution. This
  recipe removes that distribution and verifies the FlagTree-provided `triton`
  namespace against its release RECORD hashes.
- TorchGCU declares lark 1.1.4, while vLLM 0.28 requires lark 1.2.2.
- The inherited enflame-modelopt declares compressed-tensors 0.13.0 and
  transformers 4.56–4.57.6; vLLM uses compressed-tensors 0.17.0 and the validated
  stack uses transformers 5.6.0. ModelOpt-specific workflows are not established
  by this vLLM CI validation.

An earlier PR-target revision contained a 35B graph skip for a MoE MLIR
limitation. The updated target `ad8ebc972a7a2b8666796891f14ca81d674618b0`
reverts that skip with `45ebd66`. Neither the baseline validation nor the
updated target relies on it.

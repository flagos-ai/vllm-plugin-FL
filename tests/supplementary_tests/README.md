# Supplementary coverage suite (multi-platform)

This directory holds supplementary coverage tests for `vllm_fl`, targeting
the four vendor platforms enabled on the 0.5.0-dev line:

| Platform | Device | Vendor dir |
|---|---|---|
| kunlunxin | p800 | `vllm_fl/dispatch/backends/vendor/kunlunxin/` |
| hygon | bw1000 | `vllm_fl/dispatch/backends/vendor/hygon/` |
| iluvatar | bi_v150 | `vllm_fl/dispatch/backends/vendor/iluvatar/` |
| thead | zw810e | `vllm_fl/dispatch/backends/vendor/thead/` |

## Isolation contract

* Core entry points never collect this tree: `tests/run.py --scope unit`
  pins `tests/unit_tests/` and `--scope functional` pins
  `tests/functional_tests/`.
* Every test here carries the `supplementary` marker (registered locally in
  this directory's `conftest.py` — the root conftest and `pyproject.toml`
  are untouched).
* Vendor-specific tests carry `pytest.mark.platforms("...")` so a run for
  platform A skips (not fails) platform B's gated tests.
* Tests must not depend on vendor hardware by construction: CPU stub tests
  are platform-agnostic; real-device tests are marked `gpu`/`multi_gpu` and
  auto-skip on accelerator-less hosts.

## Running

Per platform (mirrors the core runner):

```bash
python tests/supplementary_tests/run.py --platform kunlunxin --device p800
python tests/supplementary_tests/run.py --platform hygon --device bw1000
python tests/supplementary_tests/run.py --platform iluvatar --device bi_v150
python tests/supplementary_tests/run.py --platform thead --device zw810e
```

Coverage output is `coverage-<platform>-supp.json` (isolated
`.coverage-<platform>-supp` data file per platform).

Manual collection check:

```bash
python -m pytest tests/supplementary_tests --collect-only -q
```

Bare pytest only collects when the platform gates resolve — prefer the
runner, which injects `--platform/--device` and `FL_TEST_PLATFORM`.

## CI

Executed by `.github/workflows/coverage_supplementary_multi.yml` —
`workflow_dispatch` + a Saturday 06:00 UTC schedule, one matrix leg per
platform on that platform's real runner (kunlunxin-p800 /
hg-cicd-vllm-plugin-0240 / iluvatar-cicd-vllm-plugin / flagcicd-810e).
The daily PR CI is untouched.

## What must NOT change here

Core test files (`tests/unit_tests/**`, `tests/functional_tests/**`,
`tests/e2e_tests/**`), `tests/run.py`, `tests/conftest.py`,
`tests/platforms/*.yaml`, `pyproject.toml` — this suite only adds files.

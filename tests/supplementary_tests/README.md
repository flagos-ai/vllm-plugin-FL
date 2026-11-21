# Supplementary coverage tests

This directory holds the **supplementary coverage suite** for `vllm_fl` on
the `0.5.0-dev` line. It complements — never replaces — the core suites.

## What lives here

Tests that close coverage gaps found by comparing the core baseline
(`tests/unit_tests` + `tests/functional_tests`, measured with pytest-cov)
against the `vllm_fl` source tree. Two categories:

* **CPU unit tests** — pure-Python / torch-CPU logic that the core suites
  leave uncovered (env var handling, import fallbacks, registries, policy
  branches, serialization helpers, ...). These are fast and safe.
* **GPU functional tests** — small-input numeric tests that exercise real
  kernels or execution paths (attention, fused MoE, graph capture, FlagCX
  KV transfer, ...). These are expensive and only run on an A100 runner.

Everything here is marked `@pytest.mark.supplementary` (registered by this
directory's `conftest.py`) and organized into `unit/<area>/` and
`functional/<area>/` subdirectories mirroring `vllm_fl`'s layout.

## What must NOT be modified

* existing tests anywhere under `tests/unit_tests`, `tests/functional_tests`,
  `tests/e2e_tests`, `tests/benchmarks` — supplementary coverage is added as
  new files only;
* `tests/run.py`, `tests/conftest.py`, `tests/platforms/*.yaml`,
  `pyproject.toml` coverage/pytest sections — the core pipelines stay
  byte-identical.

## How to run

```bash
# full supplementary suite (parallel + serial passes), coverage to
# coverage-<platform>-supp.json
python tests/supplementary_tests/run.py --platform nvidia --device a100

# single file while developing (Windows-local: add -p no:asyncio)
python -m pytest tests/supplementary_tests/unit/dispatch/test_xxx.py \
    -q -p no:cacheprovider
```

## Cost model

| class | when it runs | cost |
|---|---|---|
| core (unit_tests/functional_tests) | every PR | minutes |
| supplementary CPU (this dir, `unit/`) | weekly/idle windows | ~10 min |
| supplementary GPU (this dir, `functional/`) | weekly/idle windows on A100 | ~1 h |

See `.github/workflows/coverage_supplementary.yml` for the idle-time
execution and the baseline/combined report generation.

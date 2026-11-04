# PR #675 CI repair evidence

## Source failures

Inspected the complete logs for run `41616100214`, head
`2abe629b90375ff84413374ac8a9c0657fc70453`, before editing.

- Enflame job `125850780188`: setup exits 1 before pytest. The pinned
  `v0.28.0-enflame-ci` image contains `vllm==0.28.0+empty`; setup requires
  `0.32.0+empty`. No Enflame test passed in this job. E2E, functional and
  benchmark jobs were skipped.
- CUDA job `125850771491`: exits 1, with 19 failed, 1966 passed and 72 skipped.
  Python 3.12.3, PyTorch 2.13.0+cu129, vLLM 0.32.0+cu129,
  FlagTree 0.7.0+git317a0249 and FlagGems 5.4.0+g6fe7709e7717.
  Failures comprise eight MUSA fixture failures, two schema-ownership fixture
  failures, four TXDA warmup fixture failures, two runner import failures and
  three GCU attention import failures on a host without Enflame FlashAttention.

## Changes

- Supply the existing Metax compatibility hook in the AST-isolated MUSA fixture.
- Explicitly model native schema ownership in schema-registry tests, independent
  of CUDA schemas already loaded into the process.
- Supply v0.32 worker fields and compiled-model detection to the TXDA fixture;
  retain its warmup and workspace assertions, including randomization arguments.
- Defer the runner's stream type annotation so importing it does not dereference
  an uninitialized platform's device API.
- Isolate the optional native FlashAttention import in GCU interface tests on
  non-vendor hosts. An unexpected native-kernel call raises instead of silently
  succeeding. The real kernel remains in use when available; production GCU
  attention code and precision thresholds are unchanged.

## Local validation

On the available S60 environment, after the final code changes:

- The four affected test files: **72 passed**, two warnings, 13.67 seconds.
- A controlled missing-GCU-kernel import experiment: **3 passed**, six deselected.
- `pytest -q tests/unit_tests`: blocked during collection because `torch_npu`
  is unavailable. This is not recorded as a passing full suite.
- With only `tests/unit_tests/dispatch/test_ascend_vocab_parallel_embedding.py`
  explicitly excluded: **1956 passed, 232 skipped**, four warnings, 66.48 seconds.
  This local exclusion does not change repository tests or CI configuration.
- Ruff check, Ruff format check on all five changed Python files, and
  `git diff --check`: passed.

These are local results, not a substitute for the new CUDA or Enflame CI jobs.

## Remaining Enflame infrastructure blocker

An authenticated anonymous-pull Harbor manifest probe returned HTTP 404 for
`plugin/vllm-plugin-fl:v0.32.0-enflame-ci`. The existing v0.28.0 tag returned
HTTP 200 with digest
`sha256:f21e53101ffcf00fa5b70b39913bd401be93c0d1834c0ae8d07684bf0690bc88`.
This environment has neither Docker nor a Docker socket. A real v0.32.0 Enflame
CI image must be built, published and verified before its digest can replace the
old pin. The strict version gate remains intact; no nonexistent image tag or
relaxed version assertion is introduced to conceal the failure.

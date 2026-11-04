# PR #675 CI repair evidence

## Enflame image publication audit

Registry observations on 2026-11-04 UTC confirm an external publication
blocker, not a test-code failure that can be repaired by relaxing setup:

- Full log of run `41755861468`, job `126332538953`: the pinned 0.28 image
  pulls successfully; device check finds two GCUs and completes tensor smoke.
  Setup then raises `RuntimeError: Expected vLLM 0.32.0 empty build, found
  0.28.0+empty` and exits **1**, before installing the checkout or running pytest.
- A scoped anonymous-pull token manifest request returns **404** for
  `plugin/vllm-plugin-fl:v0.32.0-enflame-ci`. The old 0.28 tag returns **200**
  with the `f21e5310...` digest recorded below. The repository's tag listing
  returns 23 tags with no pagination link; its only Enflame tags are three
  0.28 variants. No published 0.32 Enflame candidate was found in this repository.
- `.github/configs/platforms.yml` enables Enflame. The image pin in
  `.github/configs/enflame.yml` is still the verified historical 0.28 digest.
  The setup script and `docker/enflame/verify_compiler.py` /
  `docker/enflame/verify_index_fill.py` paths exist. The build helper uses
  `docker/enflame` as context, matching the Dockerfile's `COPY` paths.
  No erroneous CI path was found; no workflow, image pin or gate was changed.

### Required base and access

The exact `ENFLAME_BASE_IMAGE` default in `docker/enflame/Dockerfile` is:

```text
harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-enflame-gems_5.4.0.dev0-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.8-torch_gcu_2.11.0_3.8.20260713-pcp_tops3.8.20260714-gpu_s60-arc_amd64-driver_1.9.10:202608141853
```

The scoped anonymous manifest request for this base returns **401 Unauthorized**.
Its manifest digest is **unknown**, not inferred from the old CI image digest.
An authorized maintainer must resolve and record the base's registry digest
and make that exact base available to the builder. A 401 does not establish
whether the private tag exists. This workspace has no Docker executable or
Docker socket, so neither a container build nor candidate validation ran here.

Unblocking requires an amd64 Docker builder, private-base **pull** permission,
network access to the pinned source/wheel endpoints, **push** permission to
`harbor.baai.ac.cn/plugin/vllm-plugin-fl`, and an available S60 validation host
with the documented driver/runtime and two allocated devices. Credentials
belong in the builder's secret store, not in this report or a PR log.

### Build, publish and acceptance handoff

From the tested checkout, after resolving the authorized base digest:

```bash
docker/build.sh --platform enflame --target ci \
  --image-name harbor.baai.ac.cn/plugin/vllm-plugin-fl \
  --image-tag v0.32.0-enflame-ci
```

This builds locally and does not push. Keep the build log and the resolved
base digest; pin `ENFLAME_BASE_IMAGE` to that verified digest through
`--build-arg` when invoking the helper. Do not retag a 0.28 image. In a container
from the new candidate, mount this checkout and run the unchanged
`.github/scripts/enflame/check.sh`, `.github/scripts/enflame/setup.sh` and
`python tests/run.py --platform enflame --scope unit` on S60, with the configured
device/environment settings. Setup must verify 0.32.0+empty and all pinned
vendor packages, compiler files and imports without weakening any assertion.
Retain exit codes and full logs, then publish the validated image:

```bash
docker push harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-enflame-ci
docker pull harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-enflame-ci
docker image inspect --format '{{json .RepoDigests}}' \
  harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-enflame-ci
```

Confirm the registry manifest digest and a pull by that digest from the CI
runner. Only then update `ci_image` to the real 0.32 tag plus verified digest
and rerun CI, including downstream functional/e2e/benchmark jobs. A build,
metadata check or skipped job is not evidence those hardware suites passed.
No new digest is supplied here because no replacement image was verified.

## Follow-up: remaining Stream signature annotations

The reported CUDA job `126332517747` in run `41755861468` still failed
`test_model_runner_imports_against_target_vllm` and
`test_compiled_gcu_dispatch_warmup` at the pooling output constructor's eager
Stream annotation. A whole-file audit found two remaining import-time lookups:
the pooling output parameter and the async copy stream helper's return type.
Both now use the same deferred string expression as the previously fixed
constructor. Runtime stream creation and function-local annotations are unchanged.

A new AST-based regression evaluates every matching function signature with
`torch_device_fn=None`, then resolves its deferred expression against a Stream
class to check type semantics. It failed before the fix with `NoneType.Stream`.
After the fix, the four related test files pass: **73 passed**, two warnings,
13.70 seconds, on the available S60 environment. A separate invocation of the
two reported CUDA failures plus the new regression also passes. Ruff check,
Ruff format check and `git diff --check` pass. These local observations are not
a claim that the new remote CUDA CI job has passed.

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

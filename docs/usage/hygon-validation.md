# Hygon Validation Record

This record keeps validation results bound to their original source revisions,
images, and wheel artifacts. Use the [deployment guide](hygon.md) for the
current serving recipe. Historical counts below are not results for a newer
checkout or an exact-artifact retest of the published image.

## Current PR CI

For revision `f83325b88cbea3aecb71a75def61e03e68e4da11`, the Hygon unit,
functional, inference, serving, and benchmark jobs succeeded in
[CI run 38841537036](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/38841537036).
This records the Hygon jobs, not the overall multi-platform workflow result.
The shared runner uses the pinned image and installs an ordinary plugin wheel
from the checked-out revision. The historical counts below belong to their
separately identified runs.

## Published image and package provenance

| Artifact | Identity |
|---|---|
| Deployment image manifest | `sha256:e5a84bc28e4bfcae413e1e9288a605a49aea352ecc8b35f4b58b3ebba5e6fc9d` |
| Image build source | `544e9cb2484d6bd0d679dcbedb7c9d9621730a09` |
| Image plugin wheel | `0.0.0+g544e9cb2484d6bd0d679dcbedb7c9d9621730a09` |
| Empty vLLM source | `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| FlagGems matching source | [e12a220a0d54b40d90d0549ff83ffabedb01f6dd](https://github.com/flagos-ai/FlagGems/commit/e12a220a0d54b40d90d0549ff83ffabedb01f6dd) |
| FlagTree distribution | `0.6.0+hcu3.6` |
| FlagTree wheel SHA256 | `0fd81f0e691486dfaf85f6a196f9f09d66118b7d2064b4f8c70b6c8bfdb4d52c` |

FlagGems distribution metadata reports `0.1.0`, which does not identify its
source revision. All 1,944 installed source/package files match the official
commit above after accounting for CRLF/LF line-ending differences. The installed
`direct_url.json` contains no VCS revision; the commit identifies the matching
package source content rather than a recorded installation revision.
FlagTree is identified by its distribution version and wheel SHA256 above;
its full Git commit was not recorded. Triton's `gitc73250c4` is not a FlagTree
commit.

## Historical setup CI validation

The published image above was built and tested from commit
`544e9cb2484d6bd0d679dcbedb7c9d9621730a09`. CI setup rebuilt and replaced
the image's plugin wheel with an ordinary wheel from that same commit. Its SCM
metadata is `0.0.0+g544e9cb24`; the image build uses the full-SHA version listed
above. These are different wheel artifacts. The results below validate the
environment after setup, rather than byte-for-byte acceptance of the original
image plugin wheel. Setup preserved the vendor runtime, and the verified
packages were noneditable. A separate ordinary import loaded `WorkerFL` from
the image's Python environment with the V2 model runner disabled.

| CI scope | Actual result |
|---|---|
| Unit | 550 passed, 16 skipped |
| Functional | 23 passed, 0 skipped |
| Benchmark smoke | 1 pytest test passed, 0 skipped; 5 requests completed, 0 failed |

All setup and test commands exited successfully. Unit JUnit serializes 566
`<testcase>` nodes. Its suite count is 586 because it also counts 20 passing
`unittest.subTest` reports. Skipped tests are not counted as passed.

Distributed unit checks use CPU references and mocks. These CI results verify
the stated test scopes; they do not establish a complete collective ABI or
performance result. The benchmark uses TP1, eager execution, and
`--load-format dummy` for five requests; it establishes no model-quality or
performance conclusion. The historical 104-request model matrix below is a
separate validation run.

## Shared-runner model validation

A separate shared-runner run used source commit
`672191c7bf73158272a03c82be60332a7c61c959` and image manifest
`sha256:c697e7b71e5924c434503e26bce415241965aad1a7b5e03eda4b0b07466f5a35`.
The committed production files under `vllm_fl`, plus `pyproject.toml` and
`setup.py`, have no differences from commit `544e9cb2484d6bd0d679dcbedb7c9d9621730a09`.
The image and wheel artifacts differ, so these TP2 results are not an exact-artifact
model test of the final published image. Both groups exited successfully with
two passed cases and no failed cases. For each serving model, `model_list`
and `chat` were the two pytest items; chat covered a Paris text assertion and
generated-image assertions for `Hello VLM` and a blue rectangle.

| Group | Model | Mode | Result |
|---|---|---|---|
| Inference | Qwen3.6-35B-A3B | Graph, TP2 | Passed |
| Inference | Qwen3.6-27B | Eager, TP2 | Passed |
| Serving | Qwen3.6-27B | Graph, TP2 | Passed |
| Serving | Qwen3.6-35B-A3B | Eager, TP2 | Passed |

## Historical model validation

The model matrix below was completed before the CI image was published, using
plugin commit `04ae0f0906e97d11a4d1f5a9d4a01e87487416d5`. On that two-card
stack, both Qwen3.6 models passed the original text,
image, concurrent text/image, and mixed-request cases in eager and graph modes:

| Model | Eager | Graph |
|---|---|---|
| Qwen3.6-27B | 26/26 requests | 26/26 requests |
| Qwen3.6-35B-A3B | 26/26 requests | 26/26 requests |

The four historical runs cover 20 scenarios and 104 requests. The 52 eager
requests used the exact committed attention source as a source overlay; the
52 graph requests used the ordinary plugin wheel `0.0.0+g04ae0f090`.
Twenty native profiler traces were validated across the two graph runs,
including HIP capture begin/end and graph launches. Capture traces were from
rank 0; runtime traces verified replay on both TP ranks. These checks establish
this model/configuration scope; they are not a throughput benchmark or a
complete operator/collective ABI test. This historical matrix does not report
the separate unit, functional, or E2E CI acceptance results for the published image.

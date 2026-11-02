# Enflame S60: vLLM 0.32.0 adaptation status

This branch targets official vLLM `v0.32.0`, commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`, installed with
`VLLM_TARGET_DEVICE=empty`. The existing Enflame CI image and the deployment
instructions in `enflame.md` describe vLLM 0.28. They are not a validated
0.32 dependency image. Do not relabel that image as a 0.32 image.

The Qwen3-0.6B BF16 checks on S60 have completed eager generation, TP=2
with a real FlagCX collective, and eager/Graph HTTP concurrency and streaming.
A five-prompt greedy comparison (32 tokens per prompt) matched all 160 tokens
between eager and Graph when the rounding option below was enabled. These
checks compare S60 execution modes. Official NVIDIA vLLM 0.32.0+cu129
reference generation has now completed on RTX 4090 with identical, SHA-256
verified model files. Strict NVIDIA alignment failed: arithmetic first differs
at token 7 and code at token 13 (1-based); the other three prompts match all
32 tokens. NVIDIA V1 and V2 runners match all 160 tokens, so changing the
reference runner does not eliminate these differences.

The current working-tree GCU RMSNorm candidate changes normalization rounding.
With the original BF16 head, arithmetic still differs at token 7 and Paris newly
differs at token 11; code now matches. These differences are recorded below under the user's precision acceptance.
With the native FP32 head, the candidate matches all 160 tokens on the original
five prompts, but only eight of ten additional prompts match (Japan first differs
at token 14, JSON at token 16). Changing the head configuration does not replace
the original BF16 comparison. The user explicitly accepted the observed numerical
precision differences and ended further precision debugging. This is acceptance
with known differences, not a strict token-match pass. No CPU FP64 diagnostic is
enabled in production. Identical-input replay on official NVIDIA fused RMSNorm
matches the independently registered GCU RMS result at the investigated boundary;
the CPU FP64 value is not the CUDA reference. These checks do not establish support
for other model architectures.

## Compiled execution

The plugin preserves integer pointer types during GCU Inductor code generation.
The vendor's global int64-to-int32 source replacement can otherwise corrupt
lookup addresses without converting the underlying tensor storage. Compiler
and vLLM cache keys distinguish the corrected generated code.

BF16 compiled execution also requires the following setting for the verified
Qwen3-0.6B comparison:

```bash
export TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
```

Without that setting, default compiled generation diverged from eager.
It is an environment option, not a production default installed by the plugin.
Retain it for the tested configuration and independently verify precision for
each deployment. Successful compilation or HTTP readiness does not verify
output correctness.

## Outstanding acceptance

NVIDIA reference generation is complete. Strict token alignment failed as described
above; the user accepted these precision differences, so further precision debugging
is closed for this Qwen3-0.6B task. The broader adaptation-gate matrix and 0.32 CI
image validation remain unverified and are not covered by that acceptance. The Dockerfile now pins the official
0.32.0 commit and CI setup requires `0.32.0+empty`; a new image has not been
built or published. Actual setup passed package checks, plugin and pinned
FlagGems-vllm installation, compiler verification, imports, and device checks.
The first GitHub fetch failed with a GnuTLS connection termination; the passing
run used a shallow official clone as transport after verifying the exact
pinned commit. The CI image configuration still references the historical
0.28 digest and requires a newly built and validated 0.32 image. Full unit execution also encountered
an unavailable optional `torch_npu` dependency during collection; supporting
passes do not constitute a complete suite pass.

Keep the vendor PyTorch, FlagTree, and FlagGems packages pinned when installing
an empty vLLM build. Install the plugin with `--no-build-isolation --no-deps`
to avoid replacing the vendor runtime. A release requires the same dependency
stack in CI and the deployed environment, plus completed precision and gate
results on the submitted commit.

## Qwen3.6 TP2 gate fixes

The ongoing 27B/35B-A3B eager gate exposed two additional v0.32 compatibility
issues. GCU vocabulary masking now uses a vendor-scoped Triton `tl.where`
kernel instead of unsupported boolean-to-int64 arithmetic; native int64 eager
comparison is not a fallback on this runtime. Softmax MoE routing retains FL
dispatch with padding and sets padded expert IDs to -1, matching the upstream
contract without invoking CUDA-only `_moe_C` operators. Sigmoid routing is
unchanged.

The vocabulary mask preserves upstream int64 output for both int32 and int64
inputs. Its expanded real-GCU CPU-oracle stress test passed 24 cases, including
empty inputs, shard padding and large int64 IDs. The combined mask/router unit
run passed 19 tests. Router device checks also passed across both supported ID
dtypes and renormalization modes.

Qwen3.6-27B TP2 eager completed all five gate scenarios: 26/26 requests passed,
with runner exit 0. The 35B-A3B eager single-text case failed the unchanged
keyword assertion because `limitations` was absent; no hardware root cause is
established. The other 25 requests passed, so the complete 35B-A3B eager
run is 25/26, not an accepted gate pass.

The first 27B Graph startup exposed a missing collective custom-op boundary:
Dynamo traced Python stream handling in FlagCX. Enflame/GCU now uses the
upstream registered collective custom ops. Eight new policy/fullgraph regression
tests and ten compilation-config tests passed. Subsequent real-device startup
failed in a layout-sensitive vendor BF16 linker relocation. The GCU compiler
adapter now rejects only that exact known candidate failure through Torch's
existing autotuner exception contract; unknown failures and exhausted candidate
sets still fail. Seventy focused regression tests passed. A real-GCU replay of
the original kernel and vendor-generated configs compiled three of four layouts,
and all three matched a nonzero CPU oracle including negative indices. Precision
emulation remains enabled. The subsequent 27B TP2 Graph run completed FULL
decode capture (35/35 shapes) and passed all five scenarios, 26/26 requests.
The 35B Graph run still exhausted all candidates for a native compiled SiLU
in its shared expert. An actual-worker probe verified that explicit compilation
configuration `{"custom_ops":["none","+SiluAndMul"]}` selects the FL SiLU
implementation and passes three nonzero GCU checks. The uppercase registered
name is required; `+silu_and_mul` does not enable this FL class. A 35B Graph
retry with that configuration completed PIECEWISE and FULL decode capture
(35/35 FULL shapes), reached application startup complete, and returned HTTP 200
from its health endpoint. The full request gate subsequently passed all five
scenarios: 26/26 requests, runner exit 0.
A minimal probe isolated the repeated TLE MLIR parse failure; the existing
four-stage MoE alignment fallback passed real-GCU token/expert/padding checks
for 1, 8 and 512 rows. No production MoE alignment workaround was added.
The final focused regression run passed 107 tests.
See the dated logs and request-level results in
`/workspace/s60-adaptation/STATUS.md`. The 27B eager and Graph passes do not
establish complete backend acceptance.


### CI configuration and remaining acceptance

The S60 overlay in `tests/platforms/enflame.yaml` carries explicit
`compilation_config.custom_ops: ["none", "+SiluAndMul"]` for its four
Qwen3.6 TP2 cases. Environment defaults preserve precision emulation with
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`. Shared CUDA model cases are unchanged.
The model-config loader preserves the dictionary for offline inference and emits
valid JSON for the serving CLI. Thirteen configuration regressions passed,
including nested JSON and both invocation paths for all four S60 cases. The
combined final regression run passed 115 tests (`logs/unit-delivery-final.log`).

Across the four real-device groups, 103/104 requests passed. After disclosure,
the user explicitly accepted the specific 35B-A3B eager single-text exception.
Its original keyword assertion remains FAIL; this is not a 104/104 result or
a proven hardware defect.
Docker image build/publication and runner CI remain unverified. An isolated
Docker 28.5.2 daemon starts successfully, but pulling the exact required Harbor
base image is denied by repository authorization. The user has no Harbor account
and explicitly authorized skipping this pull and the dependent image build.
These steps are SKIPPED, not PASS; no further credentials are requested.
Image publication and runner CI remain unverified. Backend CI remains disabled
until an authorized real v0.32 image is built and verified. The build entrypoint
now defaults to vLLM 0.32.0, matching the Dockerfile; seven focused version and
argument-forwarding tests pass. A fresh 35B eager single-text run using the CI FL
SiLU configuration still fails the unchanged semantic/order checks because its
fluent answer omits the required literal `limitations`. User acceptance of the
recorded 0.6B numerical differences is separate from the subsequent explicit
acceptance of this disclosed single-text exception; Docker/CI status is unchanged.

A controlled diagnostic submitted two copies of that unchanged single-text
request concurrently to the same eager endpoint. Both responses were identical
to each other, differed from the recorded single-request response, and passed
all ten original checks. This demonstrates execution-context-sensitive output,
not a proven hardware defect or a single-request acceptance pass. The matrix
remains 103/104; diagnostic concurrency must not replace the original gate.

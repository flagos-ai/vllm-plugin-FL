# GLM5-Next vLLM 0.24 review iteration

This iteration addresses the review of `af0e829e034ddab5f647ab6ffbbd649812ed95ad`.
The supported loading contract is an unquantized checkpoint, PP1, no speculative
decoding, and no EPLB. TP and ordinary EP remain available. Quantized checkpoint
storage is distinct from the FP8 indexer KV cache; the latter remains supported.

| Review finding | Result |
| --- | --- |
| Portable MLA logger lacks `info_once` | Use the vLLM logger; exercise active-plan selection in fresh processes. |
| GLM provider influences generic MLA dispatch | Generic FlagGems candidate declines MLA; only the GLM plan selects the portable implementation. Tests cross the real dispatcher and vendor/FlagGems methods. |
| Incomplete mHC PP state | Reject PP > 1 before hybrid config adaptation and again at final validation/model construction. |
| KDA speculative state and flattened KPool request grouping | Reject every speculative configuration at the same boundaries; no speculative support is claimed. |
| FP8 projection loading and KDA quantization exclusion | Reject quantization configs and undeclared non-floating/FP8 checkpoint tensors before ordinary loading. FP8 checkpoint conversion is deferred. |
| Incomplete EPLB outer-model interface | Reject EPLB; ordinary EP is a separate mode. |
| Missing packed quantization mapping | Declare GLM gate/up mapping on both public model wrappers; this does not enable quantized loading. |
| Parameter-name-only weight audit | Additionally record successful destination/shard/local-expert loader calls; reject missing/duplicate packed slices and restore loaders on failure. Scope the audit to the complete checkpoint, including interleaved text/head/vision prefix groups. |
| MQA per-token scale interpreted as groups | Normalize exact per-vector shapes; reject incompatible scale shapes. Force missing/rejecting FlagGems fallback with N=128/512/2051. |
| Import-time public vision FlashAttention replacement | Remove all global assignments; private GLM custom op consumes FA2 output and leaves public tuple/LSE contracts untouched. |
| Whole-pool KV repacking | Read page bytes and scale offsets directly, respecting padded physical strides. |
| Existing processor tests | Correct dense-layer defaults for small configs; retain legacy last frame 295. Keep subsecond clips nonempty. |
| Integrated clamped MoE NameError | Resolve the runtime platform before both clamp and fused paths; retain clamp behavior and cover ROCm/unknown priority selection. |

Validation uses vLLM 0.24.0, Torch 2.11.0+cu129, Transformers 5.12.1 and
FlagGems 5.3.3.dev15+gf471641e5.pkgfix1. The CPU suite passed 718 tests (7 GPU
cases deselected). Installed-wheel validation passed 52 tests, including six H100 operator cases,
including updated block-table/context replay, padded page strides, and
FP16/BF16 variable-length vision attention. These checks do not by themselves
establish end-to-end multimodal quality or full-model graph support.

A wheel built from `d83b7f978338e495a475c441c510593c9377c697` also passed
BF16 TP16/EP startup with all 47 checkpoint shards and 23 targeted serving
requests: short arithmetic, mixed 12,122/14,322/18,722/23,122-token retrieval,
16 concurrent independent codes, one image and a 0.4-second video. Every
response finished with `stop` and matched its expected content; isolation
responses contained only their own request code. Both service installations
matched all 212 wheel source hashes, and neither node logged an inference
error. These are smoke checks, not a full quality or maximum-capacity test;
startup used `--skip-mm-profiling`, and full GPQA was not rerun.

The sparse MLA graph opt-in remains off. Historical GPQA and throughput
measurements belong to their original source snapshots; they are not reruns of
this iteration. Cross-vendor tests cover dispatch contracts, not accelerator
numerical parity.

## Operator handoff

| Item | Owner for follow-up | Contract and evidence | Remaining scope |
| --- | --- | --- | --- |
| `kernels/glm5_next/paged_mqa.py` | FlagGems | FP8 page-stride read, per-token FP32 scales, unchanged dot/ReLU/head reduction; `test_paged_mqa_stride.py` covers page boundaries, padding and metadata replay | Upstream page-stride/scale-offset interface; vendor numerical/performance qualification |
| `kernels/glm5_next/vision_attention.py` | FlagTree + FlagGems | Model-private composition of existing FA2, fake implementation and static sequence bound; FP16/BF16 replay tests | Verify complete vision encoder compile/graph path and dynamic image/video generation workloads |

At that snapshot, architectural follow-ups included explicit provider bindings,
temporary MLA-constructor capability patches and compressed-page layout
ownership. The follow-up below addresses bindings and layout ownership, and
records the remaining constructor boundary. Full multimodal quality remains
a separate release check.

## Paged MQA measurement

H100, one active request at 2,049 tokens, 32-token pages, 25 warmups and 100
CUDA-event samples of the complete eager wrapper. Baseline is the full-pool
repack at `af0e829`; candidate and baseline both pass the FP32 PyTorch reference
at `atol=rtol=2e-5`.

| Physical pages | Baseline median (us) | Candidate median (us) | Speedup | Baseline extra peak bytes | Candidate extra peak bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 79.17 | 57.68 | 1.37x | 549,888 | 9,216 |
| 4,096 | 80.32 | 58.32 | 1.38x | 17,310,720 | 9,216 |
| 16,384 | 185.22 | 57.82 | 3.20x | 69,215,232 | 9,216 |

These are operator-wrapper measurements, not model-serving speedups. The
PyTorch reference medians were 102.11/101.07/107.79 us. Compute Sanitizer was
unavailable in the validation image, so no sanitizer pass is claimed.


## Follow-up review of 79e5dab

The follow-up unifies deployment pixel/token ceilings with actual image/video
preprocessing. `max_pixels` takes precedence over `max_image_tokens`; requests
may lower the deployment ceiling but cannot raise it. Geometry overrides and
`do_resize=False` are rejected. Profiling uses a maximal aligned image canvas
and a video canvas derived from its frame and pixel budgets. Video prompt
updates retain each request's sampling options without mutating cached state.
When vLLM's media loader has already sampled a video, request and deployment
sampling select a subset of those decoded frames. Pixel preparation and prompt
timestamps share the mapping back to source frame IDs. Profiling explicitly
keeps the maximal dummy frame count, including with a low deployment FPS.
Both sampler formats retain a nonempty temporal group for short clips.

Indexer and portable vision adapters now bind through the public selection
policy. Per-op order, strict mode, vendor filters and FlagGems allow/deny lists
apply to those calls. Only `NotImplementedError` permits fallback; failed
implementations are removed from the binding and GPU errors propagate.
Required vision implementations are preflighted before weight loading.

The GLM indexer backend owns its physical layout descriptor. Runner reshaping,
metadata creation and compressed-cache zeroing consume that capability;
other compressed backends keep their existing path. The generic DeepSeek
indexer's capability declaration stays unchanged.

Early engine/config hooks, worker hooks, the temporary MLA constructor hooks
and the FlagGems TLE compatibility hooks use the same patch ownership records.
`patch_inventory()` includes their phase, fingerprint and vLLM version.
Constructor hooks are restored on success or failure. They remain
process-global during single-model construction; replacing that compatibility
boundary with explicit upstream factories remains follow-up work. Early
registration hooks delegate unless the config or cache specs identify GLM.
Provider validation happens only after the model matches.

The follow-up is merged with target main `940480f`. Code candidate `a605ab6`
passes 809 CPU tests (4 skipped, 7 GPU cases deselected) and 163 installed-wheel
tests on each of two H100 nodes, including six GPU cases per node. All 277
package files match the candidate wheel. The native vLLM Python installation
matches its wheel RECORD on both nodes. The preceding section's serving
counts belong to the earlier wheel.

The same `a605ab6` wheel passes BF16 TP16/EP serving with normal multimodal
profiling enabled, without `--skip-mm-profiling`. Both the default image budget
and the deployment override `images_kwargs.max_image_tokens=16000` complete
19 piecewise and 11 full-decode graph captures, with breakable graphs enabled.

| Configuration | Targeted checks | Result |
| --- | --- | --- |
| GLM default image budget | Arithmetic, four mixed 12,122–23,122-token prompts, 16 independent concurrent codes, image caps/precedence/rejection, short and request-sampled video | 27/27 |
| GLM deployment image ceiling 16000 | Arithmetic, 4096-square image, lower request cap, pixel precedence, over-budget rejection and both video cases | 7/7 |

Every successful response finishes with `stop` and matches the expected answer;
the isolation audit checks that each response contains only its own code.
The over-budget image cases return the expected HTTP 400. Neither GLM run logs
a CUDA failure or OOM. These are targeted functional checks; full GPQA and
maximum-capacity measurements were not rerun.

A Qwen3-30B-A3B BF16 TP2 control starts with the deliberately invalid
`VLLM_FL_GLM5_PROVIDER=invalid`, platform-default FlagGems policy and ordinary
graphs (five piecewise and four full-decode captures). First-use records select
FlagGems attention and `topk_softmax`, without a logged fallback. All 16
independent-code requests pass at client concurrency eight. However, its first
arithmetic request at temperature 1 returns an unrelated image URL: that run
is 16/17. A subsequent greedy run on the same service passes 17/17, and
fixed-seed arithmetic probes also pass. The original failure is retained;
its cause is not established by these checks.

A separate cold start of the same Qwen wheel with `VLLM_FL_PREFER=vendor`
passes 17/17 at temperature 1, retaining the invalid GLM provider and ordinary
graph configuration. Attention and `topk_softmax` select `vendor.cuda`, without
a logged fallback. This validates that non-GLM startup can ignore the GLM-only
setting; it does not establish the cause of the earlier sampled-answer anomaly
or numerical equivalence between providers.

The real checkpoint uses the legacy pixel schema with patch expansion: its
effective default vision budget is 800 tokens. Real checkpoint preprocessing
produces 784 vision tokens for a 4096-square image by default, and 15876 with
a deployment image ceiling of 16000. The synthetic token-schema tests use a
different default of 8000. The real video dummy uses 400 frames and 800 vision
tokens; its 2006 prompt tokens remain below the declared 2202-token bound.

Two additional regressions were fixed during startup validation: both lazy
and already-initialized KV registry paths preserve the registration argument
contract, and the private tail backend explicitly advertises strided padded
pages. The generic DeepSeek backend remains unchanged.

The validated GLM deployment retains the historical restricted FlagGems
operator set and also permits the policy-governed private vision operator:

```bash
export VLLM_FL_FLAGOS_WHITELIST=grouped_topk,moe_sum,flash_attn_varlen_func
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
```

Under the tested `auto` provider, NVIDIA native implementations handle the
indexer and vision attention; observed `grouped_topk` and `moe_sum` calls select
FlagGems. The private FA2 path has separate operator-level GPU coverage. When
that private adapter is selected, omitting `flash_attn_varlen_func` from an
explicit whitelist correctly rejects it during preflight. On earlier candidate
`389b737`, the broader platform-default FlagGems operator set passed full TP16
startup and graph capture, but four concurrent long prompts triggered an illegal
memory access.
A synchronous diagnostic reproduced the failure inside graph replay; the
specific kernel has not been identified. That broader operator configuration
is not validated by the restricted-deployment checks. The control also includes
the tail capability fix, so it does not isolate a single causal change.

## Simplification review of 0f79b7d

Portable mHC and shared-expert clamp now reuse the existing policy binding:
dependency availability is resolved when binding, unsupported workloads follow
the fallback policy, and execution errors propagate. The portable model clamp
delegates to that same implementation. Generic MoE clamp also stops swallowing
RuntimeError/OOM. Activation publishes its plan only after required policy-cache
invalidation succeeds; unknown patch baselines fail explicitly. The unused
single-patch API and duplicate activation path are removed. Signature checking
is documented as a limited parameter-name check.

Deployment vision budgets are cached, and each preprocessing entry resolves
request options once. Direct processor validation, request ceilings and video
timestamp handling remain covered. Ordinary decode retains graph padding and
mixed prefill handling while dropping unsupported speculative compatibility.
Persistent top-k is now a candidate inside the common top-k binding, so explicit
reference selection and CUDA vendor exclusion also govern that fast path.

Source-string/AST assertions and copied old-error demonstrations are removed;
clamp arithmetic and translated-table storage checks execute production code.
Production code shrinks by 164 lines and tests by 116 lines. The CPU suite passes
820 tests (4 skipped, 7 GPU cases deselected, 4 subtests). Changed-file lint and
diff checks pass. No GPU, installed-wheel or serving run was repeated for this
refactor; the earlier accelerator results remain tied to their recorded commits.

FULL-graph compatibility checks for reference/fallback bindings remain follow-up
work. The restricted deployment above still does not qualify the broader default
FlagGems configuration or portable whole-model execution.

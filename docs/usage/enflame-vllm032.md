# Enflame S60 with vLLM 0.32

This backend targets vLLM `0.32.0` with `VLLM_TARGET_DEVICE=empty` and the
Enflame S60 runtime. The deployment must use matching vendor builds of
PyTorch, TorchGCU, FlagTree, FlagGems, and FlagCX. Install the plugin with
`--no-build-isolation --no-deps` so the validated vendor runtime is preserved.

## Model runner

Enflame follows upstream vLLM 0.32 V2 model-runner selection. The plugin no
longer defaults GCU to V1 or rejects V2. Do not disable V2 to bypass adaptation
failures. On GCU, V2 uses upstream explicit-copy buffers and translates block
table pointer handles to device addresses before Triton indirect loads. The same
translation applies to Mamba state and aligned-block-table pointers, without
replacing their stable tensor storage. V2 graph capture uses the platform stream
context rather than upstream CUDA/XPU-only communicator checks.

Enflame keeps `fill_.Scalar` on the native backend (`fill_scalar_` in the
FlagGems blacklist). A controlled GCU regression reproduced corrupt successive
int64 scalar assignments with the FlagGems override; excluding only that
operator restored pointer metadata. This does not disable tensor fills or copies.

The GCU FLA adapters flatten launch coordinates for chunk output and fused
sigmoid-gating updates, avoiding the device's 255-block secondary-axis limit.
The kernel arithmetic is unchanged; actual-device boundary tests at 256 chunks
and 256 sequences matched partitioned upstream execution exactly.

V2 staged penalty statistics use a host histogram fallback because the installed
GCU Triton compiler cannot legalize the required atomic OR (or atomic CAS).
Only newly staged request rows are copied back to the device; penalty application
and model graph replay remain on GCU. Tests cover repeated tokens, signed bit 31,
empty prompts, multiple token blocks, and preservation of unselected rows.

V2 logprob collection selects top-k token IDs on the host, then copies the
selected values and indices back to GCU. This avoids excessive cold compilation
in the installed FlagGems top-k kernel and the native GCU index-dtype mismatch.
The adapter preserves upstream score and rank computation and does not replace
global `torch.topk`. Actual-device checks cover the full vocabulary and both
raw-logit and normalized-logprob output.

V2 validation on Qwen3-0.6B, BF16, TP=1, eager mode confirmed the actual worker
runner as `vllm.v1.worker.gpu.model_runner.GPUModelRunner` and generated five
requests of 32 tokens. Updated regression tests passed (194 tests). Multi-group
block table writes and gathering reordered requests also passed on hardware.
Against the existing same-weight NVIDIA upstream baseline, three requests were
exact; Paris first differed at token 11 and arithmetic at token 7 (one-based).
The strict precision gate therefore remains open. These results do not verify
V2 graph capture, TP>1, prompt embeddings, multimodal models or PCP on that
small-model run.

Additional V2 validation at commit `525b527` passed both Qwen3.6-27B and
Qwen3.6-35B-A3B in BF16, TP=2, eager serving: all five adaptation-gate scenarios
per model (text, image, and mixed requests, including eight-way concurrency),
10 scenarios and 52 requests in total. This includes the formerly failing
35B-A3B eager single-text assertion. Graph-mode acceptance is tracked separately;
eager success does not establish graph support or NVIDIA precision alignment.

At commit `f55376c`, both Qwen3.6-27B and Qwen3.6-35B-A3B passed the
complete V2 serving gate in BF16, TP=2, in both eager and graph modes:

| Model | Eager | Graph |
|---|---:|---:|
| Qwen3.6-27B | 26/26 requests | 26/26 requests |
| Qwen3.6-35B-A3B | 26/26 requests | 26/26 requests |

All 20 scenarios and 104 requests passed, including text, image, and mixed
eight-way concurrency. An independent audit replayed the original inputs and
output quality assertions, confirmed one tested commit and closed service ports;
all four test and server exit codes were zero. The 35B graph run used
`max_num_seqs=128` and `custom_ops=["none", "+SiluAndMul"]`.
The scoped regression suite passed 549 tests with one skip; actual-GCU checks
also covered penalty statistics, full-vocabulary logprobs in both output modes,
and the FLA launch-grid boundary. These are functional and operator-level
checks, not strict NVIDIA token-level precision alignment.

The historical validation below used ModelRunnerFL (V1) and is not V2 validation.

## Historical V1 verified scope

- Qwen3-0.6B BF16 eager and compiled execution on S60, including TP=2 and a
  real FlagCX collective.
- Qwen3.6 27B TP=2 eager and graph request scenarios on the validated S60
  setup.
- Qwen3.6 35B-A3B TP=2 graph scenarios with the explicit FL SiLU dispatch
  configuration.
- GCU vocabulary masking, MoE routing, KV-cache views, and focused dispatch
  regressions.

These results are backend-specific evidence, not a general claim for other
models, devices, or vendor runtime versions. The complete adaptation matrix
and automatic CI are not yet accepted.

## Required configuration

For the verified compiled BF16 configuration, set:

```bash
export TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
```

The option is not a plugin-wide production default. Validate output precision
again for every deployment. The GCU compiler adapter preserves integer pointer
ABIs and salts compiler and vLLM caches using the same adapter mode.
Set `FL_GCU_PRESERVE_DTYPE_CODEGEN=0` before process startup to restore
vendor code generation for diagnosis. This mode uses a different cache
identity and requires independent integer-pointer correctness validation.

## Known limitations

- NVIDIA reference comparison found token-level differences for some prompts;
  the observed numerical differences are accepted for the recorded Qwen3-0.6B
  scope, but this is not a strict token-match pass.
- The recorded V2 eager passes do not yet establish the complete four-group
  eager/graph adaptation gate.
- Other model architectures, hybrid speculative decoding, and unlisted graph
  shapes require separate validation.
- Enflame CI uses the pinned 0.28 image as a vendor dependency base. During
  setup, `install_vllm.sh` verifies Torch/TorchGCU, installs the pinned upstream
  vLLM 0.32 empty build and compiler/operator wheels, and verifies the resulting
  stack before running tests. The image tag continues to describe its original
  contents; the CI environment is upgraded during setup. This installation path
  still requires validation on the actual CI runner.

Successful imports, compilation, or HTTP readiness do not establish completed
inference or precision alignment. Preserve request-level results when running
the adaptation gate.

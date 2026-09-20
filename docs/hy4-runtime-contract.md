# HY4 runtime contract

The adapter targets vLLM 0.24 on NVIDIA. Registration is separate from model
construction: it must not probe quantization kernels for unrelated models.
The MXFP8 alias workaround is installed lazily during HY4 architecture
conversion; its override behavior changes only when `hf_config.model_type`
is `hy_v4`.

## Correctness boundaries

- A checkpoint with `learnable_sink` requires a sink-capable backend. Backend
  failure is fatal; there is no mode that loads and silently ignores sinks.
- Shared top-k layers require a preceding full sparse producer. Shared
  indexers currently require PP=1 because intermediate tensors carry streams,
  not top-k indices. All-full PP configurations retain the existing path;
  this does not constitute new distributed PP validation.
- Weight coverage checks both parameter names and fused components. Gate/up,
  WK/weights-projection, and each local expert's weight/scale shards must all
  be supplied. Complete fused gate/up and WK/weights-projection tensors are
  also accepted: their full shape is checked before loading, respecting the
  destination parameter's TP layout, then all components are recorded.
  Direct routed-expert destination tensors remain unsupported; use the split
  expert names or packed gate_up_proj/down_proj layout. FP8 values requiring
  dequantization must retain split weight/scale names. Remote PP/EP components
  are not required on the current rank.
- The portable FlagGems provider supports BF16 KV, including `auto` when model
  dtype is BF16. FP8 KV requires a complete native FlashMLA metadata/decode
  path. Weight quantization does not imply support for quantized KV.
- FlagGems quantization respects `USE_FLAGGEMS`, backend preference, and
  whitelist/blacklist policy. An empty build that needs a forbidden operator
  fails instead of ignoring the policy. Capability checks distinguish a
  registered schema from a device or composite implementation.

## Compatibility and optimization scope

The fallback installer in `patches/hy_v4_runtime.py` modifies process-global
vLLM and FlagGems attributes in a dedicated HY4 worker. Before weight buffers
are allocated, HY4 resolves a runtime plan containing the query quantizer,
indexer/top-k, cache-update and sparse-attention implementations, MLA prefill
backend, and supported KV dtypes. Only selected FlagGems implementations must
be callable and permitted by policy: native prefill and cache updates do not
require their unused FlagGems equivalents. Native decode top-k checks follow
`index_topk`: 512/1024/2048 need persistent top-k and, on Hopper, cooperative
top-k; other sizes need the generic decode kernel. The portable dense prefill path requires
`0 < v_head_dim <= qk_head_dim <= 256`, matching the validated FlagGems API;
compressed sparse-MLA dimensions are not dense prefill dimensions. Only the
model constructor prepares the runtime. Decoder layers, attention, indexers
and the installer require that resolved plan; they do not repair a missing
plan or repeat its policy/capability checks. The prefill selector installed
for the worker returns the already resolved backend, and query forward uses
the bound quantizer.

Explicit `mla_prefill_backend` selections preserve upstream errors. Only
missing automatic candidates may select the portable backend; arbitrary
configuration ValueError and AssertionError are propagated.

Installation is locked,
failed attempts roll back, and the completion marker is set last. It no
longer changes the `has_deep_gemm` capability fact. This is still a compatibility
adapter, not an instance-isolated provider: hot-swapping unrelated models in
the same initialized process remains unsupported.

The expert mapping helper is used when exported; execution errors propagate.
Its local no-EPLB mapping is only for images without that export. HC custom-op
registration errors also propagate once the registration helper is imported.
MoE combines routed/shared results in FP32, relying on the existing runner and
shared projection for reduction; sequence-parallel all-gather remains in the
caller.

The common attention metadata producer optimization is excluded from this
model change and is tracked separately in #442. `ModelRunnerFL` retains the main branch's BlockTable producer,
including eager, padding, and speculative-decoding behavior.

HC projection, HC pointwise fusion, and shape-aware MM are opt-in pending a
matched end-to-end quality/performance comparison. Explicit switches remain:
`VLLM_HY4_HC_N8_PROJECTION`, `VLLM_HY4_HC_POINTWISE_FUSION`, and
`VLLM_FL_FLAGOS_MM_SHAPE_AWARE`. This iteration adds no new device kernels.

## Validation entry points

```bash
VLLM_PLUGINS=fl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python -m pytest -q -o addopts=--tb=short \
  tests/unit_tests/model_loader/test_hy_v4_indexer.py \
  tests/unit_tests/patches/test_hy_v4_review_contracts.py \
  tests/unit_tests/patches/test_hy_v4_selected_paths.py \
  tests/unit_tests/patches/test_hy_v4_v024.py
```

The real checkpoint case `hy4/fp8_tp16` is discoverable for CUDA/H100 through
`tests/platforms/cuda.yaml` and the standard inference/serving entry points.
It requires the checkpoint mounted at `/data/models/Hy4-preview-FP8-Testing`
and a dedicated `gpu-16` runner. Discovery is not proof that such a runner
has been provisioned or that CI has executed the case. The standard case is
a real-weight semantic smoke test with expected Paris/green answers, not a
logits-parity or accuracy benchmark.

The query-quantizer FP8 reference test requires Hopper or newer and asserts
E4M3 output. The pinned FlagGems implementation defaults to FP32 output on
pre-Hopper devices, so the A100 CI runner skips this FP8-specific case while
retaining the BF16 prefill reference tests. This does not add A100 model support.

The separate `tests/e2e_tests/inference/test_hy_v4.py` is a manually invoked
single-GPU dummy-weight smoke and must not be reported as checkpoint validation.
Full-model evidence must identify the installed wheel hash, checkpoint,
dependencies, driver, topology, and launch settings; older GPQA results do not
certify a new wheel. Native/FlagOS logits parity, multi-node PP, mixed-model
process isolation, FP8 KV, and matched optimization A/B require separate
validation before broader support claims.

# MiniMax M3 MetaX code map

This guide describes the verified BF16 TP16 profile on vLLM 0.24. It is a
structure-only follow-up to the implementation at `8488e9ab`: dependency pins,
kernel arithmetic, launch parameters, shape guards and communication order are
retained. The [example](../examples/minimax_m3_metax/README.md) contains installation,
configuration and launch instructions, together with the retained evaluation and
serving measurements.

```text
models/minimax_m3.py                       shared native model exports
models/minimax_m3_registration.py          shared CUDA/opt-in MetaX registration
models/minimax_m3_metax/                   MetaX component deltas from public vLLM
ops/minimax_m3/                            vLLM interfaces, routing and expert orchestration
attention/minimax_m3/                      TP/shape/consumer selection and fallback
dispatch/backends/vendor/metax/impl/minimax_m3/  local kernels and explicit launches
dispatch/backends/flaggems/impl/minimax_m3/  FG/FGV call contracts
patches/minimax_m3_metax/                   scoped initialization and ABI compatibility
examples/minimax_m3_metax/                 configuration, dependencies and launch
```

Follow a call from the model interface to dispatch or attention orchestration,
then to its local kernel or library implementation. `vendor` describes hardware
specialization, **not** closed-source provenance. Unchanged vLLM Triton kernels
are imported directly and still compiled with FlagTree.

## Model and operator interfaces

Paths below are relative to `vllm_fl/`.

| File | Responsibility and source |
|---|---|
| `models/minimax_m3.py` | Reuses the mainline native class exports and fused projection metadata; no hook installation during class lookup. |
| `models/minimax_m3_registration.py` | Shared registration entry; preserves native CUDA support and explicitly installs the opt-in MetaX bindings/config verifier. |
| `models/minimax_m3_metax/__init__.py` | Package marker for the existing MetaX components; no model export facade. |
| `models/minimax_m3_metax/model.py` | Local subclasses for norm, QKV, sparse attention and MoE; reuse public vLLM construction/loading. Framework adaptation. |
| `models/minimax_m3_metax/vision.py` | Patch projection and chunked vision forward; native attention/RoPE class remains upstream. Framework adaptation. |
| `models/minimax_m3_metax/bindings.py` | Explicit, idempotent replacement of named symbols inside vLLM M3 modules. This is a runtime hook. |
| `models/minimax_m3_metax/upstream.json` | Small source-revision guard for the required vLLM model files. |
| `ops/minimax_m3/msa.py` | Cached dispatch handles for index and sparse interfaces. |
| `ops/minimax_m3/torch_ops.py` | Torch schemas, mutation/fake contracts and graph boundaries. Calls the framework bridge or FG/FGV adapters; defines no calculation kernel. |
| `ops/minimax_m3/registration.py` | Opt-in registration for this profile. Five MSA entries are MetaX VENDOR implementations; FG/FGV entries remain distinct. The standard backend is unchanged. |
| `ops/minimax_m3/layers.py` | vLLM clamped-activation interface using dispatch. |
| `ops/minimax_m3/aten.py` | Scoped ATen registrar: selects local Linear policy and the generic FG `masked_fill_`; retains the non-inplace blacklist. |
| `ops/minimax_m3/linear.py` | Model shape tables, guards and launch selection. Calculation is FG Linear/GEMV, including FG #7320 primitives. |
| `ops/minimax_m3/router.py` | Local `GateLinear` specialization, graph/fake boundary and exact shape guard; calls FGV FP32 router GEMM (#960). |
| `ops/minimax_m3/routing.py` | Bias-router interface using the accepted materialized sigmoid/top-k contract. |
| `ops/minimax_m3/expert_layer.py` | Local vLLM expert/factory subclasses; no replacement of the global factory/oracle. |
| `ops/minimax_m3/experts.py` | Reuses shared FL expert execution, supplying M3 alignment, activation, GEMM and reduction hooks. |
| `ops/minimax_m3/moe.py` | Per-call G1/G2 pairing, shape/layout guards, realignment and accepted small-batch policy. Calls FGV or the local routed GEMV. |
| `ops/minimax_m3/vision.py` | vLLM encoder-attention interface; calculation goes through the FG vision dispatch entry. |
| `ops/minimax_m3/vision_chunk.py` | Vision patch grouping and output concatenation; retained chunk boundaries. |

## Attention orchestration

Prefix: `vllm_fl/attention/minimax_m3/`.

| File | Responsibility and source |
|---|---|
| `bridge.py` | TP16, dtype/layout and consumer-identity guards; selects long/base/fallback paths and prepares exact decode indexing. Framework code, no JIT kernel. |
| `attention_policy.py` | Accepted long-context index/prefill/decode parameter selection from typed configuration. |
| `base_index.py` | Base index shape checks and original fallback. |
| `index_policy.py` | Base index parameter selection. |
| `base_sparse.py` | Base query packing and subtile-decode guards, retaining native fallback. |

Query/head packing is local to a rank; it does not add cross-rank exchange.
The base and long branches are alternatives selected by the original guards,
not extra attention passes.

## MetaX local computation

Prefix: `vllm_fl/dispatch/backends/vendor/metax/impl/minimax_m3/`.

| File | Responsibility and source |
|---|---|
| `attention_launch.py` | Explicit launch arguments and partial/LSE workspaces for the selected long path. Custom FL launch code. |
| `index_score.py` | Vector/dot index scoring and launch wrapper. Custom Triton companion computation for the accepted local index layout. |
| `long_prefill.py` | Shared packed-query prefill kernel; full or selected-page traversal. Custom M3 specialization retaining the accepted BF16 softmax-to-PV materialization. |
| `packed_prefill.py` | Base-path launch of that same prefill template; no duplicated kernel. |
| `long_decode.py` | Shared subtile/page-split decode kernel, adapted from vLLM sparse GQA; merge kernel imported from vLLM. |
| `subtile_decode.py` | Base decode partial/LSE allocation and launch/merge arguments. |
| `native_index.py` | Original index/top-k fallback with explicit MetaX launch settings. JIT bodies imported from pinned vLLM. |
| `native_sparse.py` | Original sparse-attention fallback with explicit launch settings. JIT bodies imported from pinned vLLM. |
| `moe_gemv.py` | Routed G1/G2 GEMV companion to the M3 expert layout and guards. Custom Triton calculation. |

`dispatch/jit.py` reads the JIT handle under decorators for per-call launch
metadata. It does not mutate a global tuner, compiler or library export.
The small fallback launch wrappers remain because the pinned vLLM interface
does not expose these settings per call.

## FG/FGV and closed-source boundaries

Prefix: `vllm_fl/dispatch/backends/flaggems/impl/minimax_m3/`.

| File | Responsibility and actual calculation source |
|---|---|
| `numerics.py` | FG Gemma without residual; FGV residual Gemma, division-form SwiGLU and staged-BF16 QKV/norm/RoPE/cache insertion (#961). Thin call/shape adapters, no copied kernel. |
| `moe.py` | FGV no-TLE alignment, expert GEMM, sequential sum and grouped top-k; FG materialized sigmoid before selection. |
| `vision.py` | FG varlen attention and matrix-based patch projection. vLLM classes stay in models/ops. |

The same public FG/FGV integration commits remain in
[`dependency-lock.json`](../examples/minimax_m3_metax/dependency-lock.json).
The existing shared partial-RoPE dispatch from #560 is reused. Independent
vendor FlashAttention, necessary MCTlass/binary calls and MCCL communication
continue through the existing MetaX backend. Those binaries are closed-source
dependencies; the local Triton files in the vendor directory are source-visible.
Neither the `vllm_metax` framework nor FlagCX is introduced.

## Configuration, hooks and validation

| File | Responsibility |
|---|---|
| `minimax_m3_config.py` | Typed immutable policy, correctness-sensitive flags/blacklists and explicit MSA VENDOR policy validation. |
| `patches/minimax_m3_metax/startup.py` | Ordered library configuration, scoped accelerator preparation and legacy-framework import guard; the public FL platform remains in use. |
| `patches/minimax_m3_metax/bootstrap.py` | Effective provider, numerical policy and graph-capture audit. |
| `patches/minimax_m3_metax/compat.py`, `compatibility.json` | Version/signature/source-guarded Torch/vLLM/vendor ABI hooks. Unchanged by this cleanup. |
| `patches/minimax_m3_metax/platform_compat.py` | Retained MetaX accelerator compatibility. |
| `patches/minimax_m3_metax/launch.py`, `reasoning.py` | Normal vLLM CLI invocation and scoped M3 reasoning parser. |
| `platform.py`, `worker/minimax_m3.py` | The common platform applies the M3 config verifier and selects the existing worker; graph/collective boundaries and after-capture audit are retained. |
| `examples/minimax_m3_metax/config.json`, `serve.py`, `tuning-ar512k.cfg` | Accepted settings, isolated cache/launch and existing MCCL message-size policy. |
| `tests/unit_tests/test_minimax_m3_packaging.py` | Configuration, blacklists, dependency identity and launcher regressions. |
| `tests/unit_tests/test_minimax_m3_structure.py` | Scope, exact provider selection and import-boundary checks. |
| `tests/unit_tests/test_minimax_m3_registration.py` | Shared registration, CUDA/MetaX scope, profile validation and failure rollback. |

Package markers in the new directories have no initialization side effects.
No installed vLLM source is rewritten. Existing model bindings and compatibility
hooks remain real runtime hooks, not a claim of patch-free execution.

Historical accuracy and throughput tables remain in the example. This cleanup
is checked with CPU/import/source-equivalence checks and a bounded TP16 graph
smoke; it does not rerun those full evaluations or formal benchmarks.

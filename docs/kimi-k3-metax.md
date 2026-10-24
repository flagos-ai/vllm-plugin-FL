# Kimi K3 MetaX: code reading guide

This guide describes the structural refactor of [PR #607](https://github.com/flagos-ai/vllm-plugin-FL/pull/607).
The target remains the accepted single-instance TP8 × PP4 path with MBT4096.
The later PD/MBT8192 experiment is separate and is not included.

```text
examples/kimi_k3_metax/          configuration, source pins and launch example
vllm_fl/models/                 native K3/vision component subclasses
vllm_fl/ops/kimi_k3/             vLLM interfaces and TP/layout contracts
vllm_fl/attention/kimi_k3/       metadata, shapes and attention orchestration
vllm_fl/quantization/mxfp4/     packed-weight / expert interface
vllm_fl/worker/                 runner, workspace, PP ownership and Graph boundary
vllm_fl/distributed/            host-packed sampled-token feedback
vllm_fl/dispatch/backends/
  flaggems/                    thin FG/FGV adapters and ATen registrar
  vendor/metax/impl/kimi_k3/    local attention, expert, KDA and linear compute
vllm_fl/patches/kimi_k3/         explicit startup, schema and runtime compatibility
```

The call direction is **model/interface → framework orchestration → selected implementation → local kernel or pinned library**.
Model-coupled TP8 packing, state and buffer contracts remain in FL.
`VENDOR` describes backend selection, not a closed-source classification.
This refactor introduces no operator PR or dependency upgrade.

The existing `VLLM_FL_CONFIG` selects `examples/kimi_k3_metax/dispatch.yaml`.
Generic MetaX capabilities and default policy remain available.
K3 differences use namespaced VENDOR capabilities; library operations keep FG/FGV selection and blacklist filtering.
The ATen Linear provider is resolved once at worker setup; other boundaries use the existing CachedOp mechanism.
Kernel arithmetic, explicit launch tables, shape guards, quantization/rounding and communication order are preserved.

## Source labels

- **FG / FGV**: the pinned FlagGems / FlagGems-vllm implementations, including explicitly identified private kernel APIs.
- **vLLM / FLA**: inherited models/loaders/metadata/runners and reused KDA kernels.
- **Custom FL**: framework glue, layout-specific computation or compatibility adaptation; not automatically a new algorithm.
- **Closed runtime**: MetaX Torch/MACA and MCCL binaries, wrapped by reviewable Python/C++ source.
  Host token feedback uses PyTorch/Gloo. GPU TP/PP communication retains MCCL.
  K3 does not import `vllm_metax` or `mcoplib`; this single-instance profile does not use FlagCX.
  `marlin_mxfp4.py` is a local packed-weight implementation, not a call into a closed Marlin library.

## File-by-file map

Paths are repository-relative. Empty package `__init__.py` files establish namespaces without registering another backend or adding forwarding calls.


### Models and vLLM interfaces

| File | Responsibility | Source / nature |
|---|---|---|
| `vllm_fl/kimi_k3_config.py` | Immutable K3 model/topology policy. | Custom FL configuration, interface or lifecycle |
| `vllm_fl/models/kimi_k3.py` | Native model component factories and the changed methods only. | vLLM inheritance + custom FL differences |
| `vllm_fl/models/kimi_k3_context.py` | Resolve the construction-time model policy. | Custom FL configuration, interface or lifecycle |
| `vllm_fl/models/kimi_k3_vision.py` | Vision component classes, bounded MLP rows and real-view layout handling. | vLLM inheritance + custom FL differences |
| `vllm_fl/ops/kimi_k3/activation.py` | Qualified SiTU interface and its unchanged output/buffer policy. | vLLM inheritance + custom FL differences |
| `vllm_fl/ops/kimi_k3/graph_context.py` | Read the existing FULL-Graph hint for shape selection. | Custom FL configuration, interface or lifecycle |
| `vllm_fl/ops/kimi_k3/latent_moe.py` | Latent projection interfaces, TP gather/reduce order and native tail selection. | vLLM inheritance + custom FL TP orchestration; MCCL runtime |
| `vllm_fl/ops/kimi_k3/projections.py` | Native projection classes with TP-local eligibility and launch parameters. | vLLM inheritance + custom FL differences |
| `vllm_fl/ops/kimi_k3/workspace_contract.py` | Shared workspace shape constants; no allocation or computation. | Custom cache/layout/numeric compatibility or interface contract |
| `vllm_fl/quantization/mxfp4/kimi_k3.py` | MXFP4 weight/quantization interface and packed TP8 expert selection. | vLLM inheritance + custom FL differences |

### Framework attention

| File | Responsibility | Source / nature |
|---|---|---|
| `vllm_fl/attention/kimi_k3/backend.py` | Native MLA metadata, output buffers and prefill/decode orchestration. | vLLM inheritance + custom FL differences |
| `vllm_fl/attention/kimi_k3/decode.py` | FGV callback and physical-stride-safe dense fallback for TP8. | Custom FL configuration, interface or lifecycle |
| `vllm_fl/attention/kimi_k3/kda.py` | Native KDA interface with the selected local implementation. | vLLM inheritance + custom FL differences |
| `vllm_fl/attention/kimi_k3/mla.py` | Model MLA component and query/gate projection binding. | vLLM inheritance + custom FL differences |
| `vllm_fl/attention/kimi_k3/vision.py` | Bounded vision attention using registered ATen operations. | vLLM inheritance + custom FL differences |

### Worker and communication

| File | Responsibility | Source / nature |
|---|---|---|
| `vllm_fl/distributed/kimi_k3_pp.py` | Host-packed sampled-token feedback and native queue/invalidation order. | vLLM PPHandler + custom FL packets; PyTorch/Gloo feedback |
| `vllm_fl/worker/kimi_k3.py` | Runner-owned workspace and release of consumed PP input references. | vLLM inheritance + custom FL differences |
| `vllm_fl/worker/kimi_k3_graph.py` | Preserve native Graph replay and its accepted MCCL stream boundary. | vLLM inheritance + custom FL differences |
| `vllm_fl/worker/kimi_k3_workspace.py` | Reserve the accepted workspace inside KV memory budgeting. | Custom FL configuration, interface or lifecycle |
| `vllm_fl/worker/pipeline_input.py` | Preserve receive ownership and release-after-copy. | FL PP lifetime adapter |

### FG / FGV adapters

| File | Responsibility | Source / nature |
|---|---|---|
| `vllm_fl/dispatch/backends/flaggems/impl/kimi_k3.py` | Thin RMS wrapper preserving the accepted residual guard. | FG normalization |
| `vllm_fl/dispatch/backends/flaggems/impl/kimi_k3_cache.py` | NoPE cache insertion and output-layout adapter. | FGV cache primitive + FL ABI/layout |
| `vllm_fl/dispatch/backends/flaggems/impl/kimi_k3_mm.py` | One canonical FG MetaX module and unchanged native JIT handles. | FG primitives / registration + FL TP-specific policy |
| `vllm_fl/dispatch/backends/flaggems/impl/kimi_k3_routing.py` | Materialized FP32-sigmoid routing under the accepted guards. | FG/FGV thin library adapter |
| `vllm_fl/dispatch/backends/flaggems/kimi_k3_registrar.py` | Resolve the chosen Linear provider once before ATen registration. | FG primitives / registration + FL TP-specific policy |

### MetaX local implementations

| File | Responsibility | Source / nature |
|---|---|---|
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/attention/cache_gather.py` | Cache gather with the accepted ABI and physical strides. | Custom cache/layout/numeric compatibility or interface contract |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/attention/mla4k_long.py` | Shared staged QK and split merge used by TP8; H3 host/PV paths removed. | Custom TP/head-layout MLA specialization |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/attention/mla_bmm.py` | TP-local query projection and fallback. | Custom TP/head-layout MLA specialization |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/attention/mla_dense_fallback.py` | Physical page strides, int64 addresses and separate dot/add rounding. | FGV-derived kernel with substantive layout/numeric differences |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/attention/mla_tp8.py` | TP8 local-head MLA selection and unchanged kernel launches. | Custom TP/head-layout MLA specialization |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/attention/prefill_kernel.py` | Qualified prefill attention fallback. | Custom TP/head-layout MLA specialization |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/attention/prefill_mma.py` | H12 prefill QK/PV kernel and original tile/split choices. | Custom MLA prefill specialization |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/activations.py` | Qualified SiTU interface and its unchanged output/buffer policy. | FGV primitive + FL shape/buffer policy |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/decode_bits.py` | Packed MXFP4 bit decoding. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/decode_experts.py` | Small-batch packed expert G1/G2 paths and output reduction. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/expert_gemm.py` | Expert GEMM with the accepted decode, accumulation and rounding. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/grouped_moe.py` | Qualified expert fallback, alignment and weighted reduction. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/marlin_mxfp4.py` | Packed-word layout and qualified tail-GEMV support. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/mxfp4_kernels.py` | Accepted MXFP4 decode primitives and fallback compute. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/small_alignment.py` | Compact-buffer policy around the pinned FGV alignment kernel. | FGV primitive + FL shape/buffer policy |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/tp8_expert_g1.py` | Fused packed G1/SiTU small-batch specialization. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/experts/tp8_experts.py` | Packed-weight preparation, path selection and bounded G2 workspace. | Custom TP8 packed-expert compute; FGV primitives where called |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/kda/chunk.py` | Native recompute/output kernels and retained FP64 gate cumsum. | Reused vLLM K3/FLA kernels + retained numeric/launch compatibility |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/kda/chunk_delta_h.py` | Native FLA state-update kernel with local launch settings. | Reused vLLM K3/FLA kernels + retained numeric/launch compatibility |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/kda/chunk_intra.py` | Native inter/solve/sub-chunk and token-parallel selection. | Reused vLLM K3/FLA kernels + retained numeric/launch compatibility |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/kda/interface.py` | KDA normalization/materialization and state contract. | Custom cache/layout/numeric compatibility or interface contract |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/kda/launch.py` | Local kernel handles without replacing upstream exports or global tuning. | Reused vLLM K3/FLA kernels + retained numeric/launch compatibility |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/kda/normalization.py` | Preserve the evaluated KDA normalization precision. | Custom cache/layout/numeric compatibility or interface contract |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/kda/preprocess.py` | KDA input preparation with the existing layout guards. | Custom cache/layout/numeric compatibility or interface contract |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/linear/latent.py` | TP-local D projection, grouped row execution and original launch settings. | FG primitives / registration + FL TP-specific policy |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/linear/latent_up.py` | Unchanged U-shard accumulation extracted from the framework class. | FG addmm + FL local launch |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/linear/linear.py` | Qualified Linear shapes, local launches and exact generic fallback behavior. | FG primitives / registration + FL TP-specific policy |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/linear/projections.py` | TP-local projection launch table and KV expansion kernel. | FG primitives / registration + FL TP-specific policy |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/linear/small_projection.py` | Unchanged split-K launch body extracted from projection.forward. | FG partial/reduce + FL local launch |
| `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/vision.py` | Unchanged complex64 rotation and bounded temporary rows. | Custom compatibility using Torch/registered ATen |
| `vllm_fl/dispatch/backends/vendor/metax/kimi_k3_ops.py` | Add model-scoped capabilities while retaining generic MacaBackend entries. | FL VENDOR registration; not a closed library |

### Startup and compatibility

| File | Responsibility | Source / nature |
|---|---|---|
| `vllm_fl/patches/kimi_k3/cache.py` | Initialize private per-process compilation caches before library imports. | Custom FL configuration, interface or lifecycle |
| `vllm_fl/patches/kimi_k3/cuda_runtime.py` | Reuse the existing SDK wrapper; retain the verified library lookup. | MetaX wrapper inheritance + closed MACA runtime |
| `vllm_fl/patches/kimi_k3/mccl_runtime.py` | Reuse collectives; retain abort, symbol caches and unsupported-window checks. | MetaX wrapper inheritance + closed MCCL runtime |
| `vllm_fl/patches/kimi_k3/op_registry.py` | Strict schema-only declarations and capability checks for the pinned empty build. | Compatibility; no fake model-kernel implementation |
| `vllm_fl/patches/kimi_k3/op_schemas.py` | Pinned declarations generated from the accepted vLLM source. | vLLM schema metadata; no compute |
| `vllm_fl/patches/kimi_k3/profile.py` | Select the explicit profile through the existing FL YAML setting. | Startup admission; no optimization environment toggle |
| `vllm_fl/patches/kimi_k3/registration.py` | One model, quantization, attention and torch-op startup registration. | Custom FL configuration, interface or lifecycle |
| `vllm_fl/patches/kimi_k3/runtime.py` | Install the accepted MACA/MCCL bindings once. | FL hook; closed hardware runtimes underneath |
| `vllm_fl/patches/kimi_k3/runtime_library.py` | Locate the same MCCL library. | Thin closed-runtime binding |

## Existing integration files

| File | Change boundary |
|---|---|
| `vllm_fl/__init__.py` | K3-only cache/schema startup; other profiles keep the original route. |
| `vllm_fl/model_registration.py` | One version-checked K3 general-plugin entry. |
| `vllm_fl/platform.py` | Select K3 runtime hooks and the accepted MCCL constraint only for its profile. |
| `vllm_fl/worker/worker.py` | Select the same K3 registrar and runner at their canonical locations. |
| `vllm_fl/dispatch/builtin_ops.py` | Ordinary backend initialization; K3 registration errors remain explicit. |
| `vllm_fl/dispatch/backends/flaggems/register_ops.py` | Scoped library overrides with blacklist filtering. |
| `vllm_fl/dispatch/backends/vendor/metax/register_ops.py` | Retain generic capabilities and add K3 entries. |
| `vllm_fl/dispatch/backends/vendor/metax/{__init__.py,patches/}` | Preserve legacy startup outside K3; share SDK class implementations without applying unrelated patches inside K3. |
| `vllm_fl/dispatch/config/metax.yaml` | Generic policy restored; K3 configuration is in its explicit example profile. |
| `vllm_fl/ops/_C_ops_{registry,schemas}.py` | Generic defaults restored; strict K3 declarations are under its startup hooks. |
| `setup.py` / `csrc/metax/uva.cpp` | New package location for the same source-built host extension; C++ implementation unchanged. |

## Preserved behavior

Short/small-batch paths, non-normal expert scales, dense/interleaved MLA pages and KDA numerical compatibility remain behind the same guards.
The existing local FP64 KDA normalization/gate accumulation is retained as compatibility behavior, not a new speedup.
Five native KDA kernels remain reused through local precision/launch handles.
There is no global compiler, autotuner or library-export replacement.

SDK differences are subclasses of the existing wrappers, not complete duplicated SDK modules.
The fixed dependency lock is unchanged.
Use [the installation and launch example](../examples/kimi_k3_metax/README.md); examples contain no source patches, snapshots or patch installer.

## Validation scope

“Validated” refers to this project's measured configurations, not official K3 certification of a vLLM release.
The package label `0.28.0+empty` is a build identifier; exact public source revisions are in the dependency lock.
[VALIDATION.md](../examples/kimi_k3_metax/VALIDATION.md) separates this refactor's checks from retained historical quality and performance results.
Full GPQA/ERQA and formal throughput benchmarks are not rerun for this structural change.

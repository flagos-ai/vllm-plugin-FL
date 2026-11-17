# Reviewing the DeepSeek V4 MetaX integration

This is the validated TP8 W8A8 profile for vLLM 0.20.2 (`release/0.2`).
The refactor changes ownership and dispatch wiring, not kernel arithmetic,
library revisions, launch parameters, cache dtype or serving workloads.

## One entry per capability

```
model / vLLM interface
  -> CachedOp + the existing dispatch policy
     -> default.flagos.dsv4 : pinned FG / FGV adapters and graph schemas
     -> vendor.metax.dsv4   : DS attention ABI -> MetaX local kernels
```

The generic `FlagGemsBackend` is not replaced. Its unchanged operations remain
available to every model. Seven DS library capabilities are registered only
when the explicit DS profile is configured and respect the FlagGems blacklist.
Two device attention capabilities are registered by the MetaX backend only.
The example's ordinary `op_backends` table selects their exact implementation
IDs; import/registration errors cannot silently select an older implementation.

| Responsibility | Review location |
|---|---|
| FG/FGV calls, mutable/fake schemas, MHC shape fallback | `dispatch/backends/flaggems/impl/deepseek_v4.py` |
| Provider registrations | the existing FlagGems and MetaX `register_ops.py` files |
| vLLM scheduler metadata / caller output ABI | `attention/deepseek_v4/bridge.py` |
| TP head exchange, M/2 logits gather and M/4 index gather | `attention/deepseek_v4/{headpack,row_sharding,query_sharding}.py` |
| Local TP8 sparse attention math and launch policy | `dispatch/backends/vendor/metax/impl/dsv4/attention/` |
| Local logits, paged reads, row scoring and exact tree top-k | `dispatch/backends/vendor/metax/impl/dsv4/indexer/` |
| MCTlass host execution and binary bindings | `dispatch/backends/vendor/metax/impl/dsv4/` |
| vLLM attention/indexer, embedding and MoE interface adapters | existing `ops/` modules; no copied kernel implementations |
| vLLM quantization scheme/expert interface | `quantization/w8a8/deepseek_v4.py` |
| Version-specific runtime compatibility | `patches/deepseek_v4_metax/` |

`vendor` means device-specialized, not necessarily closed-source: the attention
kernels are FL-owned Triton code for the TP8 padded-head contract. Generic pure
operators stay in FG/FGV. MCTlass GEMMs remain vendor binaries. Do not infer
operator-library coverage merely from the dispatch implementation ID.

## What moved, without changing its math

- `ops/deepseek_v4_metax/{decode,prefill,staged,launch,tuning}.py` moved unchanged
  to the MetaX attention implementation directory.
- The DS backend subclass and nested adapter directory were removed. Library
  adapters and their existing opaque Torch graph boundaries are in one file.
- Scheduler metadata conversion and output-buffer adaptation moved out of the
  FlagGems adapter into the framework attention interface.
- Local indexer implementations moved from `ops/` to the MetaX `indexer/`
  implementation package. The M/2 file was split at its existing kernel/host
  boundary: the scoring function belongs to dispatch; TP groups, eligibility,
  allocations, launches and collectives belong to the attention layer.
- Headpack and M/4 orchestration live alongside that M/2 path. Startup-only Hash
  validation and serial compatibility live with the other runtime hooks.
  Package initializers are inert, and Hash validation keeps its lazy model
  dependency. The old loose-file top-k import fallback is removed.
- Internal kernels are not registered as separate dispatch capabilities: the
  existing attention/indexer entrypoints keep their graph and buffer contracts.
- Headpack, M/4, INT8 rounding, router scaling, graph schemas/fakes and the
  single-token/short-query fallbacks retain their accepted contracts.

The private FGV MetaX MHC `run` entry is a pinned dependency of the #755 backport,
not claimed to be a stable public API. Replacing it with a different public
implementation would require a separate numerical and performance comparison.

## Compatibility is not a performance kernel

The small example bootstrap invokes one FL initialization entry. It does not
patch source files. Initialization has three explicit phases: prepare the
compatibility policy, bind standard dispatch capabilities, then record startup
evidence. A later migration to another vLLM version is outside this refactor.

The following runtime hooks intentionally remain, rather than being hidden:

| Hook | Why retained / boundary |
|---|---|
| Serial multi-stream adapter | Validated execution order; version and signature guarded |
| RMS Inductor launch policy | Preserves validated numerical behavior by body fingerprint and rank; still hooks the autotuner process-wide |
| CachedOp prebinding | Keeps the established non-frozen DS tracing path and import order |
| Vendor-import guard and Hash/RMS audit | Fail-fast and diagnostic checks, not GPU optimizations |

These hooks are not a claim of zero runtime monkey-patching. Removing them or
changing their timing is an independent compatibility change, not directory
cleanup. No experimental histories, source-patch installer or vendor binary is
included under `examples/`.

## Dependencies and checks

The immutable FG/FGV requirements are unchanged; see
`examples/deepseek_v4_metax/dependency-lock.json` and its README. No upstream
version upgrade is hidden in this refactor. CPU tests cover default-backend
preservation, profile-scoped registrations, blacklist handling, exact provider
selection and the existing graph/packaging contracts. Model smoke results are
reported separately from historical quality and throughput measurements.

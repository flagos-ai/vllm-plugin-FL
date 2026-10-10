# Iluvatar attention cache binding and decode graphs

Iluvatar uses the CUDA device API with the FlagOS GPU runner while reporting
`is_cuda_alike() == False`. In vLLM 0.24, the upstream cache binder rejects two
attention caches with the same decoder layer index on that platform. Models
with separate main and index attention caches can therefore fail during cache
initialization, before eager execution or graph capture.

The FlagOS runner now binds every Iluvatar cache in layer-index order, retaining
the dictionary order within a layer and the original tensor references. Shared
cache aliases, strides and dtypes remain intact. Other platforms continue to
use the upstream binder. This does not change `is_cuda_alike()`, graph defaults,
kernel dispatch, CUDA allocator capabilities or collective selection.

## Historical decode configuration

The M3 W8A8 adaptation used BI-V150 devices with COREX 4.5, vendor Torch
`2.10.0+corex.4.5.0`, vLLM 0.24, FlagGems
`5.3.4.post1.dev11+gbc6d9426c`, FlagGems-vllm at
`073fccd3f5a2c8870e55c6d1a16d1a906581a4d2`, and the public plugin baseline
`7c05ba28c5c3340b0f75cde7a45f538ead2a3177`. The compiler distribution was
FlagTree `0.7.0rc2+iluvatar3.6` with Triton 3.6 APIs; its original build SHA was
not recorded. The vLLM source baseline was
`ee0da84ab9e04ac7610e28580af62c365e898389`, with the PP patch
`d7c1821b5a31c886cf130e50f353e49af5b79659` and separate model/operator
adaptations. Those adaptations are prerequisites, outside this change.

Once the model and operators are independently supported and capture-safe,
the historical configuration fragment was:

```sh
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
# Add these options to the independently validated serve command:
# --disable-custom-all-reduce --no-async-scheduling
# --max-num-seqs 8 --max-num-batched-tokens 128
# --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8],"cudagraph_num_of_warmups":1}'
```

This captures decode at batch size 8. Smaller decode batches are padded to that
capture size; prefill and mixed batches run eagerly. The multiple-size
`[1,2,4,8]` experiment failed and is not a validated configuration. The above
settings are opt-in; the plugin does not select them automatically.

## Validation boundaries

Historical checks covered basic/norm/QKV/INT8-MoE graph capture and replay,
2-rank and 16-rank captured all-reduce followed by eager logits all-gather, and
short model response comparisons. Those runs used an earlier adaptation that
changed the platform capability flag, not this narrowly scoped cache binder.
They are context, not hardware validation or performance results for this
change. Resetting graphs and synchronizing before destroying communicators was
part of the collective test cleanup.

The CPU regression tests cover the upstream multi-cache rejection, complete
binding, tensor identity, shared aliases, ordering, multiple attention-module
indices and delegation to other platforms. Fresh BI-V150 execution, model
correctness, graph replay, long-context behavior and performance must be
validated on this branch before expanding any support claims. No experimental
model implementation or numerical kernel is included here.

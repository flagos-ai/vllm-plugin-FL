# Qwen3.8-Flash-Next / Qwen4Exp

This branch adds the plugin-owned model to the **vLLM 0.24.0** integration line
and merges the current repository `main` integration. No upstream vLLM source
changes are required.

## Architecture boundaries

| Responsibility | Location |
| --- | --- |
| Checkpoint config aliases, feature validation, model registry | `patches/qwen3_8_flash_next.py` |
| Model composition, config, weight loader, PLE, QSA, GDN and cache metadata | `models/qwen3_8_flash_next.py` |
| Portable fused QSA, PLE state I/O, packed GDN and slot metadata kernels | semantic modules exported by `flag_gems.fused` |
| FlagGems capability check, registration and vLLM argument adaptation | `dispatch/backends/flaggems/` |
| N-gram history input and optional KV-cache bind hook | `worker/model_runner.py` |
| Common-slot metadata graph lifecycle | `worker/common_slot_mapping.py` |

The historical `gpu` package name denotes the accelerator execution path, not
NVIDIA ownership. Model code does not branch on NVIDIA/PPU identity. Linear,
normalization, rotary, MoE and convolution layers reuse the existing framework.
QSA is kept as a model-owned attention/cache backend because its sparse paging
and compressed side-state semantics differ from dense attention.

New fused operators use the existing `OpImpl` → registry → policy → `resolve_op`
path. FlagGems owns the tensor-level implementations and exports their public ABI
from `flag_gems.fused`; the plugin owns only registration and the adapter that
unpacks vLLM cache-group objects for common-slot metadata. They are registered
lazily as `DEFAULT` / `default.flagos` only when all ten required FlagGems symbols
are callable. Vendor backends can register the same operator names with `VENDOR`;
existing preference, per-op ordering and allow/deny rules remain authoritative.
Registration imports neither a model nor a device kernel for unrelated workloads.

Model bindings resolve before graph capture. In-place cache/state operations
must not retry another backend after a partial update. Consequently these
bindings deliberately do not use runtime fallback. Change dispatch policy only
before constructing the worker; recreate workers/graphs after a policy change.

## Operator ownership

The initial adaptation used plugin-local Triton kernels because FlagGems revision
`dcd0cfeae8187dee691656c0616c1ed95899c2ae` did not expose the required ABIs.
Those kernels now live on `ZenithJoyH/FlagGems:flaggems-for-day0`, based on
`187fc1f7bc59616d0c9eadd8d463e8d6068b22d9`. This keeps accelerator operators in
FlagGems and model/framework semantics in the plugin. The FlagGems APIs accept
tensors and scalar layout metadata and do not import vLLM or `vllm_fl`.

- QSA: BF16 paged MQA scoring, top-k expansion, sparse GQA, row stores and group
  compression; physical page strides are respected, including padded pages.
- PLE: FP16/BF16/FP32 stride-aware row gather/scatter, null/invalid row handling,
  duplicate-write masks, caller-owned output support. Accelerator eager and
  graph use the same Triton path. CPU paths are numerical references.
- GDN: packed non-spec decode with FP32 sigmoid(beta), preserving the upstream
  cache layout and convolution. A model-specific subclass overrides only decode;
  no global upstream function, class or Triton kernel is replaced. State indices
  must be unique among valid rows; inner `[HV, V, K]` storage must be dense.
- Common-slot metadata: refresh page mappings, computed-token counts and padded
  request rows after scheduling, including graph replay when a request replaces
  an earlier row. This correctness fix is retained from the August 25 reference,
  while unrelated async/eventfd optimizations are excluded. Computation goes
  through dispatch; graph lifecycle uses the plugin's `Graph` and platform APIs.
  Runner activation is limited to PLE models. The dispatch producer reuses the
  shared `CommonAttentionMetadataGraphRunner` lifecycle, including warmup,
  first-capture replay, graph-unavailable eager handling and PIECEWISE request
  extents. T-Head synchronizes the producer stream after replay. Other models
  retain the upstream metadata enable policy and generic producer. PLE/QSA
  dummy forwards retain their slot bindings; generic dummy forwards clear them.

## Supported boundary

Target: BF16 Qwen3.8 checkpoint, FP32 GDN state, PPU-ZW810E with the pinned vLLM,
FlagGems and Triton/FlagTree runtime. QSA currently requires BF16 cache/query.
The 24 attention heads require compatible TP partitioning (TP8, not TP16).
PLE requires PP=1 and raw token IDs. DBO/microbatching and speculative/MTP decode
are rejected. This implementation uses `ModelRunnerFL`; the experimental new
vLLM runner is not supported by this branch.

On T-Head, retain the platform blacklist and add the four deployment exclusions
with `VLLM_FL_FLAGOS_BLACKLIST_APPEND=slice,conv1d,_conv_depthwise2d,index`.
This avoids replacing the broader platform safety policy. The exclusions cover
the verified bool-slice copy, depthwise-convolution memory, and multi-rank index
autotune database-lock signatures. They can be removed individually only after
the matching FlagGems path passes eager, changed-input graph, and TP startup
regression on the deployed revision.

The checkpoint's conditional-generation architecture alias is retained for
text-only serving (`--language-model-only`). Vision/audio, MTP, quantized weights,
EP, and non-PPU devices are **not accepted configurations**. Portable source and
policy tests do not establish device-level correctness on every vendor.

Removed from the adaptation snapshot: unrelated W8A8/Qwen3.5/platform changes,
model-local NVIDIA fast paths, MTP implementation/hidden-state buffers and the
unused alternative runner bridge. No generic platform identity or dispatch
policy redesign is bundled with this model addition.

## Verification

Run `python -m pytest -q tests/qwen3_8_flash_next tests/unit_tests/dispatch`.
Tests cover loading/mapping, cache layout, CPU n-gram preparation, independent
operator oracles, strides/dtypes/nulls/duplicates, eager and graph replay with
changed inputs, lazy registration, vendor policy and upstream isolation.

The 2026-09-07 PPU result (**410 passed**) applies to the earlier plugin-owned
kernel layout. It remains historical evidence for model behavior, not validation
of the ownership move. The integrated FlagGems revision requires the focused
FlagGems operator suite and plugin dispatch/model suite to pass again before the
branch is accepted. PPU eager and changed-input graph probes must also be rerun.

Environment: vLLM 0.24.0 (`ee0da84ab9e04ac7610e28580af62c365e898389`),
FlagGems (day0 branch above), PyTorch 2.10.0, Triton 3.5.0, Python 3.12.3.
Upstream vLLM remains unchanged.

Full-weight TP8 graph smoke inference also passed on contiguous PPU devices
8–15 with `max-model-len=65536`, `max-num-seqs=32`, and
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`. The deterministic chat request
returned HTTP 200 and content `2` in 25.58 seconds; the service remained healthy.
Formal FlagEval accuracy and performance acceptance must still be rerun for the
integrated revision. Earlier results from the uncurated adaptation are not
evidence for this branch.

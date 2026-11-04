# Qwen3.8 Flash Next on MetaX

This entry integrates the checkpoint architecture with plugin-FL on MetaX C550.
The current supported serving composition uses BF16, single-host TP8/PP1 and
vLLM 0.24.0. The example is text-only. It does not claim multimodal acceptance
or a new model-quality evaluation.

## Runtime and operator dependencies

Use the matching MetaX Torch/MACA/FlagTree environment and the exact
[operator composition](../../examples/qwen3_8_flash_next/metax_tuning/operator_dependencies.json).
The vLLM base is ee0da84ab9e04ac7610e28580af62c365e898389. The current development
composition also requires the default-off modular-expert output-buffer
capability documented in that manifest; the unmodified base does not provide
it. Make that native source/package take precedence over the base installation.

Several FG/FGV changes remain unpublished. Their source fingerprints identify
the reviewed development assembly; they are not public installation revisions.
Before public delivery, publish the required operator APIs and tests and replace
the private source requirements with team-accessible fixed revisions. Do not
substitute a floating branch, silently edit installed sources, or disable FlagOS.

The MetaX UVA mode additionally needs the plugin native host-memory extension,
built with VLLM_VENDOR=metax and the matching Torch/MACA ABI. The generic
host-row gather, BF16 fusion, sampling and QSA numerical implementations belong
to the operator libraries. Model metadata, resource lifetime and routing policy
remain in the plugin. No patch or diff delivery files are installed.

## Serving entry point

Install the pinned plugin/operator composition and matching native extension.
The POSIX [serve_metax.sh](../../examples/qwen3_8_flash_next/serve_metax.sh) calls
native vllm serve directly. It has no Python launcher, source rewrite,
dependency download or service-recovery step.

```sh
MODEL_PATH=/path/to/checkpoint \
QWEN38_CACHE_ROOT=/path/to/service-cache \
sh examples/qwen3_8_flash_next/serve_metax.sh --port 18480
```

The default [metax.yaml](../../examples/qwen3_8_flash_next/metax.yaml) selects
MTP off, MBT8192, asynchronous scheduling and UVA auto. Additional arguments
are passed directly to native vLLM; normal CLI-over-config precedence applies.
The script selects the model dispatch configuration, enables FG/FGV, and
isolates the kernel/config caches by service and MTP mode. Other dispatch
environment overrides keep their framework-defined precedence. It does not
impose OMP/Torch thread counts.

Optional MTP keeps its separate
[metax_mtp.yaml](../../examples/qwen3_8_flash_next/metax_mtp.yaml) profile:

```sh
MODEL_PATH=/path/to/checkpoint \
QWEN38_CACHE_ROOT=/path/to/service-cache \
MTP=k2 sh examples/qwen3_8_flash_next/serve_metax.sh --port 18480
```

MTP accepts k1/k2/k3 and uses MBT2048, synchronous scheduling and shared_buffer
lookup. The shell selects only the native draft count and graph buckets.
Invalid combinations fail in the model's native config verifier. Prefix
caching remains disabled, chunked prefill enabled, and max length is 69632.
A max_num_seqs of 64 does not guarantee that 64 long requests fit in KV cache.

The previous Python plan generator and its --execute/--output-dir/package-root
options are removed. Configure installed dependencies (or an explicitly
selected development PYTHONPATH) before launch. For persistent flag changes,
edit a copy of the native YAML; --additional-config replaces the whole object.

## Sampling and optional MTP

MTP is off by default. MTP=k1, k2 or k3 explicitly enables the corresponding
native draft count, with matching graph buckets. These modes require synchronous
scheduling, shared_buffer lookup and MBT2048. Accepted GDN state and the native
draft acceptance/rejection contract are preserved.

generation_config=auto retains checkpoint sampling defaults. Clients may
override native request fields. The checkpoint thinking defaults used here are
temperature=1, top_p=0.95 and top_k=20; this entry does not force greedy decoding
or change thinking templates. Fixed output length and ignore_eos are benchmark
request choices, not serving defaults.

## N-gram lookup and asynchronous scheduling

shared_buffer uses the existing shared CPU lookup path. uva_broadcast maps the
full table on rank zero and broadcasts fetched rows; uva_sharded maps each
rank's head range and gathers rows across ranks. uva_auto selects these
strategies by the centralized token threshold without changing hash semantics.

The table is loaded once into shared host storage. Native device views retain
the CPU allocation and registration until all dependent work has completed.
The table address remains stable for the service lifetime. Auto mode maps the
full table only on rank zero and local ranges on the other ranks.

This checkpoint needs about 95.37 GiB of shared host storage, sufficient host
RAM/lockable pages, and compatible device registration support. Initialization
errors fail explicitly. The serving implementation does not drop system page
caches, change runtime protection settings or silently switch lookup strategy.

GPU suffix history supplies sampled token IDs to the next asynchronous step;
CPU placeholder token IDs are not used for hashing. Lookup preparation runs on
the ordered current device stream outside the target forward graph. Request
reordering and lifecycle reset retain the existing history/cache contracts.

## MetaX tuning and support boundaries

The [tuning configuration](../../vllm_fl/dispatch/config/qwen3_8_flash_next_metax_tuning.json)
is the source of shape limits, launch parameters and dependency guards.
The launch-profile module selects parameters over existing library kernels;
it is not a torch.profiler observer and does not rewrite library sources.

MTP-off policies cover the selected prefill shapes through 8192 rows, with
dtype/layout/parallelism guards and original fallbacks for unsupported shapes:

- HC grouped normalization, gate reduction, residual injection and divide/SiLU
  preserve BF16 intermediate rounding. Projection and residual order stay intact.
- QSA keeps 512 compressed groups, expansion to 2048 raw tokens plus at most
  three causal tail positions, request/page boundaries and invalid-index masks.
  Short shared-column packing and per-query folding use separate metadata
  contracts. Score buckets include 512/2048/8192 and the current cache capacity.
- Indexer/TopK and attention parameters are selected by actual score width and
  query rows. Small-row decode retains its own split and graph choices.
- GDN reuses chunk metadata already prepared by the framework. CPU-computed
  skew hints avoid device scalar readback. qwen38_gdn_host_metadata=false in additional-config disables the
  skew hint; it does not undo the separate WY chunk-index reuse.
- Eligible BF16 experts consume caller-owned activation/output buffers.
  Non-target layouts, quantized modes, LoRA and unsupported parallelism retain
  their original path.
- MCTlass BF16 projection/expert paths are selected before execution where the
  required capability and shapes match. Execution errors propagate; they do not
  trigger a second backend after computation or state mutation has begun.

## Graphs, diagnostics and evidence

Target pure decode uses FULL_DECODE_ONLY with request buckets 1/2/4/8/16/32/64;
optional MTP multiplies graph token buckets by K+1. Caller-owned workspaces keep
addresses stable. Prefill/mixed remain eager; this entry does not enable full
prefill graphs. Sampling and lookup preparation remain outside the target
forward graph.

Operator recording is off by default; flaggems_record_ops=true is an explicit
diagnostic option and may add host overhead. Benchmark RPC extensions, paired
correctness observers, trace parsers, startup cache advice and container-recovery
scripts are private experiment tools, not installed serving behavior.

Unit tests retain dispatch, fallback/error, mutable-state and ownership
contracts. Numerical and changed-input graph evidence, hardware measurements,
failed experiments and per-change results belong in PR/CI artifacts, not this
user guide. The dependency manifest records any remaining delivery limitations.


### QSA score packing and native attention

The MTP-off MetaX launcher enables qwen38_qsa_native_attention. Target prefill,
mixed and decode use the installed MACA flash_attn_2_cuda varlen API with
four-token virtual pages over the original strided K/V storage. There is no
small-batch performance split. MTP modes keep the previous path.

This is a local VENDOR policy for BF16 Q[rows,3,256], one local KV head and the
QSA four-token block/tail index format. A global FGV operator blacklist cannot
express this model/layout scope while retaining other backends and speculative
paths. Unsupported layouts retain the existing provider.

Indexer batches with request groups use BF16 query packing and skip fully
invisible column tiles. Groups are prepared once within the metadata-builder
scope and shared across QSA layers. Decode uses its measured single-query
launch settings where beneficial; the all-visible <=512-column bucket retains
the existing score/TopK bypass. FP8 cache experiments were not selected.

Native attention scratch belongs to one forward context and CUDA stream,
rather than persisting per layer. Score matrices remain temporary. The selected
numerical sources are qsa_indexer_fast.py and qsa_paged_native.py under FGV's
MetaX backend, with hashes in the tuning profile and dependency manifest.

Native attention preserves the reviewed BF16 module tolerance, rather than
bitwise equality with the previous reduction. Workspace allocation and selected
page metadata belong to the framework; the native numerical API consumes those
buffers. Missing paged API capabilities select the existing provider before
execution. Numerical or metadata execution errors propagate without retry.
The optimized numerical dependencies remain unpublished; this working tree is
not a complete public installation lock.

### Startup installation and failure recovery

Launch presets are constructed before publication. Fixed shape presets live in
a process-local overlay; initialization does not seed them into SQLite.
Unspecified shapes keep the native cache behavior. Failed installation restores
the original entry points, caches and matmul flags; repeated initialization
does not stack wrappers. Model registration and MTP compatibility hooks also
restore their registries and descriptors if setup fails.

These transactions protect initialization only. Exceptions after metadata,
state, kernel or collective execution begins propagate and never retry another
backend. An incompatible dependency fails before starting serving.

Native graph buffers are retained by the model after step metadata is released,
keyed by rows, device, capture stream and score bucket. Eager large-row buffers
remain scoped to a step. In the supported decode configuration, seven row
buckets sum to 127 rows: about 0.436 MiB per score bucket/stream; four score
buckets use about 1.745 MiB of distinct workspace, shared across QSA layers.
The references are released with the model. This does not enable dual-batch
overlap or microbatching, which remain explicitly unsupported.

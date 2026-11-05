# Qwen3.8 Flash Next on MetaX

The supported development composition serves BF16 text on a single MetaX C550
host with TP8/PP1, FP32 recurrent state and vLLM 0.24.0. Multimodal serving is not
accepted by this MetaX profile.

## Installation requirements

Use matching MetaX Torch/MACA and FlagTree 0.6.1+metax3.6, the vLLM base
ee0da84ab9e04ac7610e28580af62c365e898389, and the
[operator requirements](../../examples/qwen3_8_flash_next/metax_tuning/operator_dependencies.json).
Some optimized FG/FGV APIs and the native modular-expert output alias remain
unpublished. The base revisions alone do not form a public optimized installation.
Publish those APIs/tests and pin accessible revisions before public delivery.

UVA additionally requires the plugin host-memory extension built with
VLLM_VENDOR=metax and the matching Torch/MACA ABI. Source guards reject
incompatible numerical sources. Required optimized APIs must still be provided by the validated development composition. Source guards
are compatibility checks, not package installation instructions.

## Start serving

Configure the dependencies and MODEL_PATH, then use the POSIX
[entry](../../examples/qwen3_8_flash_next/serve_metax.sh), which directly executes
native vllm serve:

~~~sh
MODEL_PATH=/path/to/checkpoint \
QWEN38_CACHE_ROOT=/path/to/service-cache \
sh examples/qwen3_8_flash_next/serve_metax.sh --port 18480
~~~

The [default profile](../../examples/qwen3_8_flash_next/metax.yaml) selects MTP off,
MBT8192, max_num_seqs64, asynchronous scheduling and UVA auto. Additional CLI
arguments follow native CLI-over-config precedence. The entry enables FG/FGV,
selects the model dispatch configuration and isolates caches by service/MTP mode.
It preserves the caller's OMP/Torch thread settings.

Optional [MTP profiles](../../examples/qwen3_8_flash_next/metax_mtp.yaml):

~~~sh
MODEL_PATH=/path/to/checkpoint \
QWEN38_CACHE_ROOT=/path/to/service-cache \
MTP=k2 sh examples/qwen3_8_flash_next/serve_metax.sh --port 18480
~~~

MTP accepts k1/k2/k3, requires synchronous scheduling, shared_buffer lookup and
MBT2048, and selects the corresponding draft/graph buckets. Configuration
verification rejects unsupported combinations. Prefix caching is disabled,
chunked prefill is enabled and maximum length is 69632. A max_num_seqs of 64
does not imply that 64 long requests fit in the KV cache.

generation_config=auto retains checkpoint sampling defaults. Clients may
override native request fields. This checkpoint uses temperature=1/top_p=0.95/
top_k=20 for thinking mode. Fixed output length and ignore_eos are request choices.

--additional-config replaces the complete native object. Copy a supplied YAML
when changing persistent settings so required flags remain explicit.

## Lookup and state lifetime

shared_buffer publishes CPU-gathered rows. uva_broadcast maps the full table
on rank zero and broadcasts fetched rows. uva_sharded maps each rank's head range
and gathers rows across ranks. uva_auto selects by the centralized token threshold.

The table uses about 95.37 GiB of shared host storage and requires sufficient
host RAM, lockable pages and device registration support. Storage/registrations
are retained for device access and errors propagate explicitly. The entry does
not alter system page caches or runtime protection settings.

GPU suffix history supplies actual consumed token IDs to asynchronous steps.
Request reordering, reuse and EOS boundaries preserve the history/cache contract.
Lookup preparation remains outside the target forward graph.

## Operator, graph and configuration scope

The [runtime configuration](../../vllm_fl/dispatch/config/qwen3_8_flash_next_metax_tuning.json)
owns shape limits, launch choices and source guards. MTP-off policies cover
validated prefill rows through 8192 with dtype/layout/parallelism guards and
retained fallbacks.

- HC uses the original projection/residual order and BF16 intermediate boundaries.
- QSA selects 512 compressed groups, expands up to 2048 raw tokens plus up to 3
  causal tail positions, and preserves invalid-index/request/page masks.
- MetaX native QSA attention uses the MACA paged varlen API for BF16
  Q[rows,3,256], one local KV head and the four-token block/tail format. MTP uses
  the retained provider. Unsupported layouts select a provider before execution.
- Indexer packing shares groups only within the same request/cache geometry.
  Scratch ownership and stream isolation belong to the framework.
- GDN preserves FP32 state, reuses caller chunk metadata and accepts a CPU skew
  hint. qwen38_gdn_host_metadata=false disables the hint without undoing WY reuse.
- Quantized, LoRA, non-target and unsupported parallel/layout modes retain their
  native contracts. Errors after computation/state updates begin do not retry.

Target pure decode uses FULL_DECODE_ONLY with buckets 1/2/4/8/16/32/64. MTP token
buckets scale by K+1. Prefill/mixed remain eager. Caller-owned workspaces are
prepared before capture and retained for replay, with step/stream isolation.
Dual-batch overlap and microbatching remain unsupported.

Setup hooks install atomically and idempotently. Failed setup restores prior
entry points, caches, registries and matmul flags. Fixed presets are process-local;
unspecified shapes retain native cache behavior.

Operator recording is disabled by default. flaggems_record_ops=true explicitly
enables recording and can add host overhead. Hardware measurements and validation
results are maintained in PR/CI artifacts.

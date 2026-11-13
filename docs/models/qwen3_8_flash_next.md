# Qwen3.8 Flash Next on MetaX

The fixed composition serves BF16 text on a single MetaX C550
host with TP8/PP1, FP32 recurrent state and vLLM 0.24.0. Multimodal serving is not
accepted by this MetaX profile.

## Installation requirements

Use the [fixed dependency manifest](../../examples/qwen3_8_flash_next/metax_tuning/operator_dependencies.json)
and [requirements](../../examples/qwen3_8_flash_next/metax_tuning/requirements.txt).
All three source revisions are publicly accessible; upstream PRs need not be
merged before using these fixed commits.

The matching MetaX Torch/MACA, vendor libraries and FlagTree 0.6.1+metax3.6 are
available in this runtime image (Harbor project pull access is required):

```text
harbor.baai.ac.cn/flagos-inner-models-release/qwen3.8-flash-next-metax@sha256:f8a61974b673afe85a87bda27c1df777c2bfe6ef63f421d7dc335b4f0be53218
```

From the plugin checkout, replace the image's Python packages with the fixed
source revisions and build the plugin's native host-memory extension:

```sh
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.3.5 \
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.24.0+empty \
/work/.venv/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation \
    -r examples/qwen3_8_flash_next/metax_tuning/requirements.txt
VLLM_VENDOR=metax /work/.venv/bin/python -m pip install --force-reinstall \
    --no-deps --no-build-isolation .
```

Use the runtime's Python environment (`/work/.venv/bin/python` in that image).
`VLLM_TARGET_DEVICE=empty` selects the Python-only vLLM build used with MetaX;
the installed plugin selects the real device at runtime. Do not replace the
matching Torch/FlagTree packages with generic CUDA dependencies.

On a host without GitHub access, download the same commit archives on a connected
host and transfer them. Build wheels with the same environment settings above,
then install those wheels using `python -m pip install --force-reinstall
--no-deps --no-index /path/to/wheels/*.whl`. This changes transport only; the
source revisions and vendor runtime must remain identical.

UVA requires the extension built with VLLM_VENDOR=metax and matching Torch/MACA
ABI. The serving entry uses the installed plugin and preserves the caller's
PYTHONPATH. Source guards remain enabled and validate the fixed numerical
implementation before applying the measured launch profile.

## Reproduce the measured launch choices

The manifest also links the C550 launch-selection archive for the fixed runtime.
It contains the tuning database and Triton selection records, without compiled
kernels. To reuse these choices before the first MTP-off service start:

```sh
export QWEN38_CACHE_ROOT=/path/to/service-cache
curl -fL 'https://github.com/cshujie131-bot/vllm-plugin-FL/releases/download/qwen38-metax-vllm024-20261113/qwen38-c550-selected-tuning.tar.gz' -o /tmp/qwen38-c550-selected-tuning.tar.gz
printf '%s\n' 'dc80777db93657cef0c6c7e76295b478dea650c8c85d53f15d4fe4b15c4ec6b6  /tmp/qwen38-c550-selected-tuning.tar.gz' | sha256sum -c -
mkdir -p "$QWEN38_CACHE_ROOT/off"
tar -xzf /tmp/qwen38-c550-selected-tuning.tar.gz -C "$QWEN38_CACHE_ROOT/off"
```

Use these choices with the stated C550/compiler/source combination. The normal
launcher can populate its own cache when the archive is not used.

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

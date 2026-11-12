# BGE-M3 selected joint outputs

The FL BGE-M3 model can return the FlagEmbedding encode result in one vLLM request. One encoder pass produces shared hidden states; only the selected dense, sparse and ColBERT heads run. Native pooling tasks remain available with their existing launch configurations. This is a mapping of the official Python result contract into vLLM's IO processor HTTP transport.

## Launch

From the repository root, load the platform's validated FL/FG/FGV runtime environment first. Allow both plugins (the vLLM allowlist applies to every plugin group):

```bash
python -m pip install -e '.[pooling]' --no-build-isolation
export VLLM_PLUGINS=fl,bge_m3
export VLLM_FL_CONFIG=/path/to/vllm-plugin-FL/vllm_fl/dispatch/config/bge_m3_metax.yaml
vllm serve /models/bge-m3 --config examples/bge_m3_joint/serve.yaml
```

The default `embed` task keeps `/v1/embeddings` usable. `/pooling` requests with `task=plugin` select any combination in the same loaded model. The internal joint pooler is installed only when `io_processor_plugin=bge_m3` is selected; it is not exposed by ordinary launches.

## FP16 dense on MetaX

The [FP16 dense example](../../examples/bge_m3_dense/README.md) adds an explicit
native `/v1/embeddings` profile for MetaX C550. It uses an FP16 encoder, the
existing vendor FlashAttention backend and the default FG Linear path;
pooling heads retain vLLM's FP32 default. The existing FP32 joint-output
launch remains the default example.

The [launch script](../../examples/bge_m3_dense/serve.sh) applies all model
environment settings to the server process, without container ENV or manual
exports. It appends `attention_backend` to
`VLLM_FL_FLAGOS_BLACKLIST_APPEND`, excluding the default FlagOS attention
provider for performance, and selects `impl:vendor.metax` in
`dispatch/config/bge_m3_metax_fp16_dense.yaml`. This replaces FGV
`flash_attn_varlen_func` with the matched vendor implementation for this
profile while retaining other FG operators. No new attention or Linear
kernel is introduced.

Measured 2026-10-28 on C550, TP1/eager, MBT32768, max_num_seqs64,
OMP/MKL16: native CLI C64/N256, one full warmup and one formal round per
length. All four formal rounds completed without request errors or new
tuning/compiler cache entries.

| Input tokens | Total tokens/s |
|---:|---:|
| 1024 | 121,435.85 |
| 2048 | 121,840.69 |
| 4096 | 106,101.43 |
| 8192 | 80,400.07 |

See [the complete performance metrics](bge_m3_fp16_dense_performance.json).
These are hot-cache results with additional untimed worker shape priming.
The earlier FP32 joint-client measurements use different request counts,
inputs and timing boundaries; their historical ratio is not a controlled
FP32-to-FP16 speedup. MBT32768 and fast JSON were already present in
`b9269d2c`.

## HTTP serialization and benchmark

Install the plugin's optional `pooling` extra in the **server environment**.
vLLM 0.24.0's existing `get_json_response_cls()` then selects
`fastapi.responses.ORJSONResponse` for the pooling endpoint at startup.
Restart an existing server after installing the extra. No vLLM patch, custom
response class or model-specific serializer is required. The model still emits
the same Python lists/dictionaries and all FP32 vector values; JSON float
spelling may differ, but decoded values and shapes are preserved.

The [runnable example](../../examples/bge_m3_joint/README.md) includes a shared
`serve.yaml` with `max_num_seqs=64` and `max_num_batched_tokens=32768`.
The larger token budget permits up to 32 complete 1K or eight complete 4K
documents per batch; request concurrency does not guarantee these batch sizes.
It increases device-memory use and should be validated for the target device.

The example client uses `orjson.loads(response.content)`, avoiding the standard
JSON decoder and its intermediate text conversion. Its benchmark reads the
**entire** response and checks every selected vector's shape and finite values
inside the measured interval. The four modes retain identical output semantics;
no token vectors, precision or fields are omitted to improve throughput.

Run 1K and 4K separately, each at HTTP C64/N128, one warmup round and one formal
round per mode. For repeated comparisons, restart independent runs and report
all formal results. Record the source/runtime, server configuration and cache
state alongside the client output. FG can tune previously unseen packed shapes
even after one HTTP warmup: a persistent, prepopulated cache measures steady
state, not cold-start performance. Additional shape priming must be disclosed
separately, and tuning during a timed round must not be silently discarded.

## Request and response

```json
{
  "model": "bge-m3",
  "task": "plugin",
  "data": {
    "sentences": ["什么是向量检索？", "How does retrieval work?"],
    "return_dense": true,
    "return_sparse": true,
    "return_colbert_vecs": true,
    "max_length": 8192
  }
}
```

POST to `/pooling`. `input` is accepted as an alias for `sentences`. The standard vLLM response envelope has `request_id`, `created_at` and `data`; `response.data` has exactly the official keys:

| Key | List input | String input |
|---|---|---|
| dense_vecs | B × 1024 float lists | 1024 floats |
| lexical_weights | B dictionaries of string token IDs to weights | One dictionary |
| colbert_vecs | B arrays, each (L_i − 1) × 1024 | One (L − 1) × 1024 array |

JSON lists represent the NumPy arrays returned by FlagEmbedding. A disabled field is `null`; all eight flag combinations, including all disabled, are accepted. Flags default to `true/false/false`; null flags resolve to those defaults. Sparse weights filter CLS/EOS/PAD/UNK and nonpositive values and merge repeated IDs by maximum. ColBERT removes CLS and batch padding, retaining EOS and literal input tokens.

`max_length` defaults to 512, matching the fixed FlagEmbedding 1.4.2 encode default; explicitly pass 8192 for longer documents. HF tokenization truncates on the right while preserving EOS. This differs from the legacy endpoint's overlong-input rejection. `max_length` must be 2–8192 and no larger than the configured model limit. Empty strings are valid; an empty list, non-text values, unknown options or malformed flags are rejected. An input list of length one retains its batch dimension.

## Implementation and ownership

`BgeM3EmbeddingModelFL` reuses the upstream model, loaders and heads. `BgeM3JointPooler` applies the existing heads to one encoder result and packs selected tensors for vLLM's existing tensor transport. `BgeM3IOProcessor` tokenizes, sets per-request pooling options and restores the official output structure. The private transport header never appears in the public result.

Request options are isolated using ContextVar in vLLM's preprocessing thread; response options travel with each result, not a global FIFO. A private tokenizer and a bounded lock isolate mutable HF truncation settings from concurrent plugin and native requests. No request map is retained across cancellations.

This work is model-specific output semantics and framework orchestration, so it belongs in FL. It contains no kernel implementation, changes no vLLM source file and claims no kernel speedup. The MetaX FP32 path requires [FGV #995](https://github.com/flagos-ai/FlagGems-vllm/pull/995), pinned to `60f1d386c94d1084f3f5f02f71714b7bc6581e6b`. Tests cover all flag combinations, independent reference values, mixed worker batches, concurrency, shapes, errors and legacy registration.

Official contract: https://bge-model.com/bge/bge_m3.html#usage
vLLM IO extension: https://docs.vllm.ai/en/v0.24.0/models/pooling_models/specific_models/

## Ownership layout

- `models/bge_m3/`: model/position semantics, packed attention adapter, joint pooling and BGE IO contract.
- `patches/bge_m3/`: lazy architecture and IO entry-point registration.
- `dispatch/backends/flaggems/impl/bge_m3.py`: registered FGV varlen provider; no kernel implementation.
- `configs/bge_m3_metax.py`: measured Linear launch candidates and scope.
- `models/bge_m3/linear.py`: per-model-instance binding, preserving native methods and parameter ownership.
- `dispatch/backends/vendor/metax/impl/bge_m3.py`: narrow MetaX attention selection and model-owned Linear launch policy, activated by `dispatch/config/bge_m3_metax.yaml`.
- `worker/worker.py`: generic asynchronous pooling-output compatibility, with generic worker tests.
- `tests/unit_tests/models/bge_m3/` and `tests/unit_tests/dispatch/test_bge_m3_policy.py`: model and dispatch contracts.

The public FlagGems backend remains unmodified. Model-owned Linear tuning now lives in plugin configuration and uses the original FG kernel; no separate FG fork or optimization PR is required. FG's existing MetaX masked_fill fix (#6480) passed 72 GPU cases, so no model blacklist or new implementation is needed.

FP32 encoder attention calls the existing FGV flash_attn_varlen_func once per packed batch. [FGV #995](https://github.com/flagos-ai/FlagGems-vllm/pull/995) extends the already merged MetaX implementation (#808) with IEEE FP32 QK/PV, a resource bound for wider heads, and explicit unsupported-feature errors. It supports non-paged bidirectional forward attention, unequal query/key lengths, GQA, D=16/32/64/128/256, packed/strided inputs, output buffers and LSE. The FP32 support introduces no torch compute fallback. Its subsequent context-length launch configuration is included in the same FGV PR. Causal/paged/windowed attention, ALiBi, softcap and backward remain outside the new FP32 scope; existing half paths are preserved.

Upstream #759 was tested unchanged on C550: FP16 passed, but FP32 required 72 KiB shared memory and failed against the 64 KiB limit. FlagTree #1191/#1258 were tested as a minimal Python backport, but did not fix the nested-helper tiny SDPA case; the backport was reverted. The new packed path does not call that SDPA kernel or need its former AABS workaround. The required FGV dependency is pinned below; FL contains only its public-op integration.


## Model-owned Linear tuning

`configs/bge_m3_metax.py` contains the two additional M32/N64/K32/W8/S4 and M64/N64/K32/W8/S4 candidates, the M<=1024 limit, and the measured `(N,K)` pairs `(3072,1024)`, `(1024,1024)`, `(4096,1024)`, `(1024,4096)`.

The existing `bge_m3_metax.yaml` policy selects `bge_m3_linear`. The provider is resolved once at model construction, avoiding per-call policy lookup. Binding is limited to this BGE model's unquantized Linear instances, including the ColBERT projection; parameter objects, weight loading and native layer methods are retained. The vendor provider builds an independent tuner over FG's existing `linear_kernel`, copying candidate objects before adding the model settings. No FG global function, candidate list or kernel is replaced. vLLM weight wrappers are exposed to the launcher through a no-copy Tensor view; the layer retains its original Parameter and storage.

Only contiguous FP32 CUDA inference in the measured region uses these settings. M is the total flattened row count, so a packed eight-document 1K encoder batch has M=8192 and keeps its native method. Other shapes/dtypes, gradients, noncontiguous tensors and batch-invariant mode retain the original layer call. The default policy also retains that call. Training, compile/graph and additional hardware are not new validation targets.

The FGV dependency below includes the existing FP32 attention support and its context configuration. FG remains at the original baseline revision. Pure model parameter tuning is maintained in this model PR, not a separate operator PR.

## Validated dependency pins

| Component | Revision |
|---|---|
| vLLM | 0.24.0, ee0da84ab9e04ac7610e28580af62c365e898389 |
| orjson (optional pooling extra) | 3.11.4 validated |
| FlagGems | 4897cc306cd83390c95b83b50df28f7edb1684c5 |
| FlagGems-vllm | [#995](https://github.com/flagos-ai/FlagGems-vllm/pull/995), [60f1d386c94d1084f3f5f02f71714b7bc6581e6b](https://github.com/flagos-ai/FlagGems-vllm/commit/60f1d386c94d1084f3f5f02f71714b7bc6581e6b) |

Validation used MetaX C550 64 GiB, MACA 3.8.1.3, Torch 2.10.0+metax3.8.1.0 and FlagTree 0.6.1+metax3.6. Preserve the matched vendor runtime. The FGV PR is not merged; install the exact public-fork revision without replacing its vendor dependencies:

```bash
python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/cshujie131-bot/FlagGems-vllm.git@60f1d386c94d1084f3f5f02f71714b7bc6581e6b'
```

## Steady-state full-response HTTP performance

Measured 2026-10-27 (Asia/Shanghai) on node 102, MetaX C550 64 GiB,
with two Intel Xeon Platinum 8460Y+ CPUs. Server and client use same-host
loopback HTTP. FP32/TP1/eager, max_model_len8192, max_num_seqs64,
gpu_memory_utilization0.25, prefix caching/chunked prefill disabled.
The benchmark uses complete official JSON, including all ColBERT token vectors,
with shape/finite checks inside timing.

1K and 4K are separate tests. Each independent service run has one warmup
and one formal round per output, in dense/sparse/ColBERT/joint order, C64/N128,
one document/request. Each fast-JSON MBT has three independent runs; the
standard-JSON baseline has one. Fast columns are means ± sample standard
deviations across all three formal runs. The interleaved fast-JSON MBT order is
32768,8192,8192,32768,32768,8192 for each length.

| Tokens/document | Output | Standard JSON MBT8192 docs/s | Fast JSON MBT8192 docs/s | Fast JSON MBT32768 docs/s | Combined change | MBT-only change |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | dense | 33.3858 | 33.4544 ± 0.0091 | 34.5135 ± 0.0159 | +3.38% | +3.17% |
| 1024 | sparse | 34.0947 | 34.0154 ± 0.0444 | 35.0541 ± 0.0417 | +2.81% | +3.05% |
| 1024 | colbert | 1.6092 | 4.8383 ± 0.0176 | 4.9362 ± 0.0802 | +206.75% | +2.02% |
| 1024 | joint | 1.6399 | 4.7459 ± 0.0108 | 4.9067 ± 0.1095 | +199.22% | +3.39% |
| 4096 | dense | 4.9867 | 4.9891 ± 0.0011 | 5.1420 ± 0.0019 | +3.11% | +3.06% |
| 4096 | sparse | 5.0006 | 4.9946 ± 0.0108 | 5.1554 ± 0.0009 | +3.10% | +3.22% |
| 4096 | colbert | 0.3690 | 1.1133 ± 0.0179 | 1.1367 ± 0.0027 | +208.04% | +2.10% |
| 4096 | joint | 0.3724 | 1.1051 ± 0.0121 | 1.1309 ± 0.0106 | +203.69% | +2.33% |

**Steady-state cache contract:** every run uses an independent copy of the same
prepopulated cache, including explicit all-shape diagnostic priming and
correctness validation. There are no new/changed tuning records in any timed
warmup/formal run. These results do not measure cold-start or replace the
historical B0 below. One HTTP warmup alone does not ensure coverage of new
dynamic shapes.

All 7,168 formal and 7,168 warmup requests passed; each round reached an actual
HTTP in-flight peak of 64, with zero errors/retries. The source passed 105
targeted tests and full HTTP reference validation, including all eight output
combinations, mixed concurrency, legacy dense and complete 1K/4K JSON codec
equivalence. Model, worker and FG/FGV compute source is unchanged by this update.

See [the machine-readable results](bge_m3_http_performance.json) for every
independent throughput value, per-run latency/response bytes, sampled GPU
memory, execution-file hashes and fixed runtime/operator provenance.

## Out-of-box performance: historical unoptimized B0

These are the previously measured **unoptimized B0** results from 2026-10-23, before the joint-output interface and packed FP32 varlen integration in this PR. Each mode ran as a separate pooling task. They are a baseline reference, not throughput measurements of the current joint endpoint. Machine-readable provenance and all six rows are in [bge_m3_baseline.json](bge_m3_baseline.json).

C550 64 GiB, TP1, FP32, eager, max_num_seqs=8, max_num_batched_tokens=8192; 64 requests, concurrency 8, one document per request, one warmup round and one measured round, HTTP keepalive=600 seconds. The client read and parsed complete JSON outputs; sparse filtering/duplicate-ID max are included. All six cases completed 64/64 with no retries or failures.

| Output | 1024 tokens/document, docs/s | 4096 tokens/document, docs/s |
|---|---:|---:|
| dense | 28.698 | 5.167 |
| sparse | 28.764 | 5.166 |
| ColBERT | 1.854 | 0.426 |

docs/s measures input-document throughput. ColBERT includes its full token-vector response (about 22.15/88.63 MB per document at 1K/4K), so these HTTP numbers are not encoder-only compute throughput. These historical B0 numbers do not measure the later model-owned Linear configuration or FGV context configuration.

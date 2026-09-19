# Common worker optimization contracts

These two opt-in policies are independent. Choose settings before creating a
worker; changing a running engine's environment is not a reconfiguration API.

## Attention metadata

`VLLM_FL_COMMON_ATTENTION_METADATA` accepts exactly:

| Value | Producer | Execution |
|---|---|---|
| `stock` (default), `0` | Original slot mapping and padding cleanup | Original worker path |
| `eager` | Common Triton producer | Eager, even if the model uses graphs |
| `graph`, `1` | Common Triton producer | Standalone graph when available and captured; otherwise eager |

The mode is resolved once at construction and logged with its reason. Graph
capability comes from the platform's graph APIs, without another vendor list.
This is separate from validating the producer on a device. Opting in on an
unvalidated device requires platform-specific correctness testing. Ubatching/DBO
and async speculative decode currently resolve to `stock`, with a logged reason,
until their scheduling integration is validated. GPU execution failures propagate;
there is no catch-and-continue fallback after an execution error.

FULL and PIECEWISE warmup/capture initialize the producer independently of model
attention capture. PIECEWISE uses the fixed maximum request extent, since its
model graph keys describe token counts rather than request counts. Updated
`query_start_loc` supplies the actual token boundary at replay time.

The graph runner owns one InputBatch generation. Synchronize and call `clear()`
before replacing any input/output buffer, block-table storage, block size, or CP
geometry. InputBatch replacement and profiling/shutdown cleanup do this explicitly.
Graphs retain their tensor references and use generation/request-extent keys.
With `VLLM_LOGGING_LEVEL=DEBUG`, replay also checks data pointers, shapes, strides,
dtypes, devices and block/CP geometry, and rejects an unannounced change.

The producer returns a `PreparedMetadata` receipt containing its generation,
step sequence and prepared request/token extent. Consumers validate it before
skipping slot/padded-row cleanup or using the computed-token cache. A receipt
expires on the next producer invocation or invalidation. The stock path passes
no receipt and retains the original cleanup responsibilities.

## Shape-aware MM

`VLLM_FL_FLAGOS_MM_SHAPE_AWARE=1` opts in; the common worker default is disabled.
The switch accepts `0`, `1`, `false` and `true`. The positive integer
`VLLM_FL_FLAGOS_MM_DECODE_MAX_M` defaults to 64. Eligible CUDA FP16/BF16/FP32
matrices with M at or below that threshold use the captured native CUDA kernel;
other shapes use the captured FlagGems kernel. Existing whitelist/blacklist
selection can exclude MM entirely.

The policy is immutable for the process lifetime. Repeated worker initialization
with identical settings checks the dispatcher registration and returns
`already_active`, without running FlagGems registration again. Changing enable,
threshold, USE_FLAGGEMS or backend selection requires a fresh process. External
registration changes raise `conflicting_owner`. Worker shutdown intentionally
does not uninstall a process-wide kernel; the retained handles remain alive.
A failed registration cannot be retried in the same process.

The wrapper uses `SafeKernelFunction.call_boxed` handles captured before and after
FlagGems registration. FlagGems receives an observed `torch.library.Library`
through its `lib=` argument. The observer records the callable and boxed handle
only after successful CUDA MM registration, after vendor and condition filtering.
Initialization checks the current registration stack, callable provider and a
distinct native backend before installing the wrapper. Startup status records
both backend identities and the routing threshold. Ownership checks happen at
worker initialization, not on every MM invocation; external dispatcher mutation
while a worker is running is unsupported.

Validation uses real dispatcher registrations in isolated processes, GPU graph
capture/replay, and worker capture entry points. Numerical checks and performance
measurements for a particular model are separate requirements; these contracts
do not imply that every model/platform/scheduling combination is validated.

See [the recorded validation and reproduction commands](common_worker_validation.md)
for the tested snapshot, installed-wheel checks, and scheduling limitations.

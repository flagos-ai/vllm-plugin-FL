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

Mixed FULL model graphs are experimental and emit a startup warning after the
model's graph policy is resolved. This does not warn for FULL_DECODE_ONLY,
whose decode execution also uses the FULL runtime mode. The full-worker
acceptance probe never reports mixed FULL as accepted; stock metadata does
not fix the known model-graph reuse issue.

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

`flaggems_runtime.configure_flaggems` owns process initialization, configuration
and failed-initialization state. The worker resolves platform, model and
deployment operator policy before calling it; the MM module retains only its
kernel handles and shape selector. When FlagGems or MM is disabled, or MM is
excluded, the MM threshold is not parsed.

The active policy is immutable for the process lifetime. Repeated worker initialization
with identical settings checks the dispatcher registration and returns
`already_active`, without running FlagGems registration again. Changing enable,
threshold, USE_FLAGGEMS or backend selection requires a fresh process. External
registration changes raise `conflicting_owner` on the tested PyTorch 2.11
adapter. Other Torch builds can use the public installation APIs but have no
verified repeated-initialization ownership check. Worker shutdown intentionally
does not uninstall a process-wide kernel; the retained handles remain alive.
A failed registration cannot be retried in the same process.

The wrapper uses `SafeKernelFunction.call_boxed` handles captured before and after
FlagGems registration. FlagGems receives an observed `torch.library.Library`
through its `lib=` argument. The observer records the callable and boxed handle
only after successful CUDA MM registration, after vendor and condition filtering.
Successful registration through this library establishes which callable and
boxed handle were installed. Kernel repr strings and source-package paths are
diagnostic only and never gate initialization. The registration-stack check
is restricted to the PyTorch 2.11 adapter. Startup status records the selected
backends and routing threshold. Ownership checks happen at
worker initialization, not on every MM invocation; external dispatcher mutation
while a worker is running is unsupported.

Validation uses real dispatcher registrations in isolated processes, GPU graph
capture/replay, and worker capture entry points. Numerical checks and performance
measurements for a particular model are separate requirements; these contracts
do not imply that every model/platform/scheduling combination is validated.

See [the recorded validation and reproduction commands](common_worker_validation.md)
for the tested snapshot, installed-wheel checks, and scheduling limitations.

## Integration owner

PR544 owns the metadata policy, receipt and graph lifetime. PR455 uses that
same owner for ordinary and packed block-table storage; there is no separate
packed policy or graph cache. Padding clears a group's logical width, which
can be smaller than its row stride in packed storage. Functional tests protect
neighboring groups, real request rows and allocation guards during replay.

PR442's fused multi-group producer is now upstream in main `71f6148` and is
retained under this shared owner. The merge uses logical group widths and
retires the producer's pointer tables with the InputBatch generation. One
explicit stock/eager/graph parser and one graph owner serve both ordinary
and packed storage. The historical local-fusion benchmark uses this same
producer; publication inherits PR442 through main.

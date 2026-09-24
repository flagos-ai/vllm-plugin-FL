# Common attention metadata graphs

The producer writes slot mappings, clears padded block-table rows, and updates
GPU computed-token counts using persistent input and output buffers. Capture
warms the kernel outside the graph and replays the new graph before returning,
so attention builders always consume current metadata, including with zero
model warmups. Dummy model forwards still use PAD_SLOT_ID for every token.

FULL and PIECEWISE model modes both capture this independent metadata graph.
PIECEWISE uses the runner's maximum request extent because its model graph
shapes describe tokens, not request counts. Both real and dummy steps initialize
the complete query-start and sequence-length tails. Changing actual request
counts or lengths therefore reuses the same metadata graph without capturing
on the request path. FULL retains request-count keys. Microbatching uses the
stock path; model eager mode uses eager metadata generation when the common
producer is enabled. Profiling cleanup drops graphs and pointer tables.

`VLLM_FL_COMMON_ATTENTION_METADATA=0` restores the original per-group
`BlockTable.compute_slot_mapping` path, builder padding, and computed-token cache
behavior. The default is `stock` on every platform. `eager` opts into the
pointer-table Triton producer without metadata graph capture; `graph` or `1`
also enables metadata graphs. Graph support is checked separately through the
platform graph API. An unavailable graph uses the producer eagerly only when
that producer is enabled. See [the common worker contracts](common_worker_optimizations.md)
for unsupported scheduling combinations, receipt validation and buffer lifetime.

The new kernel uses uint64 device pointer tables. Exposing a graph API alone
does not establish support for that kernel on a different compiler or device.
MUSA PIECEWISE and other vendor hardware require their own device validation;
CUDA tests cannot establish their numerical or runtime compatibility.

The functional GPU tests invoke the producer explicitly and require a supported
device. Run them as part of validating a new vendor; opt-in and graph API
availability alone are not a compatibility claim.

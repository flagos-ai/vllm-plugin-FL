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
on the request path. FULL retains request-count keys. Microbatching and model
eager mode use eager metadata generation. Profiling cleanup drops cached graphs.

`VLLM_FL_COMMON_ATTENTION_METADATA=0` restores the original per-group
`BlockTable.compute_slot_mapping` path, builder padding, and computed-token cache
behavior. NVIDIA enables the new producer by default. Other platforms retain
the original producer until validated; `VLLM_FL_COMMON_ATTENTION_METADATA=1`
opts into the new pointer-table Triton kernel for validation. Graph support is
checked separately through the platform graph API. An unavailable graph uses
the new producer eagerly only when that producer is enabled.

The new kernel uses uint64 device pointer tables. Exposing a graph API alone
does not establish support for that kernel on a different compiler or device.
MUSA PIECEWISE and other vendor hardware require their own device validation;
CUDA tests cannot establish their numerical or runtime compatibility.

The producer's functional GPU tests follow the same enable policy. On other
vendors, explicitly set `VLLM_FL_COMMON_ATTENTION_METADATA=1` to run those
tests while validating the new kernel; the default suite retains the original
producer's platform contract.

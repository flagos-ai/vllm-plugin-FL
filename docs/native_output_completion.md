# Experimental native async-output completion

This opt-in subsystem was extracted from PR455. Qwen throughput acceptance
uses accelerator Event synchronization and does not depend on this feature.
`VLLM_FL_ASYNC_OUTPUT_NATIVE_COMPLETION=0` remains the default.

With `=1`, the NVIDIA extension schedules a native CUDA host callback that
writes an eventfd. The output thread polls with a deadline and checks the
copy event for accelerator errors. A timeout, cancellation, context error or
missing notification retires the descriptor instead of allowing a late
callback to write into a recycled descriptor. Shutdown cancels waits; the
fixed pool bounds quarantined descriptors to 64 per process.

The CPU suite exercises real Linux eventfds, timeout, cancellation, partial
construction, late notification and pool shutdown (14 tests). It does not
establish CUDA-context fault behavior or performance benefit.

This feature remains a draft pending a source-bound native extension build,
real CUDA copy/callback integration, context-fault injection and an Event
control benchmark on the same workload. Do not infer those results from
PR455's Event-based serving evidence.

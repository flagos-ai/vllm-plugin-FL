# Qwen3.8 Flash Next MetaX tuning configuration

Use [serve_metax.py](../serve_metax.py) to select the model configuration.
It defaults to MTP off, synchronous scheduling, shared CPU lookup and MBT2048;
MTP-off deployments may explicitly select async scheduling, UVA auto and
MBT4096/8192. The [model guide](../../../docs/models/qwen3_8_flash_next.md)
describes build requirements, interfaces and supported boundaries.

Parameters live in
vllm_fl/dispatch/config/qwen3_8_flash_next_metax_tuning.json.
The launcher supplies this directory's moe-configs through
VLLM_TUNED_CONFIG_FOLDER. Reviewed FG LibTuner entries and FGV launch parameters
are selected before computation; shape misses retain the existing path.

QSA retains the fixed selection budget and independent decode/prefill policy.
HC, projection and expert shape limits extend through 8192 prefill rows.
The model owns metadata, graph workspace lifetime and state; numerical kernels
remain in FG/FGV or the native vendor libraries. No source replacement scripts
are shipped and no prefill graph mode is enabled.

[operator_dependencies.json](operator_dependencies.json) describes the actual
development assembly, including unpublished library/native API requirements.
Source hashes are reproduction guards, not public installable versions.
manifest.json records the historical tuning base, not the complete current
deployment lock.

Use independent writable caches for concurrent service experiments. Triton and
optional operator-hit outputs use launcher-selected mode directories; FG/FGV
also use their native caches. One-off benchmarks, observers, trace processing
and startup recovery tools remain outside the model repository.

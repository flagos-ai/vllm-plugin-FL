# Qwen3.8-Flash-Next MetaX launch profile

This profile builds on [plugin-FL #455](https://github.com/flagos-ai/vllm-plugin-FL/pull/455).
It retains the validated BF16/TP8 C550 parameters, shared CPU N-gram lookup,
K1 MTP graphs and context-width QSA buckets.

Current operator requirements are listed in the model guide's
[runtime and operator dependencies](../../../docs/models/qwen3_8_flash_next.md#runtime-and-operator-dependencies).
The manifest in this directory records the historical tuning base only.
The per-row boundary fix is published in FGV #1024; the required MetaX radix
candidate still needs a public fixed reference. Operator implementations remain
in their operator library.

Add these settings to the existing serving configuration:

```json
{
  "additional_config": {
    "qwen38_ngram_mode": "shared_buffer",
    "qwen38_mtp_graph": true,
    "qwen38_qsa_length_buckets": true,
    "qwen38_tuning_profile": true
  }
}
```

Point VLLM_TUNED_CONFIG_FOLDER to this directory's moe-configs folder.
Use a dedicated container and keep the native FG and FGV caches separate
(defaults: $HOME/.flaggems and $HOME/.flagdnn). Do not export a common
FLAGGEMS_CACHE_DIR or FLAGGEMS_DB_URL for both libraries.

The model installs the profile before constructing layers and capturing graphs.
Reviewed linear/MV choices use native LibTuner/ConfigCache entries. Remaining
presets select launch metadata on existing FG/FGV/vLLM kernels. GDN output
dispatch keeps a native cache per context bucket. The profile does not rewrite
third-party source files or ship diff files; kernel computations remain in their
original libraries. Geometry misses keep the native path.

Parameters are packaged in
vllm_fl/dispatch/config/qwen3_8_flash_next_metax_tuning.json.
QSA preserves the K512 selection contract and cache capacity. Widths N<=512
skip score/TopK; supported M<=128 query-row shapes use corrected non-TLE FGV
per-row selection, larger M uses radix, and other shapes retain the FG fallback.
Score widths remain 512/2048/8192/cache-capacity. Target and MTP graph keys include the selected width.
No new communication optimization or operator exclusion is included.

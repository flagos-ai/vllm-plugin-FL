# Qwen3.8-Flash-Next MetaX launch profile

This profile builds on [plugin-FL #455](https://github.com/flagos-ai/vllm-plugin-FL/pull/455).
It retains the validated BF16/TP8 C550 parameters, shared CPU N-gram lookup,
K1 MTP graphs and context-width QSA buckets.

Install the pinned runtime and operator prerequisites in manifest.json:
FGV base 0f3df8e, the PLE fix from [#959](https://github.com/flagos-ai/FlagGems-vllm/pull/959)
and the MetaX inplace-MoE fix from [#1000](https://github.com/flagos-ai/FlagGems-vllm/pull/1000).
These are operator-library dependencies, separate from the model launch profile.

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
QSA retains K512 and the existing two-stage selection algorithm; score widths are
512/2048/8192/cache-capacity. Target and MTP graph keys include the selected width.
No new communication optimization or operator exclusion is included.

# Qwen3.8-Flash-Next MetaX launch profile

This profile builds on [plugin-FL #455](https://github.com/flagos-ai/vllm-plugin-FL/pull/455).
It retains the validated BF16/TP8 C550 parameters, shared CPU N-gram lookup,
optional MTP graphs and context-width QSA buckets. The executable
[serving entry point](../serve_metax.py) defaults to MTP off and enables K1/K2/K3
only when explicitly selected; it keeps checkpoint sampling defaults.

See the model guide's
[runtime and operator dependencies](../../../docs/models/qwen3_8_flash_next.md#runtime-and-operator-dependencies)
for the current required composition and unpublished dependency status.
This directory's `manifest.json` records the historical tuning base only; it
does not provide a complete current deployment lock.

Use the [serving entry point](../serve_metax.py) to select this profile. It owns
the serving configuration, enables the shared N-gram and QSA tuning settings,
and pairs each explicit MTP choice with its matching graph geometry. There is
no separate manual `additional_config` recipe in this profile.

The launcher selects this directory's `moe-configs` through
VLLM_TUNED_CONFIG_FOLDER. FG and FGV use distinct native caches
(`$HOME/.flaggems` and `$HOME/.flagdnn`); the deployment supplies separate
container filesystems or homes when isolation between services is needed.
Triton and the operator-hit log use the launcher's mode-specific cache paths.

The model installs the profile before constructing layers and capturing graphs.
Reviewed linear/MV choices use native LibTuner/ConfigCache entries. Remaining
presets select launch metadata on existing FG/FGV/vLLM kernels. GDN output
dispatch keeps a native cache per context bucket. The profile does not rewrite
third-party source files or ship diff files; kernel computations remain in their
original libraries. Geometry misses keep the native path.

Parameters are packaged in
vllm_fl/dispatch/config/qwen3_8_flash_next_metax_tuning.json.
QSA retains K512 and the reviewed shape routing: width N<=512 skips score/TopK;
otherwise supported M<=128 query-row shapes use corrected FGV non-TLE per-row selection,
larger M uses radix, and out-of-scope shapes retain FG fallback. Score widths
are 512/2048/8192/cache-capacity. Target and MTP graph keys include the selected width.
No new communication optimization or operator exclusion is included.

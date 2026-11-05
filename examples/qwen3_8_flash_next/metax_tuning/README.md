# Qwen3.8 Flash Next MetaX configuration

[serve_metax.sh](../serve_metax.sh) directly executes native vLLM.
The default profile is MTP off, asynchronous scheduling, UVA auto and MBT8192.
MTP=k1/k2/k3 selects synchronous scheduling, shared_buffer and MBT2048.

Shape limits, source guards and launch parameters have one runtime source:
[vllm_fl/dispatch/config/qwen3_8_flash_next_metax_tuning.json](../../../vllm_fl/dispatch/config/qwen3_8_flash_next_metax_tuning.json).
The launcher passes [moe-configs](moe-configs/) to VLLM_TUNED_CONFIG_FOLDER.

Use the [model guide](../../../docs/models/qwen3_8_flash_next.md) for installation,
commands, supported contracts and the remaining
[dependency gaps](operator_dependencies.json). Independent services need
independent writable cache directories.

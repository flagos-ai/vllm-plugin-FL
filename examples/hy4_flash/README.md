# Hy4 Flash on NVIDIA H100

Use vLLM 0.32.0 with the vllm-plugin-FL 0.5.0-dev base
`814b6d3e5ce7de251f9b503ff212bf3fd7cf3a48` and the vLLM V2 model runner.
Start from the validated CUDA and DeepGEMM environment with eight H100 GPUs.
Apply the separate local IndexPool operator patch to FlagGems-vLLM at
`19eb1dd3e43341d951be9a69341b90bdf3d5dcfb` and install that checkout.
This commit includes the public grouped routing padding change. The local
operator patch supplies the checkpoint-specific IndexPool implementation.
The plugin dispatches its kernels through the installed FlagGems-vLLM package.

Run the example from the plugin repository and set `MODEL_DIR` to the local
checkpoint directory. It uses the FP8 validation service settings and binds to
loopback.

```bash
export VLLM_PLUGINS=fl
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_FL_CONFIG="$PWD/examples/hy4_flash/dispatch.yaml"
export VLLM_FL_FLAGOS_WHITELIST=grouped_topk,hy4_flash_fwht_quant_fp8,hy4_flash_gated_pool,hy4_flash_rotate_indexer,hy4_flash_select_groups,hy4_flash_expand_groups,hy4_flash_write_pool,hy4_flash_prepare_groups,hy4_flash_write_tails
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
export FLASH_MODE=load HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export FLASHINFER_WORKSPACE_BASE=/tmp

vllm serve "${MODEL_DIR:?Set MODEL_DIR to the checkpoint directory}" \
  --served-model-name Hy4-Flash-FP8-Testing-FlagOS \
  --host 127.0.0.1 --port 8000 \
  --tokenizer-mode hy_v4 --dtype bfloat16 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --distributed-executor-backend mp \
  --max-model-len 204800 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype auto \
  --no-enable-prefix-caching --limit-mm-per-prompt '{"image":16,"video":0}' \
  --reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4 \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64]}' \
  --seed 1 --no-enable-log-requests
```

The W8A8 checkpoint was smoke-tested separately in eager mode, with the
checkpoint retaining its INT8 MoE weights. The FP8 example above reflects
the resident evaluation profile.

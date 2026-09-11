export USE_FLAGGEMS=true
# PR #5437 explicitly excludes expert_map. Keep its fused W4A16 path opt-in
# for non-EP deployments; TP8/EP8 uses the offline native-Marlin sidecar.
export VLLM_FL_FLAGGEMS_FUSED_MARLIN="${VLLM_FL_FLAGGEMS_FUSED_MARLIN:-0}"
export VLLM_FL_FLAGGEMS_FUSED_MARLIN_PATH=/public-nvme/models/DeepSeek-V4-Flash-0731-INT-FlagGems-Marlin-TP8
export VLLM_FL_OFFLINE_MARLIN=1
export VLLM_FL_OFFLINE_MARLIN_PATH=/public-nvme/models/DeepSeek-V4-Flash-0731-INT-Marlin-TP8

# Keep known unsafe/regressive non-MoE overrides on native CUDA.  The DSV4
# projection has its own logical op name, and the W8A8 selector must also be
# disabled so compile uses vLLM's graph-safe CUTLASS kernel.  Marlin MoE and
# the remaining supported operators stay on FlagGems.
if [[ "$VLLM_FL_FLAGGEMS_FUSED_MARLIN" == "1" ]]; then
  export VLLM_FL_FLAGOS_BLACKLIST=scaled_mm,deepseek_v4_int8_scaled_mm,w8a8_dynamic_per_token_linear,empty,sum
else
  export VLLM_FL_FLAGOS_BLACKLIST=scaled_mm,deepseek_v4_int8_scaled_mm,w8a8_dynamic_per_token_linear,empty,moe_sum,sum
fi
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

vllm serve /public-nvme/models/DeepSeek-V4-Flash-0731-INT \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'

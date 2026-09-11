export USE_FLAGGEMS=true
# Keep known unsafe/regressive non-MoE overrides on native CUDA.  The DSV4
# projection has its own logical op name, and the W8A8 selector must also be
# disabled so compile uses vLLM's graph-safe CUTLASS kernel.  Marlin MoE and
# the remaining supported operators stay on FlagGems.
export VLLM_FL_FLAGOS_BLACKLIST=scaled_mm,deepseek_v4_int8_scaled_mm,w8a8_dynamic_per_token_linear,empty,sum
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# FlagGems PR #5437 consumes output-major GPTQ nibbles rather than vLLM 0.24's
# post-repack int32 CUDA-Marlin layout. Preserve that representation across
# vLLM's framework initialization and dispatch the fused MoE to FlagGems.
export VLLM_FL_FLAGGEMS_FUSED_MARLIN=1
export VLLM_FL_OFFLINE_MARLIN=0

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

export USE_FLAGGEMS=false
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# The rank-sharded checkpoint stores MoE weights after vLLM's Marlin repack.
# Keep this paired with --load-format sharded_state and TP/EP size 8.
export VLLM_FL_OFFLINE_MARLIN=1

vllm serve /public-nvme/models/DeepSeek-V4-Flash-0731-INT-Marlin-TP8 \
  --load-format sharded_state \
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

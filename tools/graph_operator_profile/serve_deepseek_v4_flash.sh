#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${PROFILE_RUN_ROOT:-/vllm-workspace/graph_operator_profile_runs}
RUN_DIR="$RUN_ROOT/deepseek_v4_flash"
PROFILE_DIR="$RUN_DIR/profile"
CAPTURE_DIR="$PROFILE_DIR/capture_traces"
if [[ -d "$RUN_DIR" ]]; then
  archive="$RUN_ROOT/archive/deepseek_v4_flash_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$(dirname "$archive")"
  mv "$RUN_DIR" "$archive"
fi
mkdir -p "$PROFILE_DIR" "$CAPTURE_DIR"

printf -v PROFILER_CONFIG '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_with_memory":false,"ignore_frontend":true}' "$PROFILE_DIR"

export VLLM_PLUGINS=fl
export VLLM_FL_GRAPH_CAPTURE_PROFILE_DIR="$CAPTURE_DIR"
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

exec vllm serve /models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --safetensors-load-strategy prefetch \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cudagraph_num_of_warmups":0}' \
  --profiler-config "$PROFILER_CONFIG" \
  > "$RUN_DIR/serve.log" 2>&1

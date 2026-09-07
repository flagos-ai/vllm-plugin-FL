#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$script_dir/all_on_env.sh"

artifact_root=${ARTIFACT_ROOT:-/artifact}
runtime_dir="$artifact_root/runtime"
mkdir -p "$runtime_dir"

# Repeat the identity check immediately before server exec so that the server
# and its worker children use the same mounted trees that were audited.
python3 "$script_dir/identity_preflight.py" \
  --model "$MODEL_CONTAINER" \
  --plugin-root /opt/vllm-plugin-FL \
  --flaggems-root /opt/FlagGems \
  --output "$runtime_dir/source_identity_at_server_start.json" \
  --expected-init-sha256 "$EXPECTED_FLAGGEMS_INIT_SHA256" \
  --expected-plan-cache-sha256 "$EXPECTED_FLAGGEMS_PLAN_CACHE_SHA256" \
  --expected-gpu-count "$EXPECTED_GPU_COUNT" \
  --expected-gpu-substring "$EXPECTED_GPU_SUBSTRING"

server_cmd=(
  python3 -m vllm.entrypoints.cli.main serve "$MODEL_CONTAINER"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --distributed-executor-backend mp
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --no-enable-prefix-caching
  --disable-custom-all-reduce
  --moe-backend triton
  --language-model-only
  --profiler-config.profiler cuda
  --compilation-config "$COMPILATION_CONFIG"
)

{
  printf 'model=%q\n' "$MODEL_CONTAINER"
  printf 'served_model_name=%q\n' "$SERVED_MODEL_NAME"
  printf 'plugin_image=%q\n' "${PLUGIN_IMAGE:-unknown}"
  printf 'plugin_image_id=%q\n' "${PLUGIN_IMAGE_ID:-unknown}"
  printf 'source_plugin=%q\n' /opt/vllm-plugin-FL
  printf 'source_flaggems=%q\n' /opt/FlagGems
  printf 'PYTHONPATH=%q\n' "$PYTHONPATH"
  printf 'VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE=%q\n' "$VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE"
  printf 'VLLM_FL_FLAGOS_BLACKLIST=%q\n' "$VLLM_FL_FLAGOS_BLACKLIST"
  printf 'QWEN4_HC_BACKEND=%q\n' "$QWEN4_HC_BACKEND"
  printf 'QWEN4_QSA_FUSED_COMPRESS=%q\n' "$QWEN4_QSA_FUSED_COMPRESS"
  printf 'QWEN4_QSA_MQA_DOT=%q\n' "$QWEN4_QSA_MQA_DOT"
  printf 'QWEN4_QSA_SPLIT_TOPK=%q\n' "$QWEN4_QSA_SPLIT_TOPK"
  printf 'QWEN4_QSA_SPLIT_REQUIRE=%q\n' "$QWEN4_QSA_SPLIT_REQUIRE"
  printf 'VLLM_FL_PACKED_BLOCK_TABLE_ARENA=%q\n' "$VLLM_FL_PACKED_BLOCK_TABLE_ARENA"
  printf 'VLLM_FL_PACKED_BLOCK_TABLE_REQUIRE=%q\n' "$VLLM_FL_PACKED_BLOCK_TABLE_REQUIRE"
  printf 'VLLM_FL_GDN_STRICT_PATCH=%q\n' "$VLLM_FL_GDN_STRICT_PATCH"
  printf 'QWEN4_QSA_SPLIT_REQUIRE=%q\n' "$QWEN4_QSA_SPLIT_REQUIRE"
  printf 'VLLM_FL_PACKED_BLOCK_TABLE_ARENA=%q\n' "$VLLM_FL_PACKED_BLOCK_TABLE_ARENA"
  printf 'command='
  printf '%q ' "${server_cmd[@]}"
  printf '\n'
} >"$runtime_dir/server_command.env"

# Keep a complete worker log at the artifact root so the post-warmup cache
# gate can inspect it while Docker also receives the same stream.
exec > >(tee -a "$artifact_root/server.log") 2>&1
echo "launching all-on vLLM server on port $PORT" >&2
exec "${server_cmd[@]}"

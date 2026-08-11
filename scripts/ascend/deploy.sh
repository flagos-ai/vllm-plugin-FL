#!/usr/bin/env bash
# ============================================================================
# deploy.sh — FlagOS vLLM 推理服务启动 (Ascend 910B2C, 多卡 TP)
#
# 前提: install.sh 已执行成功 (vllm / FlagCX / FlagTree / FlagGems 已装好)
#
# 用法:
#   bash deploy.sh
#
# 可调参数 (通过环境变量覆盖):
#   MODEL_PATH               模型路径 (必填, 无默认值)
#   ASCEND_RT_VISIBLE_DEVICES  NPU 卡号 (默认 0,1,2,3)
#   TP_SIZE                  张量并行数 (默认 4)
#   PORT                     服务端口 (默认 8000)
#   SERVED_MODEL_NAME        模型服务名 (默认 qwen36)
# ============================================================================
set -euo pipefail

# --- 1. CANN 环境 ---
if [ -z "$ASCEND_HOME_PATH" ]; then
  _SET_ENV="/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  [ -f "$_SET_ENV" ] && source "$_SET_ENV" || { echo "[ERROR] CANN not found: $_SET_ENV"; exit 1; }
fi

# --- 2. conda env 检查 ---
if [ -z "$CONDA_PREFIX" ]; then
  echo "[ERROR] conda env not activated. Run: conda activate ascend-infer"; exit 1
fi
PY_BIN=$(command -v python3)
[ "$PY_BIN" != "$CONDA_PREFIX/bin/python3" ] && { echo "[ERROR] python3 not in conda env. Run: conda activate ascend-infer"; exit 1; }

# --- 3. FLAGCX_PATH 加载 ---
if [ -z "${FLAGCX_PATH:-}" ]; then
  _ACTIVATE_D="${CONDA_PREFIX}/etc/conda/activate.d/zz_flagos_env.sh"
  [ -f "$_ACTIVATE_D" ] && source "$_ACTIVATE_D"
fi
[ -z "${FLAGCX_PATH:-}" ] && { echo "[ERROR] FLAGCX_PATH not set. Run install.sh first."; exit 1; }
echo "[OK] FLAGCX_PATH=$FLAGCX_PATH"

# --- 4. 推理参数 ---
export TRITON_ALL_BLOCKS_PARALLEL=1
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"

JEMALLOC_SO=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2
[ -f "$JEMALLOC_SO" ] && export LD_PRELOAD=$JEMALLOC_SO:${LD_PRELOAD:-} || true

# --- 5. 启动 vllm serve ---
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required, e.g. MODEL_PATH=/path/to/model bash deploy.sh}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36}"

echo "[INFO] model=$MODEL_PATH  devices=$ASCEND_RT_VISIBLE_DEVICES  TP=$TP_SIZE  port=$PORT"

VLLM_USE_MODELSCOPE=true vllm serve "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-num-seqs 1 \
        --gpu-memory-utilization 0.6 \
        --enforce-eager \
        --trust-remote-code \
        --allowed-local-media-path / \
        --mm-processor-cache-gb 0 \
        --additional-config '{"enable_cpu_binding":true}'

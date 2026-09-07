#!/usr/bin/env bash
set -euo pipefail

# Source this in the plugin container for every preflight, server, correctness
# and benchmark process. There is deliberately no baseline/no-cache branch.
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda
export TRITON_LIBCUDA_PATH=/driver
export LD_LIBRARY_PATH=/driver:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PORT=${PORT:-19821}
export MODEL_CONTAINER=${MODEL_CONTAINER:-/models/Qwen3.8-Flash-Next}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next}
export TP_SIZE=${TP_SIZE:-8}
export INPUT_LEN=${INPUT_LEN:-1024}
export OUTPUT_LEN=${OUTPUT_LEN:-1024}
export CONCURRENCY=${CONCURRENCY:-64}
export WARMUP_PROMPTS=${WARMUP_PROMPTS:-64}
export FORMAL_PROMPTS=${FORMAL_PROMPTS:-128}
export FORMAL_ROUNDS=${FORMAL_ROUNDS:-3}
export SEED=${SEED:-12345}
export COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"mode":"NONE","cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64],"max_cudagraph_capture_size":64,"pass_config":{"fuse_allreduce_rms":false}}'}

# Mounted source precedence is part of the identity gate. Never fall through
# to an image-preinstalled plugin or FlagGems tree.
export PYTHONPATH=/opt/vllm-plugin-FL:/opt/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}
export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1
export VLLM_FL_PREFER=flagos
export VLLM_FL_PREFER_ENABLED=true
export VLLM_FL_OOT_ENABLED=1
export VLLM_FL_STRICT=0
export VLLM_FL_FLAGOS_BLACKLIST=${VLLM_FL_FLAGOS_BLACKLIST:-index_put_,index_put,_index_put_impl_,nonzero,copy_,to_copy,index,index_select,conv1d,_conv_depthwise2d,conv2d,pad,constant_pad_nd,mul}
export QWEN4_HC_BACKEND=${QWEN4_HC_BACKEND:-fallback}

# All-on Qwen/QSA paths present in the mounted candidate source.
export QWEN4_QSA_FUSED_COMPRESS=1
export QWEN4_QSA_MQA_DOT=1
export QWEN4_QSA_SPLIT_TOPK=${QWEN4_QSA_SPLIT_TOPK:-8}
export QWEN4_QSA_SPLIT_REQUIRE=${QWEN4_QSA_SPLIT_REQUIRE:-1}
export VLLM_FL_PACKED_BLOCK_TABLE_ARENA=${VLLM_FL_PACKED_BLOCK_TABLE_ARENA:-1}
export VLLM_FL_PACKED_BLOCK_TABLE_REQUIRE=${VLLM_FL_PACKED_BLOCK_TABLE_REQUIRE:-1}
export VLLM_FL_GDN_STRICT_PATCH=${VLLM_FL_GDN_STRICT_PATCH:-1}
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP8=0

# The plugin bridge is the sole plan-cache activation path. Direct FlagGems
# auto-install is held off; bridge apply() must call modern enable() and verify
# enabled+installed. REQUIRE=1 makes an inactive bridge fail closed.
export VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE=1
export VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REPORT=1
export VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE_REQUIRE=1
export FLAGGEMS_ATEN_PLAN_CACHE=0
export FLAGGEMS_ATEN_PLAN_CACHE_SIZE=${PLAN_CACHE_SIZE:-128}

# These values are passed into the container by start_container.sh and are
# repeated here so direct helper invocation fails closed rather than relying
# on an unset shell variable.
export EXPECTED_FLAGGEMS_INIT_SHA256=${EXPECTED_FLAGGEMS_INIT_SHA256:-9f487a0890ec12beb18b40d55a9fe26387417bbb505a1af29c716f118b26af2d}
export EXPECTED_FLAGGEMS_PLAN_CACHE_SHA256=${EXPECTED_FLAGGEMS_PLAN_CACHE_SHA256:-d9afa71ffb3aada02b82468569d9e02c07b5981a83066fab56f82bd2f3390921}
export EXPECTED_GPU_COUNT=${EXPECTED_GPU_COUNT:-8}
export EXPECTED_GPU_SUBSTRING=${EXPECTED_GPU_SUBSTRING:-H100}

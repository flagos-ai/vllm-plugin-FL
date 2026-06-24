# FlagCX Connector Disaggregated Serving (1P1D)

This guide walks through setting up disaggregated prefill-decode serving with the **FlagCXConnector** on two nodes (one Prefill, one Decode).

## Prerequisites

- Two nodes with RDMA (InfiniBand) connectivity
- NVIDIA GPUs with CUDA toolkit installed
- vLLM (v1 architecture) installed
- Python 3.10+

## 1. Build FlagCX

On **both** Prefill and Decode nodes:

```bash
git clone https://github.com/FlagOpen/FlagCX.git
cd FlagCX

# Build with NVIDIA backend
make USE_NVIDIA=1 -j$(nproc)

# Verify the shared library is built
ls build/lib/libflagcx.so
```

Set environment variables (add to your shell profile or export before running):

```bash
export FLAGCX_PATH=/path/to/FlagCX
# FLAGCX_LIB_PATH is optional; defaults to ${FLAGCX_PATH}/build/lib/libflagcx.so
```

## 2. Install vllm-plugin-FL

On **both** Prefill and Decode nodes:

```bash
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL
pip install --no-build-isolation -e .
```

## 3. Start the Prefill Instance

Run on the **Prefill node** (e.g. `10.8.2.168`):

```bash
# ---- Network / RDMA settings (adjust to your cluster) ----
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=version
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3

# ---- FlagCX settings ----
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_PATH=/path/to/FlagCX

# ---- vLLM settings ----
export VLLM_RPC_TIMEOUT=600000
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_PLUGINS=fl

vllm serve <model_path> \
    --host 0.0.0.0 \
    --port 20001 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --max-model-len 40960 \
    --served-model-name base_model \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"FlagCXConnector","kv_role":"kv_producer"}'
```

## 4. Start the Decode Instance

Run on the **Decode node** (e.g. `10.8.2.169`):

```bash
# ---- Network / RDMA settings (same as Prefill) ----
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=version
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3

# ---- FlagCX settings ----
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_PATH=/path/to/FlagCX

# ---- vLLM settings ----
export VLLM_RPC_TIMEOUT=600000
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_PLUGINS=fl

vllm serve <model_path> \
    --host 0.0.0.0 \
    --port 20002 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --max-model-len 40960 \
    --served-model-name base_model \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"FlagCXConnector","kv_role":"kv_consumer"}'
```

## 5. Start the Router

The router is a FastAPI proxy that coordinates Prefill and Decode instances. Run it on **any** node that can reach both:

```bash
cd vllm-plugin-FL/examples/disaggregated_serving_xpyd

python3 router.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefill http://<prefill_host>:20001 8998 \
  --decode http://<decode_host>:20002
```

- The second argument after the Prefill URL is the **ZMQ side-channel base port** (`FLAGCX_BOOTSTRAP_PORT`, default `8998`).
- You can specify multiple `--prefill` / `--decode` for xPyD topologies.

## 6. Send Requests

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base_model",
    "prompt": "Hello, world!",
    "max_tokens": 64
  }'
```

## 7. Benchmarking

```bash
vllm bench serve \
  --base-url http://<router_host>:8000 \
  --endpoint /v1/completions \
  --model <model_path> \
  --served-model-name base_model \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --request-rate 70 \
  --num-prompts 350 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99
```
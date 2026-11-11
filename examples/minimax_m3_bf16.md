# MiniMax-M3 BF16 on two Ascend A3 hosts

Use vLLM 0.24.0 with the matching FL checkout and an Ascend torch-npu/CANN environment. Select FL with `VLLM_PLUGINS=fl`. Before launching, ensure `command -v npu-smi` and `npu-smi info` succeed in the runtime environment. Model paths and network settings are supplied by the user.

Provide the same BF16 checkpoint on both hosts as `MINIMAX_M3_BF16_MODEL`. Set `NODE_IP` to the local node address, `COMM_IFACE` to its communication interface, and `HEAD_ADDR` and `DP_RPC_PORT` to the shared head address and RPC port. Run this command on the head node:

```bash
export VLLM_PLUGINS=fl USE_FLAGGEMS=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_ENGINE_READY_TIMEOUT_S=3000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_HOST_IP="$NODE_IP" HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
vllm serve "$MINIMAX_M3_BF16_MODEL" \
  --served-model-name minimax-m3-bf16 --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 16 --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser minimax_m3 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --data-parallel-size-local 1 --data-parallel-address "$HEAD_ADDR" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --data-parallel-backend mp --distributed-executor-backend mp
```

On the second node run the same command with its own `NODE_IP` and append `--headless --data-parallel-start-rank 1`. This is DP2 × TP16 / EP32. Send the following to the head `/v1/chat/completions` endpoint; the optional `X-data-parallel-rank` header selects replica 0 or 1.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d ' {"model":"minimax-m3-bf16","messages":[{"role":"user","content":"What is 2 + 2? Answer briefly."}],"temperature":0,"max_tokens":1024,"logprobs":true,"top_logprobs":1} '
```

Check for readable nonempty content, finite logprobs and a natural stop.

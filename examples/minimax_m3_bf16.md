# MiniMax-M3 BF16 on two Ascend A3 hosts

Use vLLM 0.24.0 with FL and a matching Ascend torch-npu/CANN environment. Verify `command -v npu-smi` and `npu-smi info` inside each container.

Install the Ascend paged-cache API from official FlagGems-vllm and the optional ATen provider from official FlagGems. The example allowlists `rsqrt` when `USE_FLAGGEMS=1`. Use FlagTree and CANN SHMEM wheels matching the container's Python ABI and architecture for the FlagGems path.

```bash
pip install 'flaggems_vllm @ git+https://github.com/flagos-ai/FlagGems-vllm.git@2a02eb9f6143e7adc42bc58e272e1a4d48739dc0'
pip install 'flag_gems @ git+https://github.com/flagos-ai/FlagGems.git@eecc900af170774baa309aa625e5d1533a7cfbd1'
```

Both nodes need the same BF16 checkpoint. Set `MINIMAX_M3_BF16_MODEL`, `NODE_IP`, `COMM_IFACE` and `HEAD_ADDR` on each node. Set the same `DP_RPC_PORT` on both if the default 18501 is unavailable. This example serves text with DP2 × TP16 and 32 expert-parallel ranks.

Run on the head node:

```bash
DP_RANK=0 bash examples/minimax_m3_bf16.sh
```

Run on the second node with its own `NODE_IP`:

```bash
DP_RANK=1 bash examples/minimax_m3_bf16.sh
```

Send a chat request to the head node:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3-bf16","messages":[{"role":"user","content":"Suggest three simple things I can bring to a picnic."}],"temperature":0,"max_tokens":256}'
```

Use the optional `X-data-parallel-rank` request header to select replica 0 or 1.

# DeepSeek-V4-Flash FlagOS W8A8 on Ascend A3

Use vLLM 0.24.0 with the matching FL checkout and an Ascend torch-npu/CANN environment. Select FL with `VLLM_PLUGINS=fl`. Before launching, ensure `command -v npu-smi` and `npu-smi info` succeed in the runtime environment. Model paths and network settings are supplied by the user.

Install the required DeepSeek and paged-cache writer library APIs at the published revision before model loading. These explicit calls are required even with `USE_FLAGGEMS=0`; they do not enable optional ATen replacement.

```bash
pip install 'flaggems_vllm @ git+https://github.com/xiemingda-1002/FlagGems-vllm.git@2c5a9d2447d9fd128bfef5fd06668bac67294a9b'
```

Use one host with 16 logical devices and a FlagOS-Compressor compressed-tensors checkpoint with dynamic per-token symmetric INT8 activations and per-channel INT8 weights. Preserve its quantization metadata for automatic detection. Set `DEEPSEEK_FLAGOS_W8A8_MODEL` to the checkpoint.

```bash
export VLLM_PLUGINS=fl USE_FLAGGEMS=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ENGINE_READY_TIMEOUT_S=3000 OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_BUFFSIZE=200 HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
vllm serve "$DEEPSEEK_FLAGOS_W8A8_MODEL" \
  --served-model-name deepseek-flagos-w8a8 --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Send a chat request to the running service:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-flagos-w8a8","messages":[{"role":"user","content":"What is 2 + 2? Answer briefly."}],"temperature":0,"max_tokens":128}'
```

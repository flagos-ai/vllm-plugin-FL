# DeepSeek-V4-Flash FlagOS W8A8 on Ascend A3

Use vLLM 0.24.0 with the matching FL checkout and an Ascend torch-npu/CANN environment. Select FL with `VLLM_PLUGINS=fl`. Before launching, ensure `command -v npu-smi` and `npu-smi info` succeed in the runtime environment. Model paths and network settings are supplied by the user.

DeepSeek V4 compression, sparse attention, indexing and paged cache writes use FL's PyTorch reference backend. Install official FlagGems to run the example with the `sigmoid` ATen override.

The tested A3 runtime uses FlagTree at commit `df9531a7` as its Triton provider and the matching CANN SHMEM Python package (`cann_shmem` 1.6.0). Install wheels matching the runtime's Python ABI and architecture. Verify the compiler and SHMEM bindings in the same environment with `python -c 'import triton, shmem'`.

```bash
pip install 'flag_gems @ git+https://github.com/flagos-ai/FlagGems.git@eecc900af170774baa309aa625e5d1533a7cfbd1'
```

Use one host with 16 logical devices and a FlagOS-Compressor compressed-tensors checkpoint with dynamic per-token symmetric INT8 activations and per-channel INT8 weights. Preserve its quantization metadata for automatic detection. Set `DEEPSEEK_FLAGOS_W8A8_MODEL` to the checkpoint.

```bash
bash examples/deepseek_v4_flagos_w8a8.sh
```

The launcher sets `USE_FLAGGEMS=1` and `VLLM_FL_FLAGOS_WHITELIST=sigmoid`. Rank 0 records FlagGems operator entries in `/tmp/flaggems_enable_oplist.txt`.

Send a chat request to the running service:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-flagos-w8a8","messages":[{"role":"user","content":"Suggest three practical ways to stay focused while working from home. Keep your answer brief."}],"temperature":0,"max_tokens":1024}'
```

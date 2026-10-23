# MiniMax-M3.1-Flash

## Model and runtime requirements

The example uses the public, official checkpoint
[`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash).
Its configuration declares MXFP8 dense layers and NVFP4 MoE experts. Download
these weights directly or use the model ID in the example:

```bash
hf download MiniMaxAI/MiniMax-M3.1-Flash --local-dir /models/MiniMax-M3.1-Flash
```

The [upstream vLLM M3.1 recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml)
specifies a minimum vLLM version of **0.33.0** and requires a nightly build,
with the dedicated NVIDIA image `vllm/vllm-openai:minimax-m31-flash`.
This repository's `main` branch currently targets **vLLM 0.24.0**. The standard
[setup](../README.md#setup) alone does not establish M3.1 compatibility.
Use an M3.1-capable runtime and verify plugin compatibility before enabling
`VLLM_PLUGINS=fl`. This example has not been validated end to end with the FL
plugin; the README entry is therefore marked **Pending validation**.

## Offline text generation

The [example script](../examples/minimax_m31_offline_inference.py) targets
**eight NVIDIA H100/H200 GPUs**, following the upstream Hopper recipe's FP8
KV/indexer caches, Marlin MoE backend, 128-token cache blocks, and disabled
hybrid KV cache manager. Prepare sufficient device memory for the official
checkpoint and runtime. Accelerator count and settings are specific to this
example.

For a small text generation check, the script uses eager execution, skips
the vision encoder, limits context to 8192 tokens and concurrent sequences
to one, and generates up to 512 tokens. It applies the checkpoint's chat
template through `LLM.chat` and uses the model card's recommended sampling
settings (`temperature=1.0`, `top_p=0.95`).

After preparing a compatible vLLM and plugin environment, run from the
repository root:

```bash
VLLM_PLUGINS=fl python examples/minimax_m31_offline_inference.py
```

To use downloaded weights, replace `model` in the script with
`/models/MiniMax-M3.1-Flash`. Other platforms require their own supported
quantization kernels and configurations; consult the upstream recipe rather
than reusing the Hopper settings. The example does not enable multimodal
inputs or DSpark speculative decoding.

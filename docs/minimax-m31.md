# MiniMax-M3.1-Flash

## Model weights

The [example](../examples/minimax_m31_offline_inference.py) uses a FlagOS W8A8
checkpoint of MiniMax-M3.1-Flash. Set `model` in the script to the local
checkpoint directory in your deployment environment:

```python
model="/path/to/MiniMax-M3.1-Flash-W8A8"
```

The official
[`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
checkpoint uses MXFP8 dense layers and NVFP4 experts. Its
[upstream vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml)
has different runtime and kernel requirements. Use the configuration appropriate
for your checkpoint's quantization format.

## Offline text generation

Prepare the chip runtime, compatible vLLM, FlagGems, and vllm-plugin-FL for
your checkpoint. See the repository's [setup instructions](../README.md#setup)
and your deployment environment's model-specific requirements.

The example uses tensor parallel size 8, BF16 activations, cache block size
128, eager execution, 2048 context tokens, and one concurrent sequence. It
uses `LLM.chat` for a short text-generation request with up to 64 output
tokens. Adjust parallelism and memory settings for your hardware.

From the repository root, inside the configured environment:

```bash
VLLM_PLUGINS=fl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python examples/minimax_m31_offline_inference.py
```

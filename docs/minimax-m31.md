# MiniMax-M3.1-Flash

## Model weights

The [example](../examples/minimax_m31_offline_inference.py) uses an existing
FlagOS W8A8 checkpoint named `MiniMax-M3.1-Flash-W8A8-FlagOS-20261019`.
Its configuration declares `MiniMaxM3SparseForConditionalGeneration` and
`compressed-tensors` INT8 quantization. Use a local model directory; no model
download is needed to run the example when the weights are already available.

For example, if the host model directory is
`/public-nvme/models/MiniMax-M3.1-Flash-W8A8-FlagOS-20261019` and
`/public-nvme/models` is mounted at `/models` inside the container, set:

```python
model="/models/MiniMax-M3.1-Flash-W8A8-FlagOS-20261019"
```

Change this path to match your own model mount. This is a local checkpoint
name, not a public Hugging Face repository ID.

The separate official checkpoint
[`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
uses MXFP8 dense layers and NVFP4 experts. Its
[upstream vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml)
has different runtime and kernel requirements. Do not apply its Marlin/NVFP4
settings to the local W8A8 model.

## Offline text generation

Prepare the chip runtime, compatible vLLM, FlagGems, and vllm-plugin-FL in the
deployment environment for your checkpoint. This repository's default
[setup](../README.md#setup) targets vLLM 0.24.0; model-specific runtime
compatibility must match your deployment environment.

The example uses tensor parallel size 8, BF16 activations, cache block size
128, eager execution, 2048 context tokens, and one concurrent sequence. It
uses `LLM.chat` for a short text-generation request with up to 64 output
tokens. Adjust parallelism and memory settings for your hardware.

From the repository root, inside the configured environment:

```bash
VLLM_PLUGINS=fl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python examples/minimax_m31_offline_inference.py
```

## Validation record

On 2026-10-23, one offline generation attempt used the existing W8A8 weights
in the `fl-nvidia032-empty-h100-dev` environment. The process exited during
runtime initialization, before weight loading or text generation: FlagGems
reported `RuntimeError: can't start new thread`, followed by
`ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'`.
The attempt did not establish successful inference. No retry or environment
repair was performed.

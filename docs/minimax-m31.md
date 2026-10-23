# MiniMax-M3.1-Flash

[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
is the released MiniMax-M3.1 model. FlagOS provides platform-specific W8A8
weights and deployment environments. See the official
[FlagOS multi-chip deployment guide](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/docs/FlagOS-Multi-Chip-Support-and-Downloads.md)
for model downloads and the corresponding chip model cards.

## Prepare the environment

Select the model card for your chip from the official guide. Follow that card
to prepare the host drivers, runtime, compatible PyTorch build, model weights,
and container image. The platform model card specifies framework versions,
plugin branches, model-specific patches, accelerator count, and memory
requirements. A configured FlagRelease container is the recommended starting
point; for source installation, also see the repository's
[setup instructions](../README.md#setup).

## Offline text generation

The [example script](../examples/minimax_m31_offline_inference.py) follows the
official guide's **T-Head (Zhenwu) W8A8 Express** configuration with
`tensor_parallel_size=8`. From the repository root, run it inside the prepared
environment:

```bash
VLLM_PLUGINS=fl python examples/minimax_m31_offline_inference.py
```

The script loads
`FlagRelease/MiniMax-M3.1-Flash-W8A8-zhenwu-FlagOS-Express`, generates a response
to a text prompt, and prints the prompt and generated text. To use downloaded
weights, replace the script's `model` value with the local model directory.

For another chip, select its W8A8 repository from the official guide and adjust
`tensor_parallel_size`, `max_model_len`, `max_num_batched_tokens`, and
`max_num_seqs` according to that platform's model card. The original
`MiniMaxAI/MiniMax-M3.1-Flash` checkpoint and the platform-specific FlagOS W8A8
checkpoints have different deployment requirements; use the weights specified
by your platform's instructions.

The example uses `temperature=0.0` and `max_tokens=512` for a short generation
check. The official model card recommends `temperature=1.0` and `top_p=0.95`
for general use. This example covers text generation; consult the platform
model card for online serving, chat templates, image/video inputs, DSpark
speculative decoding, and `reasoning_effort` settings.

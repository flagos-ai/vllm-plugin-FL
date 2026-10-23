# MiniMax-M3.1-Flash

## Model weights

Set `model` in the [example](../examples/minimax_m31_offline_inference.py)
to the local MiniMax-M3.1-Flash checkpoint directory in your deployment
environment:

```python
model="/path/to/MiniMax-M3.1-Flash"
```

See the official
[model card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) and
[upstream vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml)
for model details and deployment requirements. Select runtime versions and
settings compatible with your checkpoint.

## Offline text generation

Prepare the chip runtime, compatible vLLM, FlagGems, and vllm-plugin-FL for
your checkpoint. See the repository's [setup instructions](../README.md#setup)
and your deployment environment's model-specific requirements.

The validated NVIDIA environment uses vLLM `0.24.0+cu129` and FlagGems
`5.4.0+g6fe7709e7717` (commit `6fe7709e7717`). Use this FlagGems revision to
reproduce the validation environment; other revisions have not been verified
for this example.

The example uses tensor parallel size 8, BF16 activations, cache block size
128, 2048 context tokens, and one concurrent sequence. CUDA Graph is enabled
only for decode with `FULL_DECODE_ONLY`; prefill runs without graph capture.
The capture size is 1 to match the single concurrent sequence, and compilation
mode is `NONE`. Adjust the capture sizes if you increase `max_num_seqs`.

The script uses `LLM.chat` for a short text-generation request with up to 64
output tokens. Adjust parallelism and memory settings for your hardware.

From the repository root, inside the configured environment:

```bash
export VLLM_FL_FLAGOS_BLACKLIST_APPEND=pow_scalar,reciprocal,add,gelu,fused_moe,to_copy,copy_,sub,full,w8a8_dynamic_per_token_linear

VLLM_PLUGINS=fl USE_FLAGGEMS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python examples/minimax_m31_offline_inference.py
```

`VLLM_FL_FLAGOS_BLACKLIST_APPEND` disables the listed FlagGems operators while
preserving platform defaults. Other FlagGems operators remain enabled.

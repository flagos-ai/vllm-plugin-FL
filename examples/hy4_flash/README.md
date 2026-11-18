# Hy4 Flash on NVIDIA H100

This backport uses vLLM 0.32.0 and the V2 model runner. The example runs
target-only FP8 serving on eight NVIDIA H100 GPUs with tensor and expert
parallelism. It uses eager prefill and CUDA Graph decode. The
[public checkpoint](https://huggingface.co/tencent/Hy4-flash-FP8) may require
accepting the model terms before download.

## Install the operator dependency

Use a CUDA environment with vLLM 0.32.0 and its compatible DeepGEMM build.
In a fresh checkout, install the IndexPool implementation from
[FlagGems-vllm #1295](https://github.com/flagos-ai/FlagGems-vllm/pull/1295) together
with the masked grouped-top-k change in
[FlagGems-vllm #1291](https://github.com/flagos-ai/FlagGems-vllm/pull/1291).

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
git -C FlagGems-vllm switch --detach 1173c491888c870aeed9d51cbf073dd6f2d6f85d
git -C FlagGems-vllm cherry-pick --no-commit 19eb1dd3e43341d951be9a69341b90bdf3d5dcfb
python -m pip install ./FlagGems-vllm
python -m pip install -e /path/to/vllm-plugin-FL
```

## Start the server

From the plugin checkout, pass a local checkpoint directory or the public
Hugging Face model ID. The example binds to loopback and enables the model's
reasoning and tool-call parsers.

```bash
bash examples/hy4_flash/serve.sh /path/to/Hy4-flash-FP8
```

The example uses a 204800-token context and a 64-request limit. DFly speculative
decoding is outside the scope of this backport.

# Hy4 Flash on NVIDIA H100

This integration targets the Hy4 Flash W8A8 no-MTP vision-language checkpoint
on eight NVIDIA H100 GPUs. It uses vLLM 0.32.0 and the V2 model runner with
tensor and expert parallelism, eager prefill, and CUDA Graph decode. Checkpoint
weights stay INT8, with BF16 compute and BF16 MLA KV storage. The indexer uses
its FP8 representation.

## Install the operator dependency

Use the Hy4-compatible vLLM 0.32.0 distribution and its matching DeepGEMM
build. The plugin uses the existing HYV4 classes from that distribution.
Install FlagGems and the matching FlagGems-vllm source with the companion
`flaggems-vllm-w8a8-coverage.patch`. The patch is applied to the W8A8 operator
source snapshot used to validate this integration. It supplies the sparse
FlashMLA `out` contract, large-offset paged MQA, and the W8A8 MoE and
attention operators. It is distributed separately from this plugin branch.

```bash
python -m pip install /path/to/patched/FlagGems-vllm
python -m pip install -e .
```

## Start the server

From the plugin checkout, pass the W8A8 no-MTP checkpoint directory. The
example registers the full vision-language entry point, binds to loopback,
and enables the model's reasoning and tool-call parsers.

```bash
bash examples/hy4_flash/serve.sh /path/to/Hy4-flash-testing-W8A8-no-MTP
```

The example uses a 204800-token context and a 64-request limit. Specialized
INT8 scaled GEMM and routed-expert GEMM dispatch through FlagGems-vllm.
Generic ATen `mm`, `bmm`, and `addmm` use the native backend.

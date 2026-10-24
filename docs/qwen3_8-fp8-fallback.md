# Qwen3.8 / Qwen4 FP8 selection

The Qwen3.8-Flash-Next / Qwen4 model checks the active platform's
`supports_fp8()` before constructing layers. Native FP8 platforms keep the
original quantization config and vLLM kernels. NVIDIA support requires compute
capability 8.9 or newer; AMD uses vLLM's ROCm capability and FP8 format queries.
Other plugin platforms currently report no native FP8 support.

On platforms without native FP8, use `--dtype bfloat16`. Standard vLLM `fp8`
checkpoints (per-tensor or block-scaled, static or dynamic activations) load
through a model-local BF16 adapter. Online FP8 requests use the existing
unquantized methods. INT8 and other quantization configurations are unchanged;
this adapter does not convert ModelOpt/MXFP8 or compressed-tensors FP8 formats.

| Qwen path | FP8 dependency | Non-FP8 execution |
| --- | --- | --- |
| Attention QKV/output, GDN projections, dense/shared-expert MLP, indexer projection, quantized vision projections | Linear GEMM and activation quantization when configured as FP8 | Existing `UnquantizedLinearMethod`; dequantize checkpoint weights once during loading |
| Routed MoE experts | FP8 expert GEMMs, activation quantization, expert weight scales | Existing `UnquantizedFusedMoEMethod` and plugin backend selection; preserve each expert's separate gate/up/down scales |
| Ordinary attention KV cache | FP8 cache store/read when explicitly requested | Switch the shared cache config to BF16 before cache allocation |
| QSA attention, index scoring and raw/compressed key caches | Already BF16; QSA rejects FP8 main KV cache on the native path | Existing BF16 public FlagGems-vllm operators |
| HC, PLE convolution/state and GDN recurrent state | No mandatory FP8 dependency | Existing operators and state dtypes |

The loader converts FP8 values to BF16 on CPU before copying them to the
accelerator, so an unsupported vendor does not have to allocate or cast FP8
device tensors. It reuses vLLM's `block_dequant` with the checkpoint scales,
including partial blocks and independent scales for merged projections.
No new numerical kernel or FP8-to-INT8 requantization is introduced.

BF16 resident weights take twice the storage of FP8 weights, and a requested
FP8 KV cache also doubles in size when switched to BF16. Loading additionally
uses CPU staging (one expert at a time for MoE). Kernel dispatch and graph
capture happen after conversion; there is no per-token dequantization.

The fallback tests cover capability selection, native-path preservation,
per-tensor/static/block-scaled loading, merged projections, per-expert scales,
and repeated post-load processing. Hardware support and whole-model accuracy
must still be validated on the deployment platform; the existing Day0 results
are not measurements of this fallback.

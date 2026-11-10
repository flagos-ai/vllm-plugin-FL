# Hygon backend code map

The backend is split by execution scope:

```text
patch.py
patches/models/deepseek_v4.py       DeepSeek-V4 class and attention patches
patches/models/deepseek_v4_metadata.py
impl/attention/                     DV4 Indexer/MLA/Prefill leaves
impl/moe/                            Hygon MoE and LightOp/Triton stage paths
impl/gemm/                           Dense and BF16 projection paths
impl/native_ops/                     Hygon custom operator leaves
impl/comm/                           Hygon custom allreduce
impl/other/                          mHC and other Hygon leaves
```

The `DeepseekV4...` patches are installed from the Hygon backend but are
attached to DeepSeek-V4 vLLM classes. They are guarded again by dtype, shape,
stride, cache ABI and environment variables. Other models do not enter those
paths merely because they run on Hygon; they need the same vLLM class and ABI.

Some leaves are reusable on Hygon when their contracts match, notably dynamic
symmetric INT8 quantization, custom allreduce, and the small-M dense INT8
`scaled_mm` tile selection. The shared activation, router, MoE sum and mHC
leaves retain DV4-oriented shape guards, so unsupported models fall back to the
original implementation.

Production switches and the current scope table are documented in:

`/workspace/performance_opt/dv4_model/Hygon代码结构与模型绑定说明_20260918.md`


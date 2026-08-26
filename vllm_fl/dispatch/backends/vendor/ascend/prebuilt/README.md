# FL-built Ascend artifacts

This directory is an output location, not a copy of vLLM-Ascend. Build with
`VLLM_VENDOR=ascend` and set `SOC_VERSION` to generate the artifacts for one
SoC family:

```text
<soc>/lib/_C_ascend*.so
<soc>/opp/vendors/custom_transformer/
```

The reduced `_C_ascend` library is compiled from `csrc/ascend/torch_binding.cpp`
and registers only the operators used by the Qwen3.6 path. The OPP package is
built from the five selected source directories under `csrc/ascend`.

| CANN family | Output directory | Accepted values |
| --- | --- | --- |
| Atlas A2 / `ascend910b` | `ascend910b1` | `ascend910b*`, `910b` |
| Atlas A3 / `ascend910_93` | `ascend910_93` | `ascend910_93*`, `910c` |

Artifacts must be built with the same Python, PyTorch/torch-npu and CANN stack
used at runtime. A3 never falls back to A2 artifacts.

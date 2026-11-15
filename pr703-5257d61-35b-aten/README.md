# Qwen3.6-35B-A3B Runner V2 ATen comparison on Ascend 910C

The same warmed eager engine ran the original long-text Gate input three times at temperature 0 with thinking disabled, TP2 and a 512-token limit. The [original request](https://github.com/flagos-ai/vllm-plugin-FL/blob/5257d614511f2ba7752032564cbe556ffb15f0d5/tools/adaptation-gate-cases/case_utils.py) produced 65 input tokens in every trial. Both ranks used Ascend Runner V2 and the native C API library from the official FlagCX 0.14.1+cann9.0.0.910c wheel.

| Trial | First token ID | Top log probability | Next token log probability | Completion tokens | Gate result |
| --- | --- | --- | --- | --- | --- |
| Initial FlagGems ATen policy | 248046 | -0.259772 | -2.697272 | 1 | Empty content |
| Native ATen dispatch | 33119 | -0.000085 | -11.000085 | 295 | All checks passed |
| Restored FlagGems ATen policy | 248046 | -0.002009 | -8.752008 | 1 | Empty content |

Both FlagGems policy trials selected the same stop token and emitted no text. Relative L2 distance was 0.601571 on both ranks at the output of transformer layer 0, normalized by the norm of the FlagGems policy output. Three prefill rows per layer were sampled at the first, middle and last positions. Each sample remained finite. The [receipt](receipt.json) includes token IDs, log probabilities, checks, worker state and the full layer comparison.

The diagnostic changed FlagGems ATen registrations in the running worker processes and restored their original configuration for the third trial. The FL provider and native FlagCX path stayed active. The model, prompt, process group, and greedy decoding settings were fixed. The experiment reused an exact-commit accelerator compilation cache, and kept the Gate assertions unchanged. An application-level numerical difference is established. The affected operator and behavior of the image cases require separate isolation.

The runtime was [vllm-plugin-FL 5257d61](https://github.com/flagos-ai/vllm-plugin-FL/commit/5257d614511f2ba7752032564cbe556ffb15f0d5), vLLM 0.32.0+empty, FlagGems 5.4.0, FlagTree 0.7.0+ascend3.5, Triton 3.5.1, and the [FlagGems-vllm 622f5bf provider archive](https://github.com/flagos-ai/FlagGems-vllm/commit/622f5bf0e476dcdb861e50ea49cc94b0852c2619).

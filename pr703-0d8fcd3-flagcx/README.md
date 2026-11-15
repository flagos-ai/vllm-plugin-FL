# Ascend 910C Runner V2 with FlagCX at 0d8fcd3

Two physical Ascend 910C 64 GiB NPUs served both Qwen3.6 models in BF16 TP2. The checkout was [0d8fcd3](https://github.com/flagos-ai/vllm-plugin-FL/commit/0d8fcd38419d78cb08030e857035b4d0ce902ded). The environment used vLLM 0.32.0+empty, torch-npu 2.10.0, CANN 9.0.0, FlagGems 5.4.0, the FlagGems-vllm 622f5bf provider archive, and FlagTree 0.7.0+ascend3.5 providing Triton 3.5.1. The run reused accelerator compilation caches from the preceding 5257d61 Gate with the same image and provider archive.

Both worker ranks ran GPUModelRunnerFLV2 and mapped the native C API library extracted from the official FlagCX 0.14.1+cann9.0.0.910c wheel. USE_FLAGGEMS was 1. Runner selection and operator allowlist variables were unset. The test files retained the original prompts and assertions. [Wheel payload checksums](flagcx-sha256.txt) and [model fingerprints](model-fingerprints.txt) identify the run inputs.

| Model and mode | Passed | Worker FULL replay log entries |
| --- | --- | --- |
| Qwen3.6-27B-eager | 26/26 | 0 |
| Qwen3.6-27B-graph | 26/26 | 432 |
| Qwen3.6-35B-A3B-eager | 26/26 | 0 |
| Qwen3.6-35B-A3B-graph | 26/26 | 618 |

The independent Gate passed **104/104 requests** over 20 scenarios. Each group includes its launch command, initial environment, operator and graph log excerpts and individual output checks. The [summary](summary.json) lists failed requests and checks. [Official CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43728829627) records a separate set of test jobs.

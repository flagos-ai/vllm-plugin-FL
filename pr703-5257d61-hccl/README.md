# Ascend 910C Runner V2 gate at 5257d61

This run exercised [5257d614511f2ba7752032564cbe556ffb15f0d5](https://github.com/flagos-ai/vllm-plugin-FL/commit/5257d614511f2ba7752032564cbe556ffb15f0d5) from [PR #703](https://github.com/flagos-ai/vllm-plugin-FL/pull/703) on two Ascend 910C 64 GiB NPUs in TP2 with the default native HCCL path.

The environment used the [digest-pinned Ascend image](https://github.com/flagos-ai/vllm-plugin-FL/blob/5257d614511f2ba7752032564cbe556ffb15f0d5/docker/ascend/README.md) and [FlagGems-vllm provider backport 622f5bf](https://github.com/flagos-ai/FlagGems-vllm/commit/622f5bf0e476dcdb861e50ea49cc94b0852c2619). The provider source archive has SHA-256 9dc791f6c8ef66291bab751805c35a4bee7477f81cca90baf4fb6c2d3f00a7fe. The active stack was vLLM 0.32.0+empty, torch-npu 2.10.0, CANN 9.0.0, FlagGems 5.4.0, FlagTree 0.7.0+ascend3.5, and Triton 3.5.1.

| Model | Mode | Passed | Failed requests | Worker replay log entries |
| --- | --- | --- | --- | --- |
| Qwen3.6-27B | Eager | 26/26 | 0 | N/A |
| Qwen3.6-27B | FULL_DECODE_ONLY | 26/26 | 0 | 432 |
| Qwen3.6-35B-A3B | Eager | 24/26 | 2 | N/A |
| Qwen3.6-35B-A3B | FULL_DECODE_ONLY | 23/26 | 3 | 520 |

The [summary](summary.json) and [four groups](groups/) record 99 of 104 passing requests across the 20 original scenarios. Both modes missed 35B long-form text and the concurrent image yes/no prompt. The 35B graph run also missed OCR, returning Hello VLM where the fixture contains HELLO VLLM. The graph long-form case stopped after one completion token and returned empty content. Both ranks used GPUModelRunnerFLV2, with the runner setting and operator allowlist unset and USE_FLAGGEMS=1. The replay counts are log entries across both ranks after service startup.

Each group includes the launch command, effective initial environment, Runner V2 and graph replay log lines where applicable, and the original request checks with generated responses. Model and checkout paths are relative. The [corresponding official CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43677579212) used its own serving cases and completed successfully.

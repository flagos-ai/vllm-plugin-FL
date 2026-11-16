# MetaX C550 Runner V2 with FlagCX at 824d96d

The exact [plugin checkout](https://github.com/flagos-ai/vllm-plugin-FL/commit/824d96d8bdc5f217c5ac88c96ce6ffc98154b04b) served both Qwen3.6 models in BF16 TP4. The run used vLLM 0.32.0, the v0.32.0-metax-ci image, FlagGems 5.5.0.dev0, FlagGems-vllm 0.3.0.dev1 at [e005571](https://github.com/flagos-ai/FlagGems-vllm/commit/e00557103e284db5b57168386db60faa051989ef), FlagTree supplying Triton 3.6.0, and native FlagCX 0.13.0.

The worker startup log names the upstream GPUModelRunner selected by the MetaX Runner V2 branch. Runtime logs show that the FlagGems-vllm public operators were bound for topk_softmax and topk_sigmoid. All four TP workers in each group mapped the native FlagCX shared library. Runner selection and the operator allowlist variables were unset. The original prompts and assertions were used for all 20 scenarios.

| Model and mode | Requests passed | PIECEWISE replay counter sum |
| --- | ---: | ---: |
| [Qwen3.6-27B eager](groups/Qwen3.6-27B-eager/) | 26/26 | 0 |
| [Qwen3.6-27B graph](groups/Qwen3.6-27B-graph/) | 26/26 | 341 |
| [Qwen3.6-35B-A3B eager](groups/Qwen3.6-35B-A3B-eager/) | 26/26 | 0 |
| [Qwen3.6-35B-A3B graph](groups/Qwen3.6-35B-A3B-graph/) | 26/26 | 311 |

The [run summary](summary.json) records **104/104** requests, the image digest, the native library hash and 32 passing focused tests. Each group directory includes its launch arguments, runtime selections, replay rows and scenario assertions. [Official CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43794411290) records the separate repository workflow for the same commit.

# Historical PR description before review cleanup

Tested revisions and limitations below are preserved verbatim from the public PR body at head `80a83d78`. These are historical records, not validation of the new merge commit.

Enable Sunrise PTPU on vLLM 0.32 with native Model Runner V2. The branch is synchronized with `0.5.0-dev` at `045afdc4` (head `80a83d78`). The shared FlagCX loader retains the merged AMD source path and uses the packaged FlagCX fallback on PTPU.

## A. Images and software stack

| Item | Tag, digest, or version |
| --- | --- |
| Base image | `harbor.baai.ac.cn/flagos-app/vllm0.24.0-sunrise-tangrt1.2.0:2.3.0-0.3.0@sha256:662fbf70cc5002b6c084a3601f25c99fc6dd1450543444e236b694425e9a01f5` |
| vLLM 0.32 image | `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-sunrise-release-8905f42@sha256:e0c5ee48b79317fda2dd8d97d1d89f2ddb487db6375a7d13cd58a57f635547e5`. Includes the plugin and its pinned operator runtime. [Deployment guide](https://github.com/flagos-ai/vllm-plugin-FL/blob/77c980594a32453788a89565d151e58564055083/docs/usage/sunrise.md) |
| Runtime | vLLM `0.32.0+empty` (`5df63a7`), Python 3.10.20, PyTorch `2.11.0+cpu`, torch-ptpu `0.2.3+torch2.11`, host driver 0.25.0, TangRT/PCCL 1.2.0, FlagCX `0.14.0+tangrt1.2.0`. |

## B. Changes and reasons

| Change | Reason |
| --- | --- |
| Build the empty wheel with the vendor Python 3.10 ABI. | Preserve the Sunrise runtime and compiler stack. |
| Add native V2 attention, KV cache, masked MoE and PTPU graphs. | Support eager execution and `FULL_DECODE_ONLY` replay. |
| Call the public FlagGems-vllm strided KV writer for the combined 0.32 cache. | Keep numerical writes and replay tests in the operator library. The plugin selects the cache layout and passes slot metadata. [Provider PR #1232](https://github.com/flagos-ai/FlagGems-vllm/pull/1232) |
| Wait for FULL graph completion before eager FlagCX collectives. | Address the PCCL watchdog observed when eager AllGather followed replay. [FlagCX #715](https://github.com/flagos-ai/FlagCX/issues/715) |
| Use native PTPU GEMM for unaligned BF16 vision MLP shards. | Avoid NaNs in the Qwen3.6 TP2 vision projection. |
| Route Sunrise `linear` through torch-ptpu. | Use the vendor implementation for dense projections. The hardware runs kept FlagCX and the remaining FlagGems dispatch enabled. [Source](https://github.com/physics31415926/vllm-plugin-FL/commit/0f385a3c188ffb77edfe0a2f63ceb5def21f88b4) |

## C. Tests and results

At `8905f42`, both GPUs passed exact tensor copies and FP32/BF16 FlagCX all-reduce and all-gather. The pinned provider suite passed 5/5, and the Gate, worker and FlagCX suites passed 109/109. On the alternate PTPU pair, Qwen3.6-27B TP2 BF16 eager passed 26/26 and a 4,118-token prefill under native Runner V2 with async scheduling. The graph run passed both text scenarios (9/9), then failed on the first image prompt (1,062 tokens). The worker log recorded a PCCL BF16 AllReduce watchdog and the engine timed out in sample_tokens. One card retained device memory after the owned service stopped. The device cleanup guard stopped the run before both 35B-A3B groups. A diagnostic retry on another directly connected pair kept the same eager/graph behavior and added per-rank timing logs at multimodal embedding preparation. It passed all 26/26 original 27B graph requests and the 4,118-token prefill and shut down cleanly. Both ranks completed multimodal embedding preparation and model forward on the image that failed earlier. The same diagnostic also passed all 26/26 35B-A3B eager and 26/26 35B-A3B graph requests, each with a 4,118-token prefill and clean shutdown. Both graph groups recorded request-time FULL replay. An uninstrumented repeat on the same pair with unchanged 8905f42 passed the 9 text requests and the first image request in 27B eager, then hit the same PCCL BF16 AllReduce watchdog during eight concurrent image requests. The sampled native worker stacks showed one rank blocked in a PTPU tensor copy waiting on stream synchronization and the other in PTPU device synchronization. The shutdown guard stopped the rest of the matrix because one card retained device memory after its process exited. The failing execution modes include eager and graph. On one additional pair, an unchanged 27B eager run with synchronous scheduling passed all 26 requests and the 4,118-token prefill. Its async control failed the concurrent image case with the same BF16 AllReduce watchdog. The captured worker stacks showed both ranks inside vision encoder GEMM paths, one waiting on an addmm-related tensor copy and the other on a Triton autotuning synchronization. Standalone PCCL and FlagCX BF16 probes passed 48 checks each.

The published image passed two Qwen3-4B TP1 eager smoke prompts in a fresh network-disabled container using its baked packages, with Runner V2 verified. The usage and CI image changes preserve the tested Sunrise runtime. [Full CI at 870e037](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43863228741) passed, including the [aggregate gate](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43863228741/job/133525540495). The independent Sunrise TP2 gate remains open with the watchdog described above. [Prior CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43829342441/job/133171831288) hit a Git transport failure while installing the Iluvatar provider. The [CodeQL import finding](https://github.com/flagos-ai/vllm-plugin-FL/pull/697#discussion_r4583081428) is resolved.

The first post-merge model run at `f466a90` timed out at 27B startup on the original pair after weight load and GDN warmup. One physical card retained 28.5 GiB and 100% utilization after service cleanup, leading to the alternate-pair retest.

After merging the Ascend base, 57 FlagCX contract tests and 17 Sunrise/Ascend runner-selection tests passed in the Sunrise image. The merged installer also passed VCS and checksummed archive provenance checks. On the same merged runtime, a physical TP2 preflight passed tensor copies, BF16/FP32 FlagCX all-reduce and all-gather. The first CI at `86e62fd` found two AST-extracted router fixtures missing the other vendor's device predicate. The test-only fix passed both suites locally (86/86). [Full CI at 74d76ee](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/44004538263) passed on that historical revision.

On `870e037`, an additional 27B async eager diagnostic with warmed Triton and FlagGems autotuning caches passed 26/26 and a 4,118-token prefill, including eight concurrent image requests. That result is scoped to the warm-cache diagnostic. The cold-cache watchdog remains an open acceptance issue.
### Current integration and prior acceptance evidence

| Test | Device, topology, and scope | Result and evidence |
| --- | --- | --- |
| Current integration `80a83d78` | Latest `0.5.0-dev` base `045afdc4` (#770) | Conflict resolved and pushed. All Iluvatar image, usage, setup and Docker files match the target branch. The 26 existing adaptation-gate orchestration tests passed. [Full CI](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/44520823232) **passed**, with 42 successful jobs, 10 configured skips and the [aggregate ci-gate](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/44520823232/job/135865798005) successful. [Code Scan](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/44520822873) passed. Prior Sunrise hardware results below retain their tested revisions. |
| Static and focused tests | Local lint and the pinned Sunrise runtime | Ruff lint and format checks passed. The gate, Sunrise worker, and worker-selection suites passed 81/81 in the pinned Sunrise image. |
| Sunrise preflight | PTPU cards 3 and 7, FlagCX `0.14.0+tangrt1.2.0`, FlagGems `5.4.0`, FlagTree `0.7.0+sunrise3.6`, provider `3b0f388` | 9/9 exact tensor copy checks per card. FP32/BF16 AllReduce and AllGather matched on both ranks. The provider passed 5/5 exact stride-aware KV write and graph replay tests. |
| Sunrise async gate | One PTPU host, cards 3 and 7, TP2, native Runner V2 with `--async-scheduling`. Qwen3.6-27B and Qwen3.6-35B-A3B in BF16 eager and `FULL_DECODE_ONLY` graph, 8,192-token context, 8 sequences, 4 GiB KV cache. | 20/20 scenarios, 104/104 shared requests and 4/4 long-prefill requests. Each group measured 4,118 prompt tokens. The 27B and 35B graph groups recorded 221 and 445 request-time FULL replays. |
| Sunrise sync gate | Same models, host, runtime and configuration with `--no-async-scheduling`. | The shared validator accepted 20/20 scenarios, 104/104 shared requests and 4/4 long-prefill requests. Each group measured 4,118 prompt tokens. The 27B and 35B graph groups recorded 214 and 441 request-time FULL replays. |
| Sunrise run isolation and cleanup | Both schedulers, eight original groups with separate fresh Triton compiler caches. | The first synchronous 27B eager group passed 0/26 quality checks. Its fresh-container rerun on `097cab2` passed 26/26 and the long prefill. The sync validator used that receipt alongside the other three original sync groups. Owned-process cleanup succeeded, and cards 3 and 7 had no remaining device handles or ECC errors. |
| GitHub Actions | CI and Code Scan for `097cab2` | Attempt 2 passed the aggregate `ci-gate`. The targeted T-Head serving rerun passed all four Qwen3.6 TP4 cases, covering 27B and 35B-A3B in eager and graph mode. The first serving attempt ended when its runner shut down. NVIDIA, Iluvatar, T-Head inference, lint, build, discovery, CodeQL and dependency review also passed. [CI attempt 2](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43792265468/attempts/2) [T-Head serving](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43792265468/job/133090174620) [Aggregate gate](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43792265468/job/133129986137) [Code Scan](https://github.com/flagos-ai/vllm-plugin-FL/actions/runs/43792265319) |

<details>
<summary>Earlier reproductions and logs</summary>

[Sunrise graph and vision reproducers](https://github.com/physics31415926/vllm-plugin-FL/tree/c90ff9149b8adb3643ae8832fb2cdc7448becda6/tools/repro/sunrise_032) · [27B image diagnostic](https://github.com/physics31415926/vllm-plugin-FL/tree/4f48155efaca48a4497d4a698bdeb67a4b22494d/pr697-7b43f8a-async-image-diagnostic) · [27B token diagnostic](https://github.com/physics31415926/vllm-plugin-FL/tree/04f71ab6ffe05ccb4f85208a1914a74698564e89/pr697-7b43f8a-graph-token-diagnostic)

</details>

## D. Additional FlagGems and FlagTree dependencies

| Component | Additional requirement | Issue, PR, or exact extra change |
| --- | --- | --- |
| FlagGems | Image-pinned 5.4.0 | The plugin uses torch-ptpu for dense `linear` on Sunrise. Other FlagGems dispatch stays enabled. |
| FlagGems-vllm | The release image includes the pinned v0.2.0 backport `3b0f388` and its build-time dependencies. It supplies `reshape_and_cache_flash_strided` for the Runner V2 combined KV cache. Deployment uses the baked distribution. | [Fixed source revision](https://github.com/flagos-ai/FlagGems-vllm/commit/3b0f3880b2a781699462eacb7dd3e00cc24f2d43) · [Usage and image](https://github.com/flagos-ai/vllm-plugin-FL/blob/77c980594a32453788a89565d151e58564055083/docs/usage/sunrise.md) · [Upstream #1232](https://github.com/flagos-ai/FlagGems-vllm/pull/1232) merged on 2026-11-18 as `dac60034`. The qualified image retains its immutable v0.2.0 backport. |
| FlagTree | Image-pinned `0.7.0+sunrise3.6`, Triton 3.6.0 | None |

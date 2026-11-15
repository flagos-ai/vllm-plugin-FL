# AMD W7900D two-host FlagCX gate

Two W7900D hosts used one GPU each with the same release image from the [single-host qualification](../pr659-be49655-release). The serving topology was TP2, PP1, DP1 with native Runner V2 in eager mode. The hosts had amdgpu 6.16.13 and 6.14.14.

The independent FlagCX GPU collectives passed on both hosts. Each rank reduced 257 FP32 elements to exactly 3 and gathered the two rank-distinct tensors as 1 and 2. Both direct collective rank processes exited 0. Gloo provided rendezvous and RCCL carried FlagCX payloads through Socket.

The original Qwen3.6-27B text_single case passed 1/1. The API returned HTTP 200, 195 completion tokens and finish reason stop. Each serving worker emitted three bounded native FlagCX all-reduce and three all-gather records for TP group [0, 1] during the request. The RCCL channels show transfers between ranks 0 and 1.

The first trial selected RoCE and both collective ranks exited 1 during RCCL initialization. Both logs show that ibv_reg_mr_iova2 failed with Cannot allocate memory before payload exchange. The containers had their default 8192 KiB memlock limit. The passing trial selected RCCL Socket per container. Both hosts kept their original RoCE and memlock settings.

Both server containers exited 0 after SIGTERM. The head-node executor reported graceful worker exit. During headless shutdown the peer logged a TCPStore broken pipe and the executor sent SIGKILL to its remaining worker. The exact serving worker exit codes were not captured. All task containers and GPU device holders cleared.

[Receipt](summary.json) includes the exact image, rank results, request-time call samples, request oracle checks, and shutdown observations. Hostnames, addresses, process identifiers and raw completions were omitted.

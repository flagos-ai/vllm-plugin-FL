# AMD W7900D Runner V2 gate at `5f3f669`

The single-host TP8 gate passed all 20 scenarios and all 104 requests on eight W7900D GPUs. The release environment used vLLM 0.32.0+empty, FlagTree Triton 3.6, FlagGems-vllm, and native FlagCX. vLLM selected upstream Runner V2 with the runner override unset. [Plugin commit](https://github.com/flagos-ai/vllm-plugin-FL/commit/5f3f669516e71a177416f72e1690c8c8bf79bb9a) · [FlagGems commit](https://github.com/flagos-ai/FlagGems/commit/292cec32414fbfea3745df5e5d640dd0fad210d7)

| Model | Mode | Scenarios | Requests | FULL replays |
| --- | --- | --- | --- | --- |
| Qwen3.6-27B | Eager | 5/5 | 26/26 | 0 |
| Qwen3.6-27B | Graph | 5/5 | 26/26 | 218 |
| Qwen3.6-35B-A3B | Eager | 5/5 | 26/26 | 0 |
| Qwen3.6-35B-A3B | Graph | 5/5 | 26/26 | 230 |

The 35B-A3B graph group used 20 GiB of KV cache per worker and capture sizes 1, 2, 4, 8, 16, and 32 in `FULL_DECODE_ONLY`. The 27B graph group used the default compilation settings. Both used BF16 and a 32,768-token model limit. All four groups recorded the V2 worker marker and mapped `libflagcx.so` and `librccl.so` in their TP workers. Before serving, 31 focused tests passed, including the AMD clone tests and the sparse MLA and post-write error regressions. [Gate scripts](https://github.com/flagos-ai/vllm-plugin-FL/tree/5f3f669516e71a177416f72e1690c8c8bf79bb9a/tools/adaptation-gate-cases)

The tested image was `sha256:6591ace360a56348d9e82e1649019b367d429ebf49be8f4e20a819a92fac2152`. Two clean builds stopped at GitHub transport errors while cloning vLLM. The offline build used the previously verified `sha256:0fd7d28f6c5c4a2921c1056bfb2b69e43784c680470edd5524ec15bf39c11c6c` image, installed FlagGems from the official Git commit `292cec32414fbfea3745df5e5d640dd0fad210d7`, and installed the plugin from the exact archived commit. It reused the other dependencies. The runtime checks confirmed the new FlagGems version and the installed clone and HIP-backend file hashes.

[summary.json](summary.json) contains the model fingerprints and results. [runtime-excerpts.json](runtime-excerpts.json) contains the runtime versions, selected operations, and replay counters. The [scenarios](scenarios) directory records checks, latency, and completion reason for each request, with the request and response text omitted.

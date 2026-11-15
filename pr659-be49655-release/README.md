# W7900D release-image Runner V2 gate

The release recipe in [be49655](https://github.com/flagos-ai/vllm-plugin-FL/commit/be496551412e20353bface934407daa37d7c5739) passed all 104 requests across 20 scenarios on eight W7900D GPUs. The Qwen3.6-27B and Qwen3.6-35B-A3B graph runs recorded 222 and 231 FULL executions during requests. The 35B-A3B group used the 20 GiB per-worker KV budget and FULL_DECODE_ONLY capture sizes 1–32, as recorded in the launch script.

| Runtime component | Observed in the release image |
| --- | --- |
| vLLM | 0.32.0+empty, source 5df63a7 |
| PyTorch and HIP | 2.13.0+git733fca1 and 7.2.53211 |
| Host driver and RCCL | amdgpu 6.16.13 and RCCL 2.27.7.70203-90~22.04 |
| FlagTree | 0.7.0+amd3.6 supplied Triton 3.6.0 from /opt/flagtree/triton/__init__.py and its native extension from /opt/flagtree/triton/_C/libtriton.so |
| FlagGems | 5.5.0.dev1+g9a51bed13, enabled and verified with compiled clone |
| FlagGems-vllm | v0.2.0 source b5c2c224, package metadata 0.1.0. The serving logs record dispatch of topk_softmax and fused MoE helper ops to this provider. |
| FlagCX | 0.15.0.dev52+g591cfbf. The platform selected CudaFlagcxCommunicatorFL and the 27B graph worker mapped the image's FlagCX and RCCL libraries. |
| Runner and operator settings | Native Runner V2 selected with VLLM_USE_V2_MODEL_RUNNER unset. VLLM_FL_FLAGOS_WHITELIST was unset. |

[Summary](summary.json) gives the image ID, Dockerfile body fingerprint, model fingerprints, and per-group totals. [Scenario receipts](scenarios) contain the original oracle booleans and timings. [Runtime log excerpts](runtime-excerpts.json) contain serving dispatch, the four numeric columns of FULL replay rows (unpadded, padded, padding, count), and dual-GPU transport and clone smoke results. Host paths, raw requests, and completions were omitted.

[Launch script](run.sh) runs against the installed image with the unchanged adaptation case directory at /gate, read-only model directories at /models, and a writable /validation mount. BuildKit used the bundled Dockerfile frontend because Docker Hub's external frontend was unreachable on the build host. The ROCm base and FlagTree source image came from their recorded digests, sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905 and sha256:074c62e585e050731e3e17facfc0463709053002b6b3c9144f459dac9cc475b2.

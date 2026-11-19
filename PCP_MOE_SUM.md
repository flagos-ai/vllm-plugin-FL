# PCP MoE sum and shared-expert addition

This branch extends `3f88d32` (FA3 with complete-prefill KV reuse). The
restricted synchronous Qwen PCP MoE path combines the native top-8 reduction
with shared-expert gate multiplication and addition in one Triton kernel.
The combined TP partial goes directly to the existing reduce-scatter.

The kernel accepts contiguous CUDA BF16 tensors, hidden size 2048, top-k 8,
and matching row counts. Other layouts use the original sum and gate/add.
The existing eligibility checks retain native custom-op routing and require
NoDPEP Triton experts, no transforms/scaling/padding, no DBO, and synchronous
shared experts. Decode and ragged PCP batches retain their existing paths.

The FP32 reduction order matches the native CUDA top-8 reduction; both the
reduced expert result and the shared gate product round to BF16 before the
final addition. FP fusion is disabled. The shared output is not mutated.
The scoped `moe_sum` override restores pre-existing instance state even on
exceptions. The caller consumes the expert workspace with reduce-scatter on
the same stream before subsequent expert workspace reuse.

Validation on four H100s, Qwen3.6 BF16 eager FA3, vLLM 0.20.2:

- 88 CP tests pass, including wide-exponent reduction versus native
  `ops.moe_sum`, unsupported-layout fallback, and exception restoration.
- Seven generation scenarios match token IDs and all saved top-5 logprobs:
  64K/128-output, short dense/ragged, 4096, mixed lengths, mixed decode, and
  chunked prefill. Validation uses an in-process scheduler to enqueue the
  entire batch before execution; TP still uses four spawned workers.
- 640 comparisons on identical real MoE inputs across four ranks and all
  40 layers are bitwise exact. Audited row counts are 512/4096/4104/65536;
  800 successful fused kernel calls confirm the optimized path executes.
  A separate run without audit also matches every generation result.

Validation artifacts and reproducible scripts:
`/vllm-workspace/validation/qwen36_pcp_moe_comm_20261001_retry/`.
No additional 128K or PD performance claim is made for this change.

Matched unprofiled 64K/C1 timing (two warmups, seven samples each):
median TTFT 713.030 → 704.165 ms, 8.864 ms
(1.24%) lower. All timed 128-token outputs match. Communication
microbenchmarks found no large-message benefit from forcing Simple or LL128;
default protocol selection is retained. See RESULTS.md in the validation folder.

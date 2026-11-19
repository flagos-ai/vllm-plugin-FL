# PCP prefill kernel fusion

Branch `prstyle-pcp-kernel-fusion` extends the preserved PCP implementation at
`8eed108`. The native TP baseline remains unchanged. These changes improve the
PCP implementation, including its TP-style shared/routed expert computations.

`vllm_fl/cp/fused_norm.py` combines BF16 residual addition, FP32 conversion and
squaring, retains the native `torch.mean` reduction, then fuses epsilon addition,
rsqrt, normalization, Gemma's FP32 `(weight + 1)` multiplication and BF16 output
conversion. Q/K projection views can be read directly with their original
strides. The original mean reduction is necessary: the initial one-kernel
Triton reduction changed a small number of BF16 outputs and failed two generation
probes. That experimental version is not the final implementation.

`vllm_fl/cp/fused_moe.py` combines the shared-expert gate multiplication and
shared/routed addition, preserving intermediate BF16 rounding. On the audited
synchronous, unquantized Triton-expert path, it also returns the already-finalized
expert output instead of copying it into another buffer. Native routing,
expert GEMMs, top-k weighting/sum and custom-op boundaries remain in use. The
specialization requires the exact supported runner, router, shared MLP and
modular backend classes, no DP/EP/DBO, no shared-expert overlap, no padding or
latent transforms, and a dense rank-major input. Other configurations use the
existing PCP path. Temporary changes are confined to the current module
instances and restored in `finally`.

Both helpers are called explicitly from PCP prefill. Native TP and PCP decode
keep native dispatch. Installed vLLM is not modified. The patch depends on the
private vLLM runner/kernel interfaces of the tested environment; changes to
those interfaces require review and validation.

Validation: 72 unit tests passed. With Qwen3.6 at 64K input, 128 output,
concurrency 1, BF16, eager, FA2, Triton MoE and four H100s, unprofiled median
TTFT fell from 970.154 ms to 903.776 ms over five measurements after two warmups.
All 128 output tokens match the preserved PCP version. Dense, ragged, batched,
mixed decode/prefill and chunked probes match tokens and logprobs exactly.
Post-measurement audits compare all 101 norms and all 40 MoE blocks on all four
ranks: 404 norm outputs and 160 MoE partial outputs are bitwise equal to native
operations on the same real inputs, including exact residual outputs. These
checks cover the tested model, shapes and hardware, not every possible input.

Results, raw measurements, source snapshots, traces and reproduction scripts:
`/vllm-workspace/validation/qwen36_pcp_kernels_64k_20260929/RESULTS.md`.

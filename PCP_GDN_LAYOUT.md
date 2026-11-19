# PCP GDN output layout fusion

This branch removes three intermediate data movements from the supported PCP
prefill GDN path: flattening the strided z view, copying the output projection
into a caller-owned full output, and packing uniform rows after projection for
ReduceScatter. The gated norm reads z using token/head strides and can emit
rank-major rows before the token-wise output projection.

Input projection, convolution, the GDN custom op, recurrent-state updates, and
the custom op's zero-initialized output contract remain native. They execute in
original sequence order. Norm uses native FP32 reduction/gating arithmetic and
the same row tile and warp count. Output projection still uses the original
RowParallelLinear with reduction temporarily disabled; the wrapper restores the
flag even when either implementation raises.

The fast path accepts native CUDA BF16 GatedDeltaNetAttention with the
non-interleaved Qwen3.5/3.6 projection layout, head dimension 128, unquantized
linear methods, and the native RMSNormGated CUDA method (no bias or groups,
norm before SiLU/swish/sigmoid gating). Other configurations use the original
path. Decode and mixed-batch dispatch are unchanged.

Uniform PCP rows use the fused rank-major output directly for ReduceScatter.
Ragged rows still use the existing post-projection packing/padding, while
benefiting from strided z reads and removal of the intermediate output copy.

The implementation is in `vllm_fl/cp/fused_gdn.py`, called explicitly from
`prstyle.py:_pr_layer_forward`. No global GDN/norm monkey patch or installed
vLLM modification is used. The branch includes the pre-existing MoE workspace
synchronization fix in its baseline commit `8c24efd`.

Correctness validation on H100/BF16:

- 34 new tests pass, including native norm arithmetic, input ownership,
  64K/multi-request output projections, unsupported inputs, and exception cleanup.
- Eight real-model scenarios match baseline token IDs and saved top-5 logprobs:
  64K + 128 outputs, 8K x 8, 32 tokens, 31 tokens, mixed lengths,
  short/decode fallback, chunked prefill, and a 65535-token ragged prefill.
- 960 same-input GDN comparisons match partial outputs, convolution state and
  recurrent state exactly. These include 120 comparisons with existing history.
- 740 full-logit comparisons across four ranks match exactly; 264 output tokens
  per variant. The fallback case makes no calls to the new GDN implementation.

The existing CP suite also has three failing MoE `already_reduced=True` layout
cases. The same three failures reproduce on the unchanged original worktree;
they are outside this GDN change. Raw logs are retained.

Performance was tested on 2026-10-13 after the user authorized the follow-up.
On the same four H100 GPUs, unprofiled prefill plus first-token batch latency
improved in both forward and reversed run order (20 samples per mode/shape):

| Input length x requests | Previous PCP ms | GDN fusion ms | Reduction |
| --- | ---: | ---: | ---: |
| 65536 x 1 | 702.675 | 687.272 | 2.19% |
| 8192 x 8 | 521.937 | 506.999 | 2.86% |

All warmed/timed first-token outputs match, and post-timing checks confirm all
30 GDN layers use the new path on all four ranks. These are fixed prefill
batches, each request producing one token; they do not measure HTTP TTFT or
continuous mixed prefill/decode serving. The measured source is commit
`b9fd49c`; no algorithm changes were made during the performance test.

Performance report and raw samples:
`/vllm-workspace/validation/qwen36_pcp_gdn_perf_20261013/REPORT.md`.

Results and reproduction scripts:
`/vllm-workspace/validation/qwen36_pcp_gdn_layout_20261013/REPORT.md`.

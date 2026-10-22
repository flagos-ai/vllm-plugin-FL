# Interfaces retained from FlagGems

This inventory targets FlagGems-vllm **v0.2.0**, commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`. It was inherited from the vLLM
0.28 adaptation; NVIDIA now uses the full official vLLM 0.32 CUDA distribution.
An existing export is substituted only when it preserves the current call
contract. The entries below retain their original providers. Missing interface requests are tracked in
[FlagGems-vllm #932](https://github.com/flagos-ai/FlagGems-vllm/issues/932).

| Interface | Call site | Reason |
| --- | --- | --- |
| `gems_rms_forward` | `vllm_fl/dispatch/backends/flaggems/impl/normalization.py` | The standalone/residual RMSNorm wrapper is not exported. Combining other kernels would require output and mutation-contract adaptation. |
| `pack_rhs_qsi4c128p` | `vllm_fl/quantization/arm_cpu_w4a8.py` | No corresponding ARM W4A8 packing export exists. |
| `w4a8_g128_linear` | `vllm_fl/quantization/arm_cpu_w4a8.py` | No corresponding ARM W4A8 execution export exists. |
| `chunk_gated_delta_rule_fwd` | `vllm_fl/ops/fla/chunk.py` | The export exists, but a compatible state-layout and vendor contract has not been established for the current vLLM integration. Preserve the branch's existing path instead of adding state conversions. |
| `fused_recurrent_gated_delta_rule_fwd` | `vllm_fl/ops/fla/fused_recurrent.py` | The export exists, but state addressing, padding, and layout require separate compatibility validation. |
| Ascend six-argument `apply_rotary_pos_emb` without the C extension | `vllm_fl/dispatch/backends/flaggems/impl/rotary.py` | The generic v0.2.0 in-place export returns new Tensor views, violating this branch's Q/K object-identity contract for packed partial-RoPE views (Ascend CI job 118173438493). Retain the existing Ascend compatibility path unchanged; no new copy or conversion adapter is introduced. |
| Ascend vendor-specific FLA/MoE implementations | `vllm_fl/dispatch/backends/vendor/ascend/` | Preserve the independently validated vLLM 0.28 Ascend implementation. Compatible generic imports use FlagGems-vllm directly; these vendor-specific implementations remain unchanged. |

ATen registration (`flag_gems.enable()`), device detection, runtime utilities,
and backend launch/configuration helpers retain their existing providers.

NVIDIA normally uses upstream attention backend selection. When
`VLLM_FL_USE_FLAGGEMS_ATTN=1` selects the existing FlagGems attention provider,
its `do_kv_cache_update` in
`vllm_fl/dispatch/backends/flaggems/impl/attention.py` uses upstream's
stride-aware Triton writer. The public FlagGems-vllm cache writer assumes
contiguous token/head layout and cannot directly replace that call for packed
K/V views. This remains a conditional provider layout gap with the full CUDA
distribution installed; it is not a missing native vLLM cache kernel.
The NVIDIA policy keeps its ATen correctness exclusions and pinned development
FlagGems wheel. Other vendor policies remain unchanged.

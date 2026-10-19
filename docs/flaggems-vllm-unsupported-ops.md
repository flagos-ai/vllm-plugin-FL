# Operators excluded from direct FlagGems-vllm replacement

This inventory targets FlagGems-vllm **v0.2.0**, commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`. Its package metadata still says
`0.1.0`; the source commit identifies the release used here.

Only an existing export compatible with the current call site is substituted
when `VLLM_FL_USE_FLAGGEMS_VLLM=1`. The entries below retain their original
FlagGems calls in both switch modes. This change adds no composite operators,
state-layout conversion, release-kernel workarounds, or missing-export errors.

| Operator / interface | Existing call site | Why it is excluded |
| --- | --- | --- |
| `gems_rms_forward` | `vllm_fl/dispatch/backends/flaggems/impl/normalization.py` | The release does not export this combined standalone/residual RMSNorm wrapper. It exports `add_rms_norm` and `fused_add_rms_norm`; using those would require composition and buffer-contract adaptation. |
| `chunk_gated_delta_rule_fwd` | `vllm_fl/ops/fla/chunk.py` | The export exists, but its generic K-major state layout differs from vLLM 0.24's V-major layout; release vendor implementations also differ. Direct import substitution does not preserve the existing state contract. |
| `fused_recurrent_gated_delta_rule_fwd` | `vllm_fl/ops/fla/fused_recurrent.py` | The export exists, but the release selects kernels with different state layouts. Sequence-state addressing and padding behavior also require compatibility handling. Those adaptations are outside this import-only change. |
| `pack_rhs_qsi4c128p` | `vllm_fl/quantization/arm_cpu_w4a8.py` | No corresponding release export exists. |
| `w4a8_g128_linear` | `vllm_fl/quantization/arm_cpu_w4a8.py` | No corresponding release export exists. |
| Ascend `LayerNormFn` | `vllm_fl/dispatch/backends/vendor/ascend/patch.py` | The Ascend FLA normalization/gating wrapper has no compatible release export. Composing other normalization operators would be additional adaptation. |
| Ascend-specific FLA and MoE paths | `vllm_fl/dispatch/backends/vendor/ascend/` | Ascend is not adapted to the current version. Its Python paths and disabled CI configuration remain unchanged. This is a platform exclusion, not a claim that every corresponding generic release export is missing. |

## Scope of replacement

W8A8 `scaled_mm` is replaced by the existing `triton_scaled_mm` export using an
import alias. The call arguments and surrounding computation are unchanged.
Sunrise attention, MLA, and cache operators likewise change only their import
source under the existing environment switch.

General ATen registration remains with `flag_gems.enable()`. Device detection,
runtime utilities, ATen helpers, and Sunrise vendor launch/configuration helpers
also retain their existing sources; they are not operator replacements.

Export existence and compatible Python arguments do not establish correctness
on every device. The eight enabled platform CI pipelines evaluate runtime
compatibility; Ascend remains disabled.

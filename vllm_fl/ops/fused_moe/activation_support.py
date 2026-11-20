# SPDX-License-Identifier: Apache-2.0
"""Which MoE activations the FlagGems fused fast path can execute."""

# flag_gems.fused.fused_experts_impl asserts silu-only
# (flag_gems/fused/fused_moe.py), while vLLM/FL support relu2_no_mul and others
# through the step-by-step path (apply_moe_activation). NemotronH's MoE uses
# relu2_no_mul, so sending it down the fast path aborts the engine.
FLAGGEMS_FAST_PATH_ACTIVATIONS = ("silu",)


def flaggems_fast_path_supported(activation):
    value = getattr(activation, "value", activation)
    return value in FLAGGEMS_FAST_PATH_ACTIVATIONS

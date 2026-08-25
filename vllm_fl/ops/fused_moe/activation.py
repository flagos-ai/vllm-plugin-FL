import torch
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation as upstream_apply_moe_activation,
)

from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    activation_config: ApplyMoEActivationConfig | None = None,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply MoE activation while preserving vLLM v0.28 configuration."""
    assert input.dim() == 2, "Input must be 2D"
    assert output.dim() == 2, "Output must be 2D"
    if activation.is_gated:
        assert output.size(-1) * 2 == input.size(-1), (
            f"{activation.value} expects 2x ratio: "
            f"{output.size(-1) * 2} vs {input.size(-1)}"
        )
    else:
        assert output.size(-1) == input.size(-1), (
            f"{activation.value} expects equal sizes: "
            f"{output.size(-1)} vs {input.size(-1)}"
        )

    # Activations with gated multiplication (gate × activation(up))
    if activation == MoEActivation.SILU and (
        activation_config is None or activation_config.clamp_limit is None
    ):
        output.copy_(_silu_and_mul(None, input))
    elif activation == MoEActivation.GELU:
        output.copy_(_gelu_and_mul(None, input))
    else:
        # v0.28 adds configured clamp/SITU/SwiGLU-OAI behavior and additional
        # activation variants. Delegate these paths to the target implementation.
        upstream_apply_moe_activation(
            activation,
            output,
            input,
            activation_config=activation_config,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )

    return output

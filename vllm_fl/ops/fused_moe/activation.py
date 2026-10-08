import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import (
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
    clamp_limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """Apply MoE activation function."""
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
    if activation == MoEActivation.SILU:
        if clamp_limit is not None:
            return upstream_apply_moe_activation(
                activation,
                output,
                input,
                clamp_limit=clamp_limit,
            )
        output.copy_(_silu_and_mul(None, input))
    elif activation == MoEActivation.GELU:
        output.copy_(_gelu_and_mul(None, input))
    elif activation == MoEActivation.SWIGLUOAI:
        torch.ops._C.swigluoai_and_mul(output, input)
    elif activation == getattr(MoEActivation, "SWIGLUOAI_UNINTERLEAVE", None):
        if clamp_limit is None:
            raise ValueError("SWIGLUOAI_UNINTERLEAVE requires clamp_limit")
        # This kernel supports the packed [all gates; all ups] layout used by
        # MiniMax-M3 and preserves its configurable alpha/beta/limit math.
        from vllm.models.minimax_m3.amd.ops.swiglu_oai import swiglu_oai_split

        output.copy_(swiglu_oai_split(input, alpha, beta, clamp_limit))
    elif activation == MoEActivation.SWIGLUSTEP:
        from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

        swiglustep_and_mul_triton(output, input)

    # Activations without gated multiplication
    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output

import torch

from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation as upstream_apply_moe_activation,
)

import vllm_fl.envs as fl_envs
from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import use_flaggems_op

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
    elif activation == getattr(MoEActivation, "SWIGLUOAI_UNINTERLEAVE", None):
        if clamp_limit is None:
            raise ValueError("SWIGLUOAI_UNINTERLEAVE requires clamp_limit")
        if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM and use_flaggems_op(
            "swigluoai_uninterleave"
        ):
            from flaggems_vllm.ops.fused_moe import (
                MoEActivation as GemsMoEActivation,
                apply_moe_activation as gems_apply_moe_activation,
            )

            gems_apply_moe_activation(
                GemsMoEActivation.SWIGLUOAI_UNINTERLEAVE,
                output,
                input,
                gemm1_alpha=alpha,
                gemm1_beta=beta,
                gemm1_clamp_limit=clamp_limit,
            )
        else:
            from vllm.models.minimax_m3.amd.ops.swiglu_oai import swiglu_oai_split

            output.copy_(swiglu_oai_split(input, alpha, beta, clamp_limit))
    else:
        return upstream_apply_moe_activation(
            activation,
            output,
            input,
            clamp_limit=clamp_limit,
            alpha=alpha,
            beta=beta,
        )

    return output


def _apply_swigluoai_with_optional_quantization(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    quantize: bool,
    alpha: float | None,
    beta: float | None,
    clamp_limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply packed activation, optionally returning its fused INT8 outputs.

    The caller establishes the dynamic per-token INT8 quantization contract.
    Missing or unsupported fusion retains the ordinary activation path;
    errors from a selected implementation are allowed to propagate.
    """
    if any(value is None for value in (alpha, beta, clamp_limit)):
        raise ValueError("split SwiGLU-OAI requires explicit gate parameters")
    if (
        quantize
        and fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM
        and use_flaggems_op("swigluoai_uninterleave")
    ):
        import flaggems_vllm

        quantizer = getattr(
            flaggems_vllm, "try_swigluoai_uninterleave_quant_int8", None
        )
        if quantizer is not None:
            result = quantizer(
                output, input, clamp_limit=clamp_limit, alpha=alpha, beta=beta
            )
            if result is not None:
                return result
    apply_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        output,
        input,
        clamp_limit=clamp_limit,
        alpha=alpha,
        beta=beta,
    )
    return None

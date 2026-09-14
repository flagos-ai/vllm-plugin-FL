import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    clamp_limit: float | None = None,
) -> torch.Tensor:
    """Apply MoE activation function.

    ``clamp_limit`` comes from ``FusedMoEQuantConfig.gemm1_clamp_limit``, which
    upstream populates from the model's ``swiglu_limit``. GLM-5-Next sets
    ``swiglu_limit=10.0`` and was TRAINED with that bound, so dropping it
    silently changes every routed expert in all 42 sparse layers. Upstream
    honours it in triton_moe.py:174-176; this plugin previously ignored the
    parameter entirely, which is invisible under dummy weights (activations
    stay well inside +-10 so the clamp never fires) but destroys semantics
    with real weights.
    """
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
        if clamp_limit is None:
            output.copy_(_silu_and_mul(None, input))
        else:
            dim = input.shape[-1] // 2
            limit = float(clamp_limit)
            try:
                from flag_gems.fused.silu_and_mul_with_clamp import (
                    silu_and_mul_with_clamp_out,
                )

                silu_and_mul_with_clamp_out(
                    input[..., :dim], input[..., dim:], output, limit
                )
            except (ImportError, OSError, NotImplementedError, RuntimeError):
                # torch.ops._C.silu_and_mul_with_clamp is unavailable here
                # (no vllm._C on this platform), so compose it in torch.
                # Mirrors vLLM's _swiglu_limit_torch (utils.py:467).
                gate = input[..., :dim].clamp(max=limit)
                up = input[..., dim:].clamp(min=-limit, max=limit)
                output.copy_(F.silu(gate.float()) * up.float())
    elif activation == MoEActivation.GELU:
        output.copy_(_gelu_and_mul(None, input))
    elif activation == MoEActivation.SWIGLUOAI:
        torch.ops._C.swigluoai_and_mul(output, input)
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

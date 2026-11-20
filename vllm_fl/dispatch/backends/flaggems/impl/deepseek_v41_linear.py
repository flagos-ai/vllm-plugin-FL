# Copyright (c) 2026 BAAI. All rights reserved.
"""Thin public FGV projection, loading and shared-expert activation contracts."""

import inspect

_REQUIRED = {
    "bf16_linear_fp32": (
        "triton_scaled_mm",
        {
            "input",
            "weight",
            "scale_a",
            "scale_b",
            "out_dtype",
            "block_size_m",
            "block_size_n",
            "block_size_k",
            "use_heuristic",
        },
    ),
    "silu_and_mul_with_clamp": ("silu_and_mul_with_clamp", {"x", "y", "limit"}),
}


def is_projection_available(op):
    if op == "mean_hc":
        import flag_gems

        return callable(getattr(flag_gems, "mean_dim", None))
    if op == "grouped_bf16_linear":
        import flag_gems

        return callable(getattr(flag_gems, "bmm", None))
    try:
        import flaggems_vllm
    except ModuleNotFoundError as error:
        if error.name != "flaggems_vllm":
            raise
        return False
    name, required = _REQUIRED[op]
    implementation = getattr(flaggems_vllm, name, None)
    if not callable(implementation):
        return False
    try:
        return required.issubset(inspect.signature(implementation).parameters)
    except (TypeError, ValueError):
        return False


def bf16_linear_fp32(inputs, weight, unit_scale):
    """BF16 [M,K] x BF16 [N,K] -> FP32 [M,N], with no BF16 output roundtrip.

    Caller owns the persistent FP32 scalar one. Explicit small tiles match
    the verified MetaX resource contract and remain within the public API.
    """
    import torch
    from flaggems_vllm import triton_scaled_mm

    if inputs.ndim != 2 or weight.ndim != 2 or inputs.shape[1] != weight.shape[1]:
        raise ValueError("projection operands must be [M,K] and [N,K]")
    if inputs.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("projection operands must be BF16")
    if (
        unit_scale.shape != (1,)
        or unit_scale.dtype != torch.float32
        or unit_scale.device != inputs.device
        or weight.device != inputs.device
    ):
        raise ValueError(
            "projection tensors must share a device and scalar FP32 unit scale"
        )
    if not inputs.shape[0]:
        return torch.empty(
            (0, weight.shape[0]), dtype=torch.float32, device=inputs.device
        )
    return triton_scaled_mm(
        inputs,
        weight.t(),
        unit_scale,
        unit_scale,
        out_dtype=torch.float32,
        block_size_m=16,
        block_size_n=32,
        block_size_k=128,
        use_heuristic=False,
    )


def silu_and_mul_with_clamp(x, y, limit):
    """Clamp gate above limit and up within +/-limit, then SiLU(gate)*up."""
    from flaggems_vllm import silu_and_mul_with_clamp as implementation

    return implementation(x, y, limit)


def grouped_bf16_linear(inputs, weight, groups):
    """BF16 grouped WO_A using the public generic FlagGems BMM API."""
    import flag_gems

    if inputs.ndim != 2 or weight.ndim != 2 or groups <= 0:
        raise ValueError("grouped projection expects matrices and positive groups")
    if inputs.shape[1] != groups * weight.shape[1] or weight.shape[0] % groups:
        raise ValueError("grouped projection input/weight dimensions differ")
    n, width = inputs.shape[0], weight.shape[1]
    per_group = weight.shape[0] // groups
    lhs = inputs.reshape(n, groups, width).transpose(0, 1)
    rhs = weight.view(groups, per_group, width).transpose(1, 2)
    result = flag_gems.bmm(lhs, rhs)
    return result.transpose(0, 1).contiguous().view(n, groups * per_group)


def mean_hc(residual):
    import flag_gems

    return flag_gems.mean_dim(residual, [1], False)

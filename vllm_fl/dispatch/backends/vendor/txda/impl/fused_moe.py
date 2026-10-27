# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda vendor implementation of invoke_fused_moe_triton_kernel.

Delegates directly to vLLM's upstream invoke_fused_moe_triton_kernel,
which uses the standard Triton fused_moe_kernel.  On txda hardware the
Triton kernel is compiled through the txda Triton backend, so no
vendor-specific kernel code is needed here.
"""

import torch
from vllm.triton_utils import tl


def invoke_fused_moe_triton_kernel_txda(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
) -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )

    invoke_fused_moe_triton_kernel(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=block_shape,
        B_bias=B_bias,
    )
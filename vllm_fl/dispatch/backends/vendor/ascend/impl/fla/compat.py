# Copyright (c) 2026 BAAI. All rights reserved.

import torch

from .chunk import chunk_gated_delta_rule as ascend_chunk_gated_delta_rule


@torch.compiler.disable
def chunk_gated_delta_rule(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=False, cu_seqlens=None, chunk_indices=None,
    chunk_offsets=None, use_qk_l2norm_in_kernel=False, core_attn_out=None,
):
    """Bridge vLLM 0.28's [V, K] states to FlagGems Ascend's [K, V].

    FlagGems derives its own chunk indices and offsets from cu_seqlens.
    Preserve the upstream output-buffer contract when one is supplied.
    """
    state = (
        initial_state.transpose(-1, -2).contiguous()
        if initial_state is not None else None
    )
    output, final_state = ascend_chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    if core_attn_out is not None:
        destination = core_attn_out.view(-1)[:output.numel()].view_as(output)
        destination.copy_(output)
        output = destination
    return output, final_state

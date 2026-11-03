# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from MetaX-MACA/vLLM-metax:
# vllm_metax/registry/custom_ops/layers/mamba/qwen_gdn_linear_attn.py

"""MetaX GDN prefill compatibility for vLLM 0.32.

Keep the upstream attention layer and its projection/state-management logic.
Only the chunk op needs a bridge for the optional external FLA implementation.
"""

import inspect
import os

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
)


@ChunkGatedDeltaRule.register_oot
class MacaChunkGatedDeltaRule(ChunkGatedDeltaRule):
    # ChunkGatedDeltaRule.__init__ selects forward_native directly for the
    # Triton/FLA backend, even on OOT platforms. Override that entry point as
    # well as forward_oot so this bridge is actually used in vLLM 0.32.
    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        aiter_prefill_metadata: object | None = None,
    ):
        if os.getenv("MACA_VLLM_USE_EXTERNAL_FLA", "0") != "1":
            return super().forward_native(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                core_attn_out=core_attn_out,
                aiter_prefill_metadata=aiter_prefill_metadata,
            )

        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        parameters = inspect.signature(chunk_gated_delta_rule).parameters
        kwargs = {}
        # vLLM 0.32 stores recurrent state as [N, HV, V, K]. Older FLA
        # releases use [N, HV, K, V]; newer releases expose a layout flag.
        transpose_state = False
        if "state_v_first" in parameters:
            kwargs["state_v_first"] = True
        elif "transpose_state_layout" in parameters:
            kwargs["transpose_state_layout"] = True
        else:
            transpose_state = True
        fla_state = initial_state
        if transpose_state and fla_state is not None:
            fla_state = fla_state.transpose(-1, -2).contiguous()

        output, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=fla_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )
        if transpose_state and final_state is not None:
            final_state = final_state.transpose(-1, -2).contiguous()
        if core_attn_out is not None:
            # Only the live tokens are written; the remainder may belong to
            # graph padding. The caller consumes this buffer, not just output.
            flat_output = output.squeeze(0).reshape(-1)
            core_attn_out.reshape(-1)[: flat_output.numel()].copy_(flat_output)
        return output, final_state

    forward_oot = forward_native

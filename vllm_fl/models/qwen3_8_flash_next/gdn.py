# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from plugin-FL PR573 at 45b24bc7748720e0022a72e47a983398ddda36d3.
"""Qwen-local FGV GDN prefill/non-spec providers; preserve native MTP semantics."""

from __future__ import annotations

import torch
from torch import nn

from vllm.forward_context import get_forward_context

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    GDNAttentionMetadata,
    QwenGatedDeltaNetAttention,
    causal_conv1d_update,
    is_conv_state_dim_first as is_gdn_conv_state_dim_first,
)

from vllm_fl.dispatch import resolve_op


class _FGVPrefill(nn.Module):
    """Use the upstream per-instance chunk hook, without replacing other models."""

    def __init__(self):
        super().__init__()
        self._implementation = resolve_op("gdn_prefill")

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_offsets=None,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=None,
    ):
        return self._implementation(
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
        )


class Qwen38GatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Keep upstream core/metadata and scope FGV to its supported contracts.

    Speculative recurrent and causal-convolution calls stay in vLLM 0.24 and
    compile through the configured FlagTree backend. FGV 0f3df8e recurrent
    lacks upstream NULL_BLOCK_ID=0 read/write guards, and its causal_conv1d
    rejects accepted-token history rotation. Neither is safe to substitute
    into the inherited MTP path merely to increase operator coverage.
    """

    def __init__(
        self,
        config,
        vllm_config,
        prefix="",
        gqa_interleaved_layout=False,
    ):
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=gqa_interleaved_layout,
        )
        self.chunk_gated_delta_rule = _FGVPrefill()
        # Retain the upstream metadata-builder backend vocabulary. The exact
        # numerical provider is recorded separately below.
        self.gdn_prefill_backend = "triton"
        self._packed_decode = resolve_op("gdn_packed_decode")
        self._mixed_qkv = resolve_op("qwen38_gdn_mixed_qkv")
        self.qwen38_gdn_providers = {
            "prefill": "FGV MetaX chunk_gated_delta_rule_fwd (FP32 V-major state)",
            "decode_non_spec": "FGV MetaX packed_decode",
            "decode_spec": "vLLM 0.24 fused_sigmoid_gating_delta_rule_update",
            "causal_conv": "vLLM 0.24 causal_conv1d (accepted-token support)",
        }

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        # Upstream gathers only when speculative and non-speculative tokens
        # coexist. Its QKV view can retain the larger QKVZ projection stride.
        # Keep provider selection local to this branch and reuse the full core.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata.get(self.prefix)
            if (
                isinstance(metadata, GDNAttentionMetadata)
                and metadata.spec_sequence_masks is not None
                and (metadata.num_prefills > 0 or metadata.num_decodes > 0)
            ):
                mixed_qkv = self._mixed_qkv(mixed_qkv)
        return super()._forward_core(mixed_qkv, b, a, core_attn_out)

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """Use packed FGV only for the upstream one-token non-spec branch."""
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        conv_state = (
            self.kv_cache[0]
            if is_gdn_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self.kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            validate_data=False,
        )
        self._packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=core_attn_out[:num_actual_tokens].unsqueeze(1),
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            use_qk_l2norm_in_kernel=True,
        )

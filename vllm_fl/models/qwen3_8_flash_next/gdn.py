# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from plugin-FL PR573 at 45b24bc7748720e0022a72e47a983398ddda36d3.
"""Qwen-local FGV GDN prefill/non-spec providers; preserve native MTP semantics."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    GDNAttentionMetadata,
    QwenGatedDeltaNetAttention,
    causal_conv1d_update,
    is_conv_state_dim_first as is_gdn_conv_state_dim_first,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import (
    NULL_BLOCK_ID,
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)

from vllm_fl.dispatch import resolve_op


class Qwen38GDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """Restore the accepted state even when this step has no draft tokens."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.supports_update_block_table = bool(
            vllm_config.additional_config.get("qwen38_reuse_gdn_metadata", False)
            and getattr(current_platform, "vendor_name", None) == "metax"
            and device.type == "cuda"
            and self.speculative_config is None
            and vllm_config.cache_config.mamba_cache_mode == "none"
            and self.gdn_prefill_backend == "triton"
        )

    def update_block_table(self, metadata, blk_table, slot_mapping):
        """Use MRV1's per-step cache; every group keeps its own state addresses."""
        del slot_mapping  # Mamba cache mode none reads column zero of its table.
        if (
            not self.supports_update_block_table
            or metadata.spec_sequence_masks is not None
        ):
            raise RuntimeError(
                "GDN metadata reuse requires non-speculative cache mode none"
            )
        if blk_table.ndim != 2 or blk_table.shape[1] < 1:
            raise ValueError("GDN cache group requires a state block column")
        indices = blk_table[:, 0]
        query_starts = metadata.non_spec_query_start_loc
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_spec_decodes == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
        ):
            # Capture uses this builder's fixed buffers. Updating only the
            # returned Python metadata would leave captured graph inputs stale.
            batch_size = metadata.non_spec_state_indices_tensor.numel()
            count = metadata.num_decodes
            self.non_spec_state_indices_tensor[:count].copy_(indices, non_blocking=True)
            indices = self.non_spec_state_indices_tensor[:batch_size]
            indices[count:].fill_(NULL_BLOCK_ID)
            self.non_spec_query_start_loc[: batch_size + 1].copy_(
                query_starts, non_blocking=True
            )
            query_starts = self.non_spec_query_start_loc[: batch_size + 1]
        prefill_indices = None
        if metadata.num_prefills > 0:
            prefill_indices = indices[metadata.num_decodes :]
        return replace(
            metadata,
            non_spec_state_indices_tensor=indices,
            non_spec_query_start_loc=query_starts,
            prefill_state_indices=prefill_indices,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            # vLLM 0.24 bypasses its speculative path when the sum of draft
            # counts is zero, losing the previous accepted-1 state offset.
            # These counts are only classification markers inside the builder;
            # the real token counts come from query_start_loc. Keep prefill's
            # negative sentinel and the caller's scheduler counts unchanged.
            num_decode_draft_tokens_cpu = torch.where(
                num_decode_draft_tokens_cpu == 0,
                1,
                num_decode_draft_tokens_cpu,
            )
        return super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )


class Qwen38GDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Qwen38GDNAttentionMetadataBuilder]:
        return Qwen38GDNAttentionMetadataBuilder


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

    def get_attn_backend(self) -> type[Qwen38GDNAttentionBackend]:
        return Qwen38GDNAttentionBackend

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

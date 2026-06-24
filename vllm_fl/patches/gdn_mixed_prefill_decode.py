# SPDX-License-Identifier: Apache-2.0
"""Backport vLLM #44700 for GDN mixed prefill+decode batches.

The upstream optimization peels 1-token non-spec decodes from a mixed
prefill+decode GDN batch and routes them through the recurrent update kernel.
Only the prefill tail goes through the chunk kernel, avoiding padding every
decode to FLA_CHUNK_SIZE.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _enabled() -> bool:
    return os.environ.get(
        "VLLM_FL_ENABLE_GDN_MIXED_PREFILL_DECODE_SPLIT", "0"
    ).lower() not in {"0", "false", "no", "off"}


def apply_gdn_mixed_prefill_decode_patch() -> None:
    """Apply the GDN mixed prefill+decode split patch once."""
    global _PATCHED
    if _PATCHED or not _enabled():
        return

    from vllm.v1.attention.backends import gdn_attn as gdn_attn_mod
    from vllm.v1.attention.backends.gdn_attn import (
        GDNAttentionMetadata,
        GDNAttentionMetadataBuilder,
    )

    if "prefill_query_start_loc" in GDNAttentionMetadata.__annotations__:
        logger.info("Skip GDN mixed prefill+decode split patch; vLLM has it")
        _PATCHED = True
        return

    from vllm.model_executor.layers.fla.ops import (
        fused_post_conv_prep,
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.model_executor.layers.mamba import gdn_linear_attn as gdn_mod
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        GatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import (
        is_conv_state_dim_first,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    original_build = GDNAttentionMetadataBuilder.build
    original_forward_core = GatedDeltaNetAttention._forward_core

    if getattr(original_build, "_fl_gdn_split_patched", False):
        _PATCHED = True
        return

    def patched_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        metadata = original_build(
            self,
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )

        metadata.prefill_query_start_loc = None
        metadata.prefill_state_indices = None
        metadata.prefill_has_initial_state = None

        if metadata.num_prefills <= 0:
            return metadata

        split_non_spec = (
            metadata.spec_sequence_masks is None
            and metadata.num_decodes > 0
            and metadata.non_spec_query_start_loc is not None
            and metadata.non_spec_state_indices_tensor is not None
        )

        if split_non_spec:
            num_decodes = metadata.num_decodes
            num_decode_tokens = metadata.num_decode_tokens
            metadata.prefill_query_start_loc = (
                metadata.non_spec_query_start_loc[num_decodes:] - num_decode_tokens
            )
            metadata.prefill_state_indices = metadata.non_spec_state_indices_tensor[
                num_decodes:
            ]
            if metadata.has_initial_state is not None:
                metadata.prefill_has_initial_state = metadata.has_initial_state[
                    num_decodes:
                ]

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            prefill_query_start_loc_cpu = (
                query_start_loc_cpu[num_decodes:] - num_decode_tokens
            )

            from vllm.model_executor.layers.fla.ops.index import (
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            gpu_device = common_attn_metadata.query_start_loc.device
            metadata.chunk_indices = prepare_chunk_indices(
                prefill_query_start_loc_cpu, FLA_CHUNK_SIZE
            ).to(device=gpu_device, non_blocking=True)
            metadata.chunk_offsets = prepare_chunk_offsets(
                prefill_query_start_loc_cpu, FLA_CHUNK_SIZE
            ).to(device=gpu_device, non_blocking=True)
        else:
            metadata.prefill_query_start_loc = metadata.non_spec_query_start_loc
            metadata.prefill_state_indices = metadata.non_spec_state_indices_tensor
            metadata.prefill_has_initial_state = metadata.has_initial_state

        return metadata

    def patched_forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        forward_context = gdn_mod.get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if split_non_spec:
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=conv_output_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        if split_non_spec:
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec[:num_decode_tokens]
            )
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=self.dt_bias,
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        if attn_metadata.num_prefills > 0:
            prefill_state_indices = getattr(
                attn_metadata, "prefill_state_indices", None
            )
            prefill_has_initial_state = getattr(
                attn_metadata, "prefill_has_initial_state", None
            )
            prefill_query_start_loc = getattr(
                attn_metadata, "prefill_query_start_loc", None
            )
            if prefill_state_indices is None:
                prefill_state_indices = non_spec_state_indices_tensor
            if prefill_has_initial_state is None:
                prefill_has_initial_state = has_initial_state
            if prefill_query_start_loc is None:
                prefill_query_start_loc = non_spec_query_start_loc

            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = ssm_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
            ssm_state[prefill_state_indices] = last_recurrent_state.to(
                ssm_state.dtype
            )

            if split_non_spec:
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec], dim=1
                )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    patched_build._fl_gdn_split_patched = True  # type: ignore[attr-defined]
    patched_build._fl_gdn_split_original = original_build  # type: ignore[attr-defined]
    patched_forward_core._fl_gdn_split_original = (  # type: ignore[attr-defined]
        original_forward_core
    )

    GDNAttentionMetadataBuilder.build = patched_build
    GatedDeltaNetAttention._forward_core = patched_forward_core

    # Document the dynamic fields for introspection without requiring a vLLM
    # source edit. The dataclass has no slots, so instances can carry them.
    gdn_attn_mod.GDNAttentionMetadata.__annotations__.setdefault(
        "prefill_query_start_loc", torch.Tensor | None
    )
    gdn_attn_mod.GDNAttentionMetadata.__annotations__.setdefault(
        "prefill_state_indices", torch.Tensor | None
    )
    gdn_attn_mod.GDNAttentionMetadata.__annotations__.setdefault(
        "prefill_has_initial_state", torch.Tensor | None
    )

    _PATCHED = True
    logger.info("Applied GDN mixed prefill+decode split patch (vLLM #44700)")

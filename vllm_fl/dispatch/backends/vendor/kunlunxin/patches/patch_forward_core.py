# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin override: replace _forward_core to avoid Triton kernels that
produce incorrect numerical results on XPU.

Key changes vs upstream:
1. Disable enable_packed_recurrent_decode fast path (uses Triton kernel)
2. Replace fused_sigmoid_gating_delta_rule_update (Triton) in decode path
   with fused_gdn_gating + fused_recurrent_gated_delta_rule (kunlunxin impls)
3. Replace ssm_state direct indexing with KunlunxinPagedAttention.reshape_and_cache_flash
"""

import logging
import torch

logger = logging.getLogger(__name__)


def _kunlunxin_write_ssm_cache(ssm_state, last_recurrent_state, indices):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinPagedAttention,
    )
    last_recurrent_state = (
        last_recurrent_state.to(ssm_state.dtype)
        .view(last_recurrent_state.shape[0], -1, last_recurrent_state.shape[-1])
    )
    cast_ssm_state = ssm_state.view(
        ssm_state.shape[0], 1, -1, ssm_state.shape[-1]
    )
    KunlunxinPagedAttention.reshape_and_cache_flash(
        last_recurrent_state, None, cast_ssm_state, None, indices,
    )


def apply_ssm_patch():
    import vllm.model_executor.layers.mamba.gdn_linear_attn as gdn_mod
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    cls = gdn_mod.GatedDeltaNetAttention
    is_conv_state_dim_first = gdn_mod.is_conv_state_dim_first
    causal_conv1d_fn = gdn_mod.causal_conv1d_fn
    causal_conv1d_update = gdn_mod.causal_conv1d_update
    fused_post_conv_prep = gdn_mod.fused_post_conv_prep

    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule as klx_fused_recurrent,
    )
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.chunk import (
        chunk_gated_delta_rule as klx_chunk_gated_delta_rule,
    )
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_gdn_gating import (
        fused_gdn_gating_kunlunxin,
    )

    def _forward_core_kunlunxin(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # Skip packed decode fast path - Triton kernel broken on XPU

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

        # 1. Convolution
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

        # 1.1: spec part
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

        # 1.2: non-spec part
        if attn_metadata.num_prefills > 0:
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

        # Rearrange spec qkv
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # For non-spec: use fused_post_conv_prep for prefill (computes g/beta + l2norm)
        # For decode: use rearrange + fused_gdn_gating (avoids Triton fused_sigmoid kernel)
        if attn_metadata.num_prefills > 0 and mixed_qkv_non_spec is not None:
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            # klx: bypass fused_post_conv_prep Triton kernel (broken on XPU)
            # Do rearrange + gating + l2norm in PyTorch
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            # Manual l2norm on q and k (avoids xtorch_ops l2norm precision issues)
            query_non_spec = torch.nn.functional.normalize(query_non_spec, p=2, dim=-1)
            key_non_spec = torch.nn.functional.normalize(key_non_spec, p=2, dim=-1)
            # Compute g/beta via kunlunxin fused_gdn_gating
            # fused_gdn_gating_kunlunxin expects a: [num_tokens, HV], returns g: [1, num_tokens, HV]
            g_non_spec, beta_non_spec = fused_gdn_gating_kunlunxin(
                self.A_log, a_non_spec, b_non_spec, self.dt_bias
            )
        elif attn_metadata.num_decodes > 0 and mixed_qkv_non_spec is not None:
            # Decode path: just rearrange qkv (gating computed later in recurrent section)
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec, beta_non_spec = None, None
        else:
            query_non_spec, key_non_spec, value_non_spec = None, None, None
            g_non_spec, beta_non_spec = None, None

        # 2. Recurrent attention

        # 2.1: spec part
        if spec_sequence_masks is not None:
            # Spec path also uses fused_gdn_gating + fused_recurrent
            g_spec, beta_spec = fused_gdn_gating_kunlunxin(
                self.A_log, a, b, self.dt_bias
            )
            if non_spec_token_indx is not None and attn_metadata.num_prefills > 0:
                g_spec = g_spec.index_select(1, spec_token_indx)
                beta_spec = beta_spec.index_select(1, spec_token_indx)

            core_attn_out_spec, last_recurrent_state = klx_fused_recurrent(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: non-spec part
        if attn_metadata.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            assert has_initial_state is not None
            initial_state[~has_initial_state, ...] = 0
            # klx fix: use xtorch_ops chunk kernel (replaces FLA Triton which
            # produces numerically incorrect SSM final_state on XPU).
            # q/k already L2-normalized above, so use_qk_l2norm_in_kernel=False.
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = klx_chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=False,
            )
            # klx diff: write ssm cache
            _kunlunxin_write_ssm_cache(
                ssm_state, last_recurrent_state, non_spec_state_indices_tensor
            )
        elif attn_metadata.num_decodes > 0:
            # klx: compute g/beta manually in PyTorch (avoiding xtorch_ops precision issues)
            # fused_sigmoid_gating_delta_rule_update is not registered for XPU, cannot use
            if spec_sequence_masks is not None:
                a_dec = a.index_select(0, non_spec_token_indx)
                b_dec = b.index_select(0, non_spec_token_indx)
            else:
                a_dec = a
                b_dec = b
            # g = -exp(A_log) * softplus(a + dt_bias), shape [num_tokens, HV] -> [1, num_tokens, HV]
            x = (a_dec + self.dt_bias).float()
            sp = torch.nn.functional.softplus(x)
            g_non_spec = (-torch.exp(self.A_log.float()) * sp).unsqueeze(0)
            beta_non_spec = torch.sigmoid(b_dec.float()).to(torch.bfloat16).unsqueeze(0)
            # Manual L2 norm
            query_non_spec = torch.nn.functional.normalize(query_non_spec, p=2, dim=-1)
            key_non_spec = torch.nn.functional.normalize(key_non_spec, p=2, dim=-1)
            core_attn_out_non_spec, last_recurrent_state = klx_fused_recurrent(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge
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

    cls._forward_core = _forward_core_kunlunxin
    logger.info("Patched GatedDeltaNetAttention._forward_core for Kunlunxin")

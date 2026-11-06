# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Adapt upstream GDN metadata to FL-dispatched numerical operators."""
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm_fl.dispatch import CachedOp

_gdn_decode = CachedOp("gdn_decode")
_gdn_prefill = CachedOp("gdn_prefill")


class AscendQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Non-speculative reference path, sharing eager and captured math."""

    def _warmup_prefill_kernels(self, *args, **kwargs):
        # There are no Triton prefill kernels in this reference implementation.
        return

    def _operator_kwargs(self):
        return dict(
            conv_weights=self.conv1d.weight.view(self.conv1d.weight.size(0), -1),
            conv_bias=self.conv1d.bias, activation=self.activation,
            A_log=self.A_log, dt_bias=self.dt_bias,
            num_key_heads=self.num_k_heads // self.tp_size,
            num_value_heads=self.num_v_heads // self.tp_size,
            head_k_dim=self.head_k_dim, head_v_dim=self.head_v_dim,
        )

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return
        metadata = metadata[self.prefix]
        if metadata.spec_sequence_masks is not None:
            raise NotImplementedError("Ascend reference GDN does not support speculative decoding")
        conv_state, ssm_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        indices = metadata.non_spec_state_indices_tensor
        num_decode = metadata.num_decode_tokens
        op_kwargs = self._operator_kwargs()
        if metadata.num_prefills == 0:
            n = metadata.num_actual_tokens
            _gdn_decode(
                mixed_qkv[:n], a[:n], b[:n], indices[:n],
                conv_state, ssm_state, core_attn_out[:n],
                packed_beta=self.enable_packed_recurrent_decode,
                **op_kwargs,
            )
            return
        if num_decode:
            _gdn_decode(
                mixed_qkv[:num_decode], a[:num_decode], b[:num_decode],
                indices[:num_decode], conv_state, ssm_state,
                core_attn_out[:num_decode],
                packed_beta=False,
                **op_kwargs,
            )

        # Variable-length prefill is outside FULL_DECODE_ONLY capture. Sequence
        # boundaries may be read on host here; never do so in the decode path.
        starts = metadata.non_spec_query_start_loc.cpu().tolist()
        slots = indices.cpu().tolist()
        for seq in range(metadata.num_decodes, len(starts) - 1):
            start, end = starts[seq : seq + 2]
            slot = slots[seq]
            if slot <= 0 or start == end:
                continue
            _gdn_prefill(
                mixed_qkv[start:end], a[start:end], b[start:end],
                indices[seq : seq + 1], metadata.has_initial_state[seq : seq + 1],
                conv_state, ssm_state[slot], core_attn_out[start:end],
                **op_kwargs,
            )

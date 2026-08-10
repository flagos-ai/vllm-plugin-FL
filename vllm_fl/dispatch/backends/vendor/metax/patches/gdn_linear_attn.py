# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
# Ported from vllm_metax customized/pluggable_layer/gdn_linear_attn.py
#
# Two fixes applied:
#
# Fix 1 — register_oot subclass (MacaGatedDeltaNetAttention):
#   Fixes TorchDynamo guard-branch issue: replaces hasattr() check with a plain
#   bool flag set in __init__. Uses self.prefix (no _encode_layer_name()) so
#   gdn_attention_core state lookup works correctly on MetaX mcoplib.
#
# Fix 2 — monkey-patch GatedDeltaNetAttention.forward_cuda (at module bottom):
#   register_oot/__new__ only works when op_registry_oot is populated BEFORE
#   model layer instantiation. In vllm v1, model layers are built before
#   dispatch/patches initialize, so __new__ always selects base class.
#   Monkey-patching forward_cuda at class level ensures the fix is applied
#   regardless of which class __new__ selected.

import torch
from einops import rearrange

from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.config import VllmConfig


@GatedDeltaNetAttention.register_oot
class MacaGatedDeltaNetAttention(GatedDeltaNetAttention):
    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        create_in_proj_qkvz: bool = True,
        gqa_interleaved_layout=False,
    ) -> None:
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            create_in_proj_qkvz=create_in_proj_qkvz,
            gqa_interleaved_layout=gqa_interleaved_layout,
        )
        # Use a plain bool flag instead of hasattr() to avoid TorchDynamo
        # guard-branch issues (https://github.com/pytorch/pytorch/issues/...)
        self.create_in_proj_qkvz = create_in_proj_qkvz

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)

        # ── Part 1: Input Projection ─────────────────────────────────────────
        if not self.create_in_proj_qkvz:
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value),
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ── Part 2: Core Attention (Custom Op) ───────────────────────────────
        # Use torch.zeros, not torch.empty: uninitialized memory causes incorrect
        # results on MACA hardware for the chunked recurrent accumulation.
        # Ref: https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ── Part 3: Output Projection ─────────────────────────────────────────
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

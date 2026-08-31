# SPDX-License-Identifier: Apache-2.0
"""ModelRegistry thin model adding W8A8 DSV4 output projection support."""

from __future__ import annotations

from itertools import islice

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import PPMissingLayer, make_layers
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4MoE,
    _select_dsv4_attn_cls,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op, vllm_lib

from vllm_fl.ops.deepseek_v4 import (
    hc_head,
    int8_scaled_mm,
    inv_rope_quant_fp8,
    mhc_fused_post_pre,
    mhc_post,
    mhc_pre,
)
from vllm_fl.ops.deepseek_v4_int8_woa import fused_inv_rope_quant_int8


def _patch_hopper_fp8_inv_rope_kernel() -> None:
    """Bridge the vLLM/Triton PDL argument mismatch on pre-SM100 GPUs."""
    capability = current_platform.get_device_capability()
    if capability is None or capability.major >= 10:
        return
    vllm_lib.impl(
        "fused_inv_rope_fp8_quant_kernel",
        inv_rope_quant_fp8,
        dispatch_key=current_platform.dispatch_key,
        allow_override=True,
    )


_patch_hopper_fp8_inv_rope_kernel()


def _deepseek_v4_fl_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        out,
    )


def _deepseek_v4_fl_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    del (
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        out,
        layer_name,
    )


direct_register_custom_op(
    op_name="deepseek_v4_fl_attention",
    op_func=_deepseek_v4_fl_attention,
    mutates_args=["out"],
    fake_impl=_deepseek_v4_fl_attention_fake,
)


class DeepseekV4FLFlashMLAAttention(DeepseekV4FlashMLAAttention):
    """FlashMLA attention with a W8A8-only wo_a branch."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        torch.ops.vllm.deepseek_v4_fl_attention(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
            self.prefix,
        )
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        grouped_weight = getattr(self.wo_a, "_fl_w8a8_grouped_weight", None)
        grouped_scale = getattr(self.wo_a, "_fl_w8a8_grouped_weight_scale", None)
        if grouped_weight is None or grouped_scale is None:
            weight = getattr(self.wo_a, "weight", None)
            weight_scale = getattr(self.wo_a, "weight_scale", None)
            if weight is None or weight.dtype != torch.int8 or weight_scale is None:
                return super()._o_proj(o, positions)
            output_per_group = weight.shape[1] // self.n_local_groups
            grouped_scale = weight_scale.reshape(self.n_local_groups, output_per_group)

        o_q, o_scale = fused_inv_rope_quant_int8(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.n_local_groups,
            self.n_local_heads // self.n_local_groups,
            self.nope_head_dim,
            self.rope_head_dim,
        )
        outputs = []
        for group_idx in range(self.n_local_groups):
            if grouped_weight is None:
                start = group_idx * output_per_group
                group_weight = weight[:, start : start + output_per_group]
            else:
                group_weight = grouped_weight[group_idx].transpose(0, 1)
            outputs.append(
                int8_scaled_mm(
                    o_q[group_idx],
                    group_weight,
                    o_scale[group_idx],
                    grouped_scale[group_idx],
                    o.dtype,
                )
            )
        return self.wo_b(torch.stack(outputs, dim=1).flatten(1))


class DeepseekV4FLDecoderLayer(DeepseekV4DecoderLayer):
    """Upstream decoder layer with a thin attention-class substitution."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        attn_cls = _select_dsv4_attn_cls(vllm_config)
        if attn_cls is DeepseekV4FlashMLAAttention:
            attn_cls = DeepseekV4FLFlashMLAAttention
        self.attn = attn_cls(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep NVIDIA MHC kernels opaque to Dynamo via registered ops."""
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            residual = x
            post_mix, res_mix, x = mhc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                1,
                attn_norm_weight,
                attn_norm_eps,
            )
        else:
            assert post_mix is not None and res_mix is not None
            residual, post_mix, res_mix, x = mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                1,
                1,
                attn_norm_weight,
                attn_norm_eps,
            )

        x = self.attn(positions, x, None)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x = mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            1,
            1,
            ffn_norm_weight,
            ffn_norm_eps,
        )
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class DeepseekV4FLModel(DeepseekV4Model):
    """Upstream DSV4 model whose layers use the FL decoder subclass."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError("DeepSeek V4 MegaMoE requires expert parallel")
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4FLDecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the upstream model flow with registered MHC custom ops."""
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        residual, post_mix, res_mix = None, None, None
        layer = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        if layer is not None:
            hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return self.norm(hidden_states)


class DeepseekV4FLForCausalLM(DeepseekV4ForCausalLM):
    """Registry entry retaining upstream behavior outside INT8 wo_a."""

    model_cls = DeepseekV4FLModel


__all__ = ["DeepseekV4FLForCausalLM"]

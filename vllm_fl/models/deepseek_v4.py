# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 model-owned composition with dispatch-selected math providers.

These building blocks retain checkpoint names and explicit norm placement.
Cache allocation and scheduler metadata remain outside the model.
"""

import torch
from torch import nn

from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe.layer import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
)
from vllm.sequence import IntermediateTensors

from vllm_fl.dispatch import CachedOp
from vllm_fl.kv_cache.deepseek_v4 import create_deepseek_v4_cache_layers
from vllm_fl.models.deepseek_v4_attention import DeepseekV4Attention
from vllm_fl.ops.activation import SiluAndMulFL, SiluAndMulWithClampFL
from vllm_fl.ops.fused_moe.layer import FusedMoEFL
from vllm_fl.ops.layernorm import RMSNormFL as RMSNorm

_hc_pre = CachedOp("deepseek_v4_hc_pre")
_hc_post = CachedOp("deepseek_v4_hc_post")
_hc_head = CachedOp("deepseek_v4_hc_head")


class DeepseekV4DecoderLayer(nn.Module):
    """Compose injected attention/MoE modules without binding their math backend."""

    def __init__(
        self,
        hidden_size,
        hc_mult,
        hc_sinkhorn_iters,
        hc_eps,
        rms_eps,
        self_attn,
        mlp,
        *,
        params_dtype=torch.bfloat16,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError("The current HC decoder requires BF16 residuals")
        self.self_attn = self_attn
        self.mlp = mlp
        self.norm_eps = rms_eps
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.input_layernorm = RMSNorm(hidden_size, rms_eps, dtype=params_dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, rms_eps, dtype=params_dtype
        )
        mix = (2 + hc_mult) * hc_mult
        for block in ("attn", "ffn"):
            for name, shape in (
                ("fn", (mix, hc_mult * hidden_size)),
                ("base", (mix,)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{block}_{name}",
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )

    def hc_pre(self, x, fn, scale, base):
        return _hc_pre(
            x, fn, scale, base, self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters
        )

    def hc_post(self, x, residual, post, comb):
        return _hc_post(x, residual, post, comb)

    def forward(self, positions, hidden_states, input_ids=None, llama_4_scaling=None):
        residual = hidden_states
        x, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(
            positions=positions, hidden_states=x, llama_4_scaling=llama_4_scaling
        )
        hidden_states = self.hc_post(x, residual, post, comb)
        residual = hidden_states
        x, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, input_ids=input_ids)
        return self.hc_post(x, residual, post, comb)


def _make_rope_cache(config, max_model_len, *, compressed):
    # YaRN correction uses the original context, not the expanded runtime limit.
    params = config.rope_parameters
    factor = params.get("factor", 1.0)
    rotary = DeepseekScalingRotaryEmbedding(
        head_size=config.qk_rope_head_dim,
        rotary_dim=config.qk_rope_head_dim,
        max_position_embeddings=params.get(
            "original_max_position_embeddings", config.max_position_embeddings
        ),
        base=config.compress_rope_theta if compressed else config.rope_theta,
        is_neox_style=False,
        scaling_factor=factor,
        dtype=torch.float32,
        beta_fast=params.get("beta_fast", 32),
        beta_slow=params.get("beta_slow", 1),
        mscale=0,
        mscale_all_dim=0,
        init_cache=False,
    )
    inv_freq = rotary._compute_inv_freq(factor)
    phases = torch.outer(
        torch.arange(max_model_len, device=inv_freq.device, dtype=torch.float32),
        inv_freq,
    )
    return torch.cat((phases.cos(), phases.sin()), dim=-1)


class DeepseekV4MLP(nn.Module):
    def __init__(self, config, intermediate_size, quant_config, *, prefix):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("DeepSeek V4 requires SiLU activation")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            return_bias=False,
        )
        limit = getattr(config, "swiglu_limit", None)
        self.act_fn = SiluAndMulWithClampFL(limit) if limit else SiluAndMulFL()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            return_bias=False,
            reduce_results=False,
        )

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class DeepseekV4MoE(nn.Module):
    def __init__(self, config, quant_config, *, prefix):
        super().__init__()
        self.is_hash = extract_layer_index(prefix) < config.num_hash_layers
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        if self.is_hash:
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size, config.num_experts_per_tok, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        shared_count = config.n_shared_experts or 0
        shared = (
            DeepseekV4MLP(
                config,
                shared_count * config.moe_intermediate_size,
                quant_config,
                prefix=f"{prefix}.shared_experts",
            )
            if shared_count
            else None
        )
        self.shared_experts = shared
        # The generic runner owns shared-expert addition, reduction and routing
        # scale. Do not repeat any of those operations around its output.
        self.experts = FusedMoEFL(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            shared_experts=shared,
            gate=self.gate,
            router_logits_dtype=torch.float32,
        )

    def forward(self, x, input_ids=None):
        if self.is_hash and input_ids is None:
            raise ValueError("DeepSeek V4 hash routing requires input IDs")
        logits = x if self.experts.is_internal_router else self.gate(x)[0]
        return self.experts(x, router_logits=logits, input_ids=input_ids)


class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__()
        config = self.config = vllm_config.model_config.hf_config
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("DeepSeek V4 model requires BF16")
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                prefix=f"{prefix}.embed_tokens",
            )
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )
        rope_caches = {}

        def make_decoder(prefix):
            compressed = config.compress_ratios[extract_layer_index(prefix)] != 0
            if compressed not in rope_caches:
                rope_caches[compressed] = _make_rope_cache(
                    config,
                    vllm_config.model_config.max_model_len,
                    compressed=compressed,
                )
            attn_prefix = f"{prefix}.self_attn"
            attention = DeepseekV4Attention(
                vllm_config,
                config,
                create_deepseek_v4_cache_layers(
                    config, vllm_config.cache_config, attn_prefix
                ),
                rope_caches[compressed],
                prefix=attn_prefix,
            )
            return DeepseekV4DecoderLayer(
                config.hidden_size,
                config.hc_mult,
                config.hc_sinkhorn_iters,
                config.hc_eps,
                config.rms_norm_eps,
                attention,
                DeepseekV4MoE(config, vllm_config.quant_config, prefix=f"{prefix}.mlp"),
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, make_decoder, prefix=f"{prefix}.layers"
        )
        self.norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        # Checkpoints contain these root parameters on every PP rank. Keep
        # their ownership uniform; only the final rank executes the HC head.
        for name, shape in (
            ("fn", (config.hc_mult, config.hc_mult * config.hidden_size)),
            ("base", (config.hc_mult,)),
            ("scale", (1,)),
        ):
            self.register_parameter(
                f"hc_head_{name}",
                nn.Parameter(
                    torch.empty(shape, dtype=torch.float32), requires_grad=False
                ),
            )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.empty(
                    batch_size,
                    self.hc_mult,
                    self.config.hidden_size,
                    dtype=dtype,
                    device=device,
                )
            }
        )

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if get_pp_group().is_first_rank:
            hidden = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            hidden = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            hidden = intermediate_tensors["hidden_states"]
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden = layer(positions, hidden, input_ids=input_ids)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden})
        hidden = _hc_head(
            hidden,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm_eps,
            self.hc_eps,
        )
        return self.norm(hidden)


class DeepseekV4ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts, SupportsQuant):
    """Checkpoint-facing envelope; operator implementations remain dispatched."""

    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    hf_to_vllm_mapper = WeightsMapper(
        # One checkpoint contract for quantization metadata and tensor loading.
        # Canonical names remain unchanged, including model-prefixed raw aliases.
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed_tokens.",
            "embed_tokens.": "model.embed_tokens.",
            "head.": "lm_head.",
            "norm.": "model.norm.",
            "hc_head_": "model.hc_head_",
            "model.embed.": "model.embed_tokens.",
            "model.head.": "lm_head.",
            "model.lm_head.": "lm_head.",
        },
        orig_to_new_substr={
            ".attn.": ".self_attn.",
            ".w1.": ".gate_proj.",
            ".w2.": ".down_proj.",
            ".w3.": ".up_proj.",
            ".ffn.": ".mlp.",
            ".ffn_norm.": ".post_attention_layernorm.",
            ".attn_norm.": ".input_layernorm.",
            ".gate.bias": ".gate.e_score_correction_bias",
        },
        orig_to_new_suffix={".scale": ".weight_scale"},
    )

    def _maybe_apply_model_mapping(self):
        from vllm_fl.quantization.w8a8.compressed_tensors import (
            bind_compressed_tensors_model,
        )

        if not bind_compressed_tensors_model(
            self.quant_config, self.hf_to_vllm_mapper, self.packed_modules_mapping
        ):
            super()._maybe_apply_model_mapping()

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepseekV4Model(
            vllm_config=vllm_config, prefix=f"{prefix}.model" if prefix else "model"
        )
        self.lm_head = (
            ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                params_dtype=torch.bfloat16,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights = []
        self.moe_layers = [
            layer.mlp.experts
            for layer in self.model.layers
            if isinstance(layer, DeepseekV4DecoderLayer)
        ]
        self.num_moe_layers = self.config.num_hidden_layers if self.moe_layers else 0
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_logical_experts = self.config.n_routed_experts
        self.num_physical_experts = self.num_logical_experts
        self.num_local_physical_experts = (
            self.moe_layers[0].routed_experts.local_num_experts
            if self.moe_layers
            else 0
        )
        self.num_routed_experts = self.config.n_routed_experts
        self.num_shared_experts = self.config.n_shared_experts or 0
        self.num_redundant_experts = 0

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def _checkpoint_name(cls, name):
        # Quantization metadata and tensor loading share the exact same mapper.
        return cls.hf_to_vllm_mapper._map_name(name)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        expert_mapping = fused_moe_make_expert_params_mapping(
            self, "gate_proj", "down_proj", "up_proj", self.config.n_routed_experts
        )
        for raw_name, tensor in weights:
            clean = raw_name.removeprefix("model.")
            if clean.startswith("mtp.") or clean.endswith("rotary_emb.inv_freq"):
                continue
            if (
                clean.startswith("layers.")
                and int(clean.split(".")[1]) >= self.config.num_hidden_layers
            ):
                continue
            name = self._checkpoint_name(raw_name)
            if is_pp_missing_parameter(name, self):
                continue
            for target, source, expert_id, shard_id in expert_mapping:
                if source not in name:
                    continue
                mapped = name.replace(source, target)
                param = params[mapped]
                success = param.weight_loader(
                    param,
                    tensor,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded.add(mapped)
                break
            else:
                shard = None
                for source, shard_id in ((".gate_proj.", 0), (".up_proj.", 1)):
                    if source in name:
                        name = name.replace(source, ".gate_up_proj.")
                        shard = shard_id
                        break
                param = params[name]
                if name.endswith(".attn_sink"):
                    tensor = tensor.chunk(
                        get_tensor_model_parallel_world_size(), dim=0
                    )[get_tensor_model_parallel_rank()]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, shard)
                loaded.add(name)
        return loaded

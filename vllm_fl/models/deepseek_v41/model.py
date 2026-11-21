# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""DeepSeek V4.1 language model with MetaX TP and DP/EP orchestration.

Adapted from vLLM e77daef89e18. Native Linear/MoE/embedding and the existing
Plugin dispatcher organize public numerical APIs; V2 owns request metadata.
"""

from itertools import islice

import torch
from torch import nn
from transformers import AutoTokenizer

from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MixtureOfExperts,
    DeepseekV4MoE,
)
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
from vllm.sequence import IntermediateTensors

from .attention import DeepseekV41Attention
from .engram import Engram, EngramHasher
from .engram_metadata import EngramLayout
from .projections import V41ClampedSiluAndMul, enable_v41_fp32_gate
from .weights import WEIGHTS_MAPPER, expert_mapping, load_text_model_weights
from vllm_fl.dispatch import resolve_op


class V41RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(width, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = eps
        self.normalize = resolve_op("mhc_rms_norm")

    def forward(self, value):
        return self.normalize(value, self.weight, self.variance_epsilon)


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self, vllm_config, *, prefix, indices, candidates, rotary_tables, layout
    ):
        super().__init__()
        c = vllm_config.model_config.hf_config
        layer = extract_layer_index(prefix)
        self.attn = DeepseekV41Attention(
            vllm_config,
            prefix=prefix + ".attn",
            indices=indices,
            candidate_blocks=candidates,
            rotary_tables=rotary_tables,
        )
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=prefix + ".ffn",
            use_sequence_parallel=False,
            num_hash_layers=0,
            n_routed_experts=(
                c.dspark_n_routed_experts if layer >= c.num_hidden_layers else None
            ),
            n_activated_experts=(
                c.dspark_num_experts_per_tok if layer >= c.num_hidden_layers else None
            ),
            image_sentinel_lo=(
                0 if layer >= c.num_hidden_layers else IMAGE_SENTINEL_BASE_ID
            ),
        )
        enable_v41_fp32_gate(self.ffn.gate)
        if self.ffn.shared_experts is not None:
            self.ffn.shared_experts.act_fn = V41ClampedSiluAndMul(c.swiglu_limit)
        self.engram = (
            Engram(
                c,
                vllm_config.quant_config,
                layout,
                layout.layer_ids.index(layer),
                vllm_config.scheduler_config.max_num_batched_tokens,
                prefix=prefix + ".engram",
                engram_config=vllm_config.engram_config,
            )
            if layer in layout.layer_ids
            else None
        )
        self.attn_norm = V41RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.ffn_norm = V41RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.eps, self.hc_eps, self.repeats = (
            c.rms_norm_eps,
            c.hc_eps,
            c.hc_sinkhorn_iters,
        )
        for kind in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", (24, 4 * c.hidden_size)),
                ("base", (24,)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{kind}_{suffix}",
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )
        self.pre = resolve_op("mhc_pre_delayed")
        self.post = resolve_op("mhc_post")

    def _pre(self, residual, kind, previous_pre):
        norm = self.attn_norm if kind == "attn" else self.ffn_norm
        return self.pre(
            residual,
            getattr(self, f"hc_{kind}_fn"),
            getattr(self, f"hc_{kind}_scale"),
            getattr(self, f"hc_{kind}_base"),
            self.eps,
            self.hc_eps,
            self.hc_eps,
            2.0,
            self.repeats,
            pre_mix=previous_pre,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
        )

    def forward(
        self, residual, previous_pre, positions, input_ids, hashes, state_inputs
    ):
        # Injection must precede the attention mix computation even for this
        # stage's first local layer; a PP boundary must not bypass Engram.
        if self.engram is not None and hashes is not None:
            from vllm.models.deepseek_v41.common.mm_preprocess import (
                image_sentinel_mask,
            )

            residual = self.engram(
                residual,
                hashes[:, self.engram.layer_hash_index],
                ~image_sentinel_mask(input_ids),
            )
        post, comb, value, attn_pre = self._pre(residual, "attn", previous_pre)
        output = self.attn(positions, value, **state_inputs)
        residual = self.post(output.contiguous(), residual, post, comb)
        post, comb, value, next_pre = self._pre(residual, "ffn", attn_pre)
        # Empty DP ranks still participate in EP dispatch/combine collectives.
        output = self.ffn(value, input_ids)
        return self.post(output.contiguous(), residual, post, comb), next_pre


class DeepseekV41Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__()
        c = vllm_config.model_config.hf_config
        if getattr(c, "num_hash_layers", 0) != 0:
            raise NotImplementedError("V4.1 text uses score routing on every layer")
        self.config = c
        if not vllm_config.use_v2_model_runner:
            raise NotImplementedError("V4.1 text requires the V2 GPU model runner")
        if c.hc_mult != 4 or c.num_hidden_layers != 40 or c.hidden_size != 5120:
            raise NotImplementedError(
                "initial V4.1 model requires 40 layers, hidden5120 and HC4"
            )
        self.layout = EngramLayout(c)
        self.rotary_tables = nn.ModuleDict()
        rows = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices",
            torch.empty((rows, c.index_topk), dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "candidate_blocks",
            torch.empty((rows, c.candidate_topk_blocks), dtype=torch.int32),
            persistent=False,
        )
        self.embed_tokens = (
            VocabParallelEmbedding(
                c.vocab_size,
                c.hidden_size,
                params_dtype=torch.bfloat16,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            c.num_hidden_layers,
            lambda prefix: DeepseekV41DecoderLayer(
                vllm_config,
                prefix=prefix,
                indices=self.topk_indices,
                candidates=self.candidate_blocks,
                rotary_tables=self.rotary_tables,
                layout=self.layout,
            ),
            prefix=prefix + ".layers",
        )
        local_engram = any(
            getattr(layer, "engram", None) is not None
            for layer in islice(self.layers, self.start_layer, self.end_layer)
        )
        self.engram_hasher = None
        if local_engram:
            tokenizer = AutoTokenizer.from_pretrained(
                vllm_config.model_config.tokenizer,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
            )
            self.engram_hasher = EngramHasher(c, tokenizer, self.layout)
        self.norm = (
            V41RMSNorm(c.hidden_size, c.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.collapse = resolve_op("mhc_collapse")
        self.aux_hidden_state_layers = ()
        self.mean_hc = resolve_op("mean_hc")
        self.engram_dp_sharded = any(
            layer.engram is not None and layer.engram.embed_tokens.dp_size > 1
            for layer in islice(self.layers, self.start_layer, self.end_layer)
        )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def reset_compressor_state(self, request_index):
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            if layer.attn.compressor is not None:
                layer.attn.compressor.reset_request(request_index)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, 4, self.config.hidden_size), device=device, dtype=dtype
                ),
                "pre_mix": torch.zeros(
                    (batch_size, 4), device=device, dtype=torch.float32
                ),
            }
        )

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        engram_history_ids=None,
        compressor_state_slots=None,
        compressor_token_to_req=None,
        compressor_query_start_loc=None,
    ):
        if get_pp_group().is_first_rank:
            embedded = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = embedded[:, None].expand(-1, 4, -1).contiguous()
            pre_mix = None
        else:
            if intermediate_tensors is None:
                raise ValueError("PP consumer requires hidden states and FP32 pre-mix")
            residual, pre_mix = (
                intermediate_tensors["hidden_states"],
                intermediate_tensors["pre_mix"],
            )
        hashes = None
        if self.engram_hasher is not None:
            if engram_history_ids is None:
                raise ValueError("V4.1 Engram requires V2 raw token-history inputs")
            hashes = self.engram_hasher(engram_history_ids)
            if self.engram_dp_sharded:
                from vllm.models.deepseek_v41.common.engram import gather_engram_hashes

                hashes = gather_engram_hashes(hashes)
        state_inputs = dict(
            compressor_state_slots=compressor_state_slots,
            compressor_token_to_req=compressor_token_to_req,
            compressor_query_start_loc=compressor_query_start_loc,
        )
        aux = []
        for index in range(self.start_layer, self.end_layer):
            layer = self.layers[index]
            if index in self.aux_hidden_state_layers:
                aux.append(self.mean_hc(residual))
            residual, pre_mix = layer(
                residual, pre_mix, positions, input_ids, hashes, state_inputs
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": residual, "pre_mix": pre_mix})
        output = self.norm(self.collapse(residual, pre_mix))
        return (output, aux) if aux else output

    def load_weights(self, weights):
        return load_text_model_weights(self, weights)

    def get_expert_mapping(self):
        return expert_mapping(self)


class DeepseekV41ForCausalLM(
    nn.Module, SupportsPP, SupportsEagle3, DeepseekV4MixtureOfExperts
):
    hf_to_vllm_mapper = WEIGHTS_MAPPER
    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }
    compressor_state_capacity = 8

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekV41Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = (
            ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                params_dtype=torch.bfloat16,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.moe_mlp_layers = [
            layer.ffn
            for layer in self.model.layers
            if isinstance(layer, DeepseekV41DecoderLayer)
        ]
        self.moe_layers = [layer.experts for layer in self.moe_mlp_layers]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.extract_moe_parameters(
            self.moe_mlp_layers[0] if self.moe_mlp_layers else None
        )

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self):
        return 3 if self.model.engram_hasher is not None else 0

    def reset_compressor_state(self, request_index):
        self.model.reset_compressor_state(request_index)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def get_mtp_target_hidden_states(self):
        # DSpark consumes explicit attention-entry aux states, not a classic
        # MTP HC-residual buffer. This matches the native V4.1 hook's None case.
        return None

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def load_weights(self, weights):
        # Checkpoint order can be model -> lm_head -> model (Engram).
        # Validate coverage once over the whole stream, not per adjacent prefix.
        from vllm_fl.quantization.modelslim import ModelSlimConfig

        if isinstance(self.quant_config, ModelSlimConfig):
            weights = self.quant_config.authoritative_weights(weights)
        return load_text_model_weights(self, self.hf_to_vllm_mapper.apply(weights))

    def get_expert_mapping(self):
        return self.model.get_expert_mapping()

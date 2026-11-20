# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Original-checkpoint DSpark stages using native 0.32 speculation lifecycle."""

import copy
import re

import torch
from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v41.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as NativeDraft,
)

from .engram_metadata import EngramLayout
from .model import DeepseekV41DecoderLayer, DeepseekV41ForCausalLM, V41RMSNorm
from .weights import load_text_model_weights
from vllm_fl.dispatch import resolve_op


class DSparkModel(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        c = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = c
        rows = vllm_config.scheduler_config.max_num_batched_tokens
        self.embed_tokens = VocabParallelEmbedding(
            c.vocab_size,
            c.hidden_size,
            params_dtype=torch.bfloat16,
            prefix=prefix + ".embed_tokens",
        )
        self.main_proj = ReplicatedLinear(
            c.hidden_size * len(c.dspark_target_layer_ids),
            c.hidden_size,
            bias=False,
            return_bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=prefix + ".main_proj",
        )
        self.main_norm = V41RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.rotary_tables = nn.ModuleDict()
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
        layout = EngramLayout(c)
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    vllm_config,
                    prefix=f"{prefix}.layers.{c.num_hidden_layers + i}",
                    indices=self.topk_indices,
                    candidates=self.candidate_blocks,
                    rotary_tables=self.rotary_tables,
                    layout=layout,
                )
                for i in range(c.num_nextn_predict_layers)
            ]
        )
        self.norm = V41RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            c.vocab_size,
            c.vocab_size,
            c.dspark_markov_rank,
            prefix=prefix + ".markov_head",
        )
        self.confidence_head = DSparkConfidenceHead(
            c.hidden_size + c.dspark_markov_rank, prefix=prefix + ".confidence_head"
        )
        self.collapse = resolve_op("mhc_collapse")
        self.normalize = resolve_op("mhc_rms_norm")

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states):
        return self.main_norm(self.main_proj(aux_hidden_states))

    def precompute_and_store_context_kv(
        self, main_x, context_positions, context_slot_mappings=None
    ):
        for i, layer in enumerate(self.layers):
            attn = layer.attn
            qkv = attn.fused_wqa_wkv(main_x)
            kv = qkv[:, attn.q_norm.weight.numel() :]
            kv = self.normalize(kv, attn.kv_norm.weight, attn.eps)
            slots = None if context_slot_mappings is None else context_slot_mappings[i]
            if slots is not None:
                empty_q = torch.empty(
                    (kv.shape[0], 0, 512), device=kv.device, dtype=kv.dtype
                )
                attn.ops["fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"](
                    empty_q,
                    kv,
                    attn.swa_cache_layer.as_page_bytes(),
                    slots,
                    context_positions,
                    attn.rotary_tables.cos_sin_cache,
                    attn.eps,
                    64,
                    apply_q_norm=False,
                )

    def forward(self, input_ids, positions, inputs_embeds=None):
        embedded = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = embedded[:, None].expand(-1, 4, -1).contiguous()
        pre_mix = None
        for layer in self.layers:
            residual, pre_mix = layer(residual, pre_mix, positions, input_ids, None, {})
        return self.collapse(residual, pre_mix)


class DSparkDeepseekV41ForCausalLM(NativeDraft):
    packed_modules_mapping = DeepseekV41ForCausalLM.packed_modules_mapping
    # This export contains separate rotated draft endpoints; do not alias them.
    has_own_embed_tokens = True
    has_own_lm_head = True

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = copy.copy(self.draft_model_config.hf_config)
        self.quant_config = vllm_config.quant_config
        self.config.n_routed_experts = self.config.dspark_n_routed_experts
        self.model = DSparkModel(vllm_config, maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.bfloat16,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def load_weights(self, weights):
        from vllm_fl.quantization.modelslim import ModelSlimConfig

        if isinstance(self.quant_config, ModelSlimConfig):
            weights = self.quant_config.authoritative_weights(weights)

        def mapped():
            for name, tensor in weights:
                if not name.startswith("mtp."):
                    continue
                _, stage, rest = name.split(".", 2)
                if rest.startswith("embed."):
                    name = "model.embed_tokens." + rest[len("embed.") :]
                elif rest.startswith("head."):
                    name = "lm_head." + rest[len("head.") :]
                elif rest.startswith(
                    (
                        "main_proj.",
                        "main_norm.",
                        "norm.",
                        "markov_head.",
                        "confidence_head.",
                    )
                ):
                    name = "model." + rest
                else:
                    name = f"model.layers.{stage}." + rest
                name = re.sub(r"\.scale$", ".weight_scale", name)
                name = name.replace(".shared_experts.w2.", ".shared_experts.down_proj.")
                name = name.replace(".markov_head.embed.", ".markov_head.markov_w1.")
                name = name.replace(".markov_head.head.", ".markov_head.markov_w2.")
                if name.endswith(".ffn.gate.bias"):
                    name = name[: -len("bias")] + "e_score_correction_bias"
                yield name, tensor

        return load_text_model_weights(self, mapped(), draft=True)

    def process_weights_after_loading(self):
        # Native parameter hooks prepare packed W4A8/W8A8 methods independently.
        pass

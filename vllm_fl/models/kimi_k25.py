# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Kimi-K2.5 model support for text-only inference.

This is a simplified implementation that wraps DeepseekV3 for text-only
benchmarking. Vision components are not included.
"""

import copy
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Model,
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class KimiK25ForConditionalGeneration(nn.Module, SupportsPP):
    """Kimi-K2.5 model for text-only conditional generation.

    This is a minimal implementation that uses DeepseekV2Model as the
    language backbone. Vision components are not included for simplicity.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        # Extract text config from KimiK25Config
        # The text_config is a DeepseekV3Config
        text_config = getattr(config, "text_config", config)
        self.hidden_size = text_config.hidden_size

        # Create a modified vllm_config with text_config as hf_config
        sub_vllm_config = copy.deepcopy(vllm_config)
        sub_vllm_config.model_config.hf_config = text_config

        # Build language model using DeepseekV2Model
        self.language_model = DeepseekV2Model(
            vllm_config=sub_vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Build lm_head
        if get_pp_group().is_last_rank:
            vocab_size = getattr(config, "vocab_size", text_config.vocab_size)
            self.lm_head = ParallelLMHead(
                vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        vocab_size = getattr(config, "vocab_size", text_config.vocab_size)
        self.logits_processor = LogitsProcessor(vocab_size, scale=logit_scale)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids."""
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, **kwargs)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Get expert parameter mapping for MoE layers."""
        text_config = getattr(self.config, "text_config", self.config)
        if not getattr(text_config, "n_routed_experts", None):
            return []
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=text_config.n_routed_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights with proper name remapping for Kimi-K2.5."""
        text_config = getattr(self.config, "text_config", self.config)

        # Weight name remapping for Kimi-K2.5 -> DeepseekV2
        _KEYS_TO_MODIFY_MAPPING = {
            "language_model.lm_head": "lm_head",
            "language_model.model": "language_model",
        }

        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if getattr(text_config, "kv_lora_rank", None) and getattr(
            text_config, "q_lora_rank", None
        ):
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}

            # Skip rotary embedding cached values
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Skip speculative decode layers
            spec_layer = get_spec_layer_idx_from_weight_name(text_config, name)
            if spec_layer is not None:
                continue

            # Skip vision tower weights (not needed for text-only inference)
            if "vision_tower" in name or "mm_projector" in name:
                continue

            # Apply key remapping
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            use_default_weight_loading = False

            # Handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id, **kwargs)
                break
            else:
                # Handle expert parameters
                for _, (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in enumerate(expert_params_mapping):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                        **kwargs,
                    )
                    break
                else:
                    use_default_weight_loading = True

            if use_default_weight_loading:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, **kwargs)

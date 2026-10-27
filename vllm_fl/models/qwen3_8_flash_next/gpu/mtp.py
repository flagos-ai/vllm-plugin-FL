# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MTP with an H-wide logits stream and real 4H feedback.

The residual-linear-shared fusion follows vLLM qwen4_exp/nvidia/mtp.py at
7b1d07440dde8426a8b3d85e5f5815c06913eb82. The decoder and checkpoint loader
use this plugin's vLLM 0.24 interfaces; no upstream source is replaced.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_next import QwenNextMixtureOfExperts
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from .model import Qwen3_8FlashNextDecoderLayer, Qwen3_8FlashNextModel
from ..common.hyperconnection import GatedResidualSimple, GroupedGemmaRMSNorm, HyperConnectionConfig


def remap_mtp_weight_name(name: str) -> str | None:
    """Keep only the draft weights and the target embedding/head it reuses."""
    for prefix in ("model.language_model.", "language_model."):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    for prefix in ("mtp.shared_head.head.", "model.shared_head.head.",
                   "shared_head.head."):
        if name.startswith(prefix):
            return "lm_head." + name.removeprefix(prefix)
    if name.startswith("mtp."):
        return "model." + name.removeprefix("mtp.")
    if name.startswith("embed_tokens."):
        return "model." + name
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith(("model.embed_tokens.", "lm_head.")):
        return name
    return None


class Qwen3_8FlashNextMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_8FlashNextModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.num_mtp_layers = config.mtp_num_hidden_layers
        if self.num_mtp_layers != 1:
            raise ValueError("This checkpoint contract requires one MTP layer")
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self._qsa_layer_ids = frozenset(range(self.num_mtp_layers))
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        linear_kwargs = dict(gather_output=True, bias=False, return_bias=False)
        self.fc_embedding = ColumnParallelLinear(
            self.hidden_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "fc_embedding"), **linear_kwargs,
        )
        self.fc_hidden = ColumnParallelLinear(
            self.hidden_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "fc_hidden"), **linear_kwargs,
        )
        self.pre_fc_norm_embedding = GroupedGemmaRMSNorm(
            self.hidden_size, config.rms_norm_eps, group_size=None,
            dtype=vllm_config.model_config.dtype, use_fused_ops=False,
        )
        # The checkpoint normalizes the entire 4H stream before shared H->H FC.
        self.pre_fc_norm_hidden = GroupedGemmaRMSNorm(
            self.hc_count * self.hidden_size, config.rms_norm_eps, group_size=None,
            dtype=vllm_config.model_config.dtype, use_fused_ops=False,
        )
        self.layers = nn.ModuleList([
            Qwen3_8FlashNextDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{config.num_hidden_layers + layer_idx}",
                force_disable_ple=True,
            )
            for layer_idx in range(self.num_mtp_layers)
        ])
        self.hyper_connection_mixer = GatedResidualSimple(
            HyperConnectionConfig(
                hc_count=self.hc_count,
                hidden_size=self.hidden_size,
                params_dtype=vllm_config.model_config.dtype,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
                use_fused_ops=False,
            ),
            use_combine=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError("MTP feedback must contain the target's real 4H stream")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("MTP requires token IDs or their embeddings")
            inputs_embeds = self.embed_input_ids(input_ids)
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        hidden = self.pre_fc_norm_hidden(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        hidden = self.fc_hidden(hidden) + embedding.unsqueeze(-2)
        hidden, _, _ = self.layers[spec_step_idx % self.num_mtp_layers](
            hidden_states=hidden.flatten(-2), residual=None, positions=positions,
        )
        sample_hidden, _ = self.hyper_connection_mixer.mix(hidden)
        return sample_hidden, hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Reuse the 0.24 QKV/shared-expert/fused routed-expert mapping once.
        return Qwen3_8FlashNextModel.load_weights(self, weights)


class Qwen3_8FlashNextMTP(nn.Module, QwenNextMixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Qwen3.8 MTP currently requires PP=1")
        if vllm_config.cache_config.mamba_cache_mode != "none":
            raise NotImplementedError(
                "Qwen3.8 MTP currently requires mamba_cache_mode=none"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8 MTP requires the BF16 checkpoint")
        if vllm_config.quant_config is not None:
            raise NotImplementedError(
                "Qwen3.8 MTP weight quantization is not validated"
            )
        self.config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.quant_config = None
        self.model = Qwen3_8FlashNextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        if (getattr(self.config, "num_experts", 0) or 0) > 0:
            QwenNextMixtureOfExperts.set_moe_parameters(self)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intermediate_tensors is not None:
            raise ValueError("PP intermediate tensors are unsupported for Qwen3.8 MTP")
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx,
        )

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remapped():
            for name, weight in weights:
                mapped = remap_mtp_weight_name(name)
                if mapped is not None:
                    yield mapped, weight
        return AutoWeightsLoader(self).load_weights(remapped())


Qwen4ExpMTP = Qwen3_8FlashNextMTP

# Copyright (c) 2026 BAAI. All rights reserved.

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.models.bert import BertModel
from vllm.model_executor.models.roberta import BgeM3EmbeddingModel, RobertaEmbedding


def packed_roberta_positions(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    padding_idx: int,
) -> torch.Tensor:
    """Return zero-based positions for RoBERTa's packed embedding input.

    A literal PAD token remains in a request even after batch padding is removed.
    Hugging Face skips that ID in the position count and gives it padding_idx.
    The caller adds padding_idx + 1, so PAD itself must receive -1 here.
    """
    is_pad = input_ids == padding_idx
    pad_prefix = torch.cumsum(is_pad, dim=0, dtype=positions.dtype)
    token_indices = torch.arange(input_ids.numel(), device=input_ids.device)
    starts = (token_indices - positions).clamp_min(0)
    pads_before_request = pad_prefix[starts] - is_pad[starts].to(positions.dtype)
    corrected = positions - (pad_prefix - pads_before_request)
    return torch.where(is_pad, -1, corrected)


class BgeM3Embedding(RobertaEmbedding):
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = packed_roberta_positions(input_ids, position_ids, self.padding_idx)
        return super().forward(input_ids, positions, inputs_embeds)


class BgeM3EmbeddingModelFL(BgeM3EmbeddingModel):
    """BGE-M3 pooling with Hugging Face PAD position semantics."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        from vllm_fl.models.bge_m3.linear import bind_linear_policy

        bind_linear_policy(self)

    def _build_pooler(self, pooler_config):
        from vllm.model_executor.layers.pooler import DispatchPooler

        from vllm_fl.models.bge_m3.pooling import BgeM3JointPooler

        native = super()._build_pooler(pooler_config)
        if get_current_vllm_config().model_config.io_processor_plugin != "bge_m3":
            return native
        poolers = dict(native.poolers_by_task)
        poolers["plugin"] = BgeM3JointPooler(poolers)
        return DispatchPooler(poolers)

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "position_embedding_type", "absolute") != "absolute":
            return super()._build_model(vllm_config, prefix)
        return BertModel(
            vllm_config=vllm_config, prefix=prefix, embedding_class=BgeM3Embedding
        )

# SPDX-License-Identifier: Apache-2.0
# Migrated from upstream vLLM (latest) to vllm-plugin-FL (v0.13.0 compat)

import itertools
from collections.abc import Iterable

import torch
from torch import nn
from transformers import RobertaConfig

from vllm.config import ModelConfig, PoolerConfig, VllmConfig
from vllm.model_executor.layers.pooler import (
    ClassifierPooler,
    CLSPool,
    DispatchPooler,
    Pooler,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bert import (
    TOKEN_TYPE_SHIFT,
    BertEmbeddingModel,
    BertModel,
    _decode_token_type_ids,
    _encode_token_type_ids,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.bert_with_rope import BertWithRope, JinaRobertaModel
from vllm.model_executor.models.interfaces import SupportsCrossEncoding
from vllm.model_executor.models.interfaces_base import default_pooling_type


class RobertaEmbedding(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_type_ids = _decode_token_type_ids(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


# Adapted from transformers
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, model_config: "ModelConfig"):
        super().__init__()
        config = model_config.hf_config
        head_dtype = model_config.head_dtype
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, dtype=head_dtype
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.num_labels, dtype=head_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token extraction has already been applied in `pooler.pooling`
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


# v0.13.0 compat: default_pooling_type uses positional arg `pooling_type`
@default_pooling_type("CLS")
class RobertaEmbeddingModel(BertEmbeddingModel):
    """A model that uses Roberta to provide embedding functionalities."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.padding_idx: int = vllm_config.model_config.hf_config.pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        replace_roberta_positions(
            input_ids=input_ids, position_ids=positions, padding_idx=self.padding_idx
        )

        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def _build_model(
        self, vllm_config: VllmConfig, prefix: str = ""
    ) -> BertModel | BertWithRope:
        hf_config = vllm_config.model_config.hf_config
        kwargs = dict(vllm_config=vllm_config, prefix=prefix)
        if getattr(hf_config, "position_embedding_type", "absolute") == "absolute":
            return BertModel(**kwargs, embedding_class=RobertaEmbedding)
        else:
            return JinaRobertaModel(**kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)
        has_roberta_prefix = any(
            name.startswith("roberta.") for name, _ in weights_list
        )
        if has_roberta_prefix:
            mapper = WeightsMapper(orig_to_new_prefix={"roberta.": "model."})
        else:
            mapper = WeightsMapper(orig_to_new_prefix={"": "model."})

        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head.", "classifier."])
        return loader.load_weights(weights_list, mapper=mapper)


def filter_secondary_weights(
    all_weights: Iterable[tuple[str, torch.Tensor]],
    secondary_weights: list[str],
) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str, torch.Tensor]]]:
    all_weights1, all_weights2 = itertools.tee(all_weights)

    def filtered(n):
        return any(n.startswith(f) for f in secondary_weights)

    return ((n, w) for n, w in all_weights1 if filtered(n)), (
        (n, w) for n, w in all_weights2 if not filtered(n)
    )


class BgeM3EmbeddingModel(RobertaEmbeddingModel):
    """A model that extends RobertaEmbeddingModel with sparse embeddings.

    This class supports loading an additional sparse_linear.pt file
    to create sparse embeddings as described in https://arxiv.org/abs/2402.03216
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size

        model_config = vllm_config.model_config
        self.head_dtype = model_config.head_dtype
        self.bos_token_id = model_config.hf_config.bos_token_id
        self.eos_token_id = model_config.hf_config.eos_token_id

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.secondary_weight_prefixes = ["sparse_linear.", "colbert_linear."]
        self.secondary_weight_files = [
            prefix + "pt" for prefix in self.secondary_weight_prefixes
        ]

        import os
        model_path = vllm_config.model_config.model
        self.secondary_weights = []
        for filename, wprefix in zip(
            self.secondary_weight_files, self.secondary_weight_prefixes
        ):
            if os.path.isdir(model_path) and os.path.exists(
                os.path.join(model_path, filename)
            ):
                self.secondary_weights.append(
                    DefaultModelLoader.Source(
                        model_or_path=model_path,
                        revision=None,
                        prefix=wprefix,
                        allow_patterns_overrides=[filename],
                    )
                )

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        self.sparse_linear = nn.Linear(
            self.hidden_size, 1, dtype=self.head_dtype
        )
        self.colbert_linear = nn.Linear(
            self.hidden_size, self.hidden_size, dtype=self.head_dtype
        )

        # v0.13.0 compat: use Pooler.for_token_embed / for_token_classify / for_embed
        # instead of pooler_for_embed / pooler_for_token_embed / pooler_for_token_classify
        # Note: BOSEOSFilter not available in v0.13.0, using basic dispatch instead
        return DispatchPooler(
            {
                "embed": Pooler.for_embed(pooler_config),
                "token_embed": Pooler.for_token_embed(pooler_config),
                "token_classify": Pooler.for_token_classify(
                    pooler_config,
                    classifier=self.sparse_linear,
                    act_fn=torch.relu,
                ),
            }
        )

    def load_weights(self, all_weights: Iterable[tuple[str, torch.Tensor]]):
        secondary, weights = filter_secondary_weights(
            all_weights, self.secondary_weight_prefixes
        )

        super().load_weights(weights)

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in secondary:
            if any(
                name.startswith(prefix) for prefix in self.secondary_weight_prefixes
            ):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# v0.13.0 compat: default_pooling_type uses positional arg `pooling_type`
@default_pooling_type("CLS")
class RobertaForSequenceClassification(nn.Module, SupportsCrossEncoding):
    """
    This class encapsulates the BertModel and provides an interface for
    sequence classification.
    """

    jina_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "roberta.": "roberta.",
            "classifier.": "classifier.",
        },
        orig_to_new_substr={
            ".self_attn.query.": ".self_attn.qkv_proj.",
            ".self_attn.key.": ".self_attn.qkv_proj.",
            ".self_attn.value.": ".self_attn.qkv_proj.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx: int = vllm_config.model_config.hf_config.pad_token_id

        self.num_labels = config.num_labels
        self.roberta = BertModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "bert"),
            embedding_class=RobertaEmbedding,
        )
        self.classifier = RobertaClassificationHead(vllm_config.model_config)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        # v0.13.0 compat: no DispatchPooler.for_seq_cls(), use explicit construction
        self.pooler = DispatchPooler(
            {
                "classify": ClassifierPooler(
                    pooling=CLSPool(),
                    classifier=self.classifier,
                    act_fn="classify",
                ),
                "score": ClassifierPooler(
                    pooling=CLSPool(),
                    classifier=self.classifier,
                    act_fn="score",
                ),
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.jina_to_vllm_mapper)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.roberta.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        replace_roberta_positions(
            input_ids=input_ids, position_ids=positions, padding_idx=self.padding_idx
        )
        if token_type_ids is not None:
            assert self.roberta.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
            assert input_ids is not None
            _encode_token_type_ids(input_ids, token_type_ids)
        return self.roberta(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )


def replace_roberta_positions(
    input_ids: torch.Tensor, position_ids: torch.Tensor, padding_idx: int
) -> None:
    position_ids += padding_idx + 1

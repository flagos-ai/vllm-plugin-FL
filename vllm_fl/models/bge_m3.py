import itertools
from collections.abc import Iterable, Set

import torch
from torch import nn

from vllm.config import PoolerConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.pooler import (
    AllPooler,
    DispatchPooler,
    Pooler,
    PoolerNormalize,
    PoolingParamsUpdate,
    StepPooler,
)
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.adapters import _load_st_projector
from vllm.model_executor.models.roberta import RobertaEmbeddingModel
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


def filter_secondary_weights(
    all_weights: Iterable[tuple[str, torch.Tensor]],
    secondary_weight_prefixes: list[str],
) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str, torch.Tensor]]]:
    all_weights_1, all_weights_2 = itertools.tee(all_weights)

    def is_secondary(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in secondary_weight_prefixes)

    secondary = (
        (name, weight) for name, weight in all_weights_1 if is_secondary(name)
    )
    primary = (
        (name, weight) for name, weight in all_weights_2 if not is_secondary(name)
    )
    return secondary, primary


class TokenEmbeddingProjectionHead(nn.Module):
    def __init__(self, projector: nn.Module | None) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        assert vllm_config is not None

        self.projector = _load_st_projector(vllm_config.model_config)
        self.token_projector = projector
        self.activation = PoolerNormalize()
        self.head_dtype = vllm_config.model_config.head_dtype

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed"}

    def forward(
        self,
        pooled_data: torch.Tensor | None,
        pooling_param: PoolingParams,
    ) -> PoolerOutput:
        if pooled_data is None:
            return None

        pooled_data = pooled_data.to(self.head_dtype)

        if self.projector is not None:
            pooled_data = self.projector(pooled_data)

        if self.token_projector is not None:
            pooled_data = self.token_projector(pooled_data)

        pooled_data = pooled_data[..., : pooling_param.dimensions]

        if pooling_param.normalize:
            pooled_data = self.activation(pooled_data)

        return pooled_data


class SpecialTokenFilterPooler(Pooler):
    def __init__(self, pooler: Pooler, token_ids_to_skip: list[int | None]) -> None:
        super().__init__()
        self.pooler = pooler
        self.token_ids_to_skip = tuple(
            token_id for token_id in token_ids_to_skip if token_id is not None
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooler.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def _filter_one(
        self,
        data: torch.Tensor | None,
        token_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if data is None:
            return None

        keep_mask = torch.ones_like(token_ids, dtype=torch.bool)
        for token_id in self.token_ids_to_skip:
            keep_mask &= token_ids != token_id
        return data[keep_mask]

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        outputs = self.pooler(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids()
        return [
            self._filter_one(output, token_ids)
            for output, token_ids in zip(outputs, prompt_token_ids)
        ]


class BgeM3EmbeddingModel(RobertaEmbeddingModel):
    """Backport the vLLM 0.15 BGE-M3 embedding adapter to vLLM 0.13.x."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size

        model_config = vllm_config.model_config
        self.head_dtype = model_config.head_dtype
        self.bos_token_id = model_config.hf_config.bos_token_id
        self.eos_token_id = model_config.hf_config.eos_token_id

        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.secondary_weight_prefixes = ["sparse_linear.", "colbert_linear."]
        self.secondary_weight_files = [
            weight_prefix + "pt" for weight_prefix in self.secondary_weight_prefixes
        ]
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=vllm_config.model_config.model,
                revision=None,
                prefix=weight_prefix,
                allow_patterns_overrides=[filename],
            )
            for filename, weight_prefix in zip(
                self.secondary_weight_files,
                self.secondary_weight_prefixes,
            )
        ]

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        self.sparse_linear = nn.Linear(self.hidden_size, 1, dtype=self.head_dtype)
        self.colbert_linear = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            dtype=self.head_dtype,
        )

        token_embed_pooler = Pooler.for_token_embed(pooler_config)
        if isinstance(token_embed_pooler, AllPooler):
            token_embed_pooler = AllPooler(
                head=TokenEmbeddingProjectionHead(self.colbert_linear)
            )
        elif isinstance(token_embed_pooler, StepPooler):
            token_embed_pooler = StepPooler(
                head=TokenEmbeddingProjectionHead(self.colbert_linear)
            )
        else:
            raise TypeError(
                f"Unsupported token_embed pooler: {type(token_embed_pooler)}"
            )

        token_classify_pooler = Pooler.for_token_classify(
            pooler_config,
            classifier=self.sparse_linear,
            act_fn=torch.relu,
        )

        return DispatchPooler(
            {
                "embed": Pooler.for_embed(pooler_config),
                "token_embed": SpecialTokenFilterPooler(
                    token_embed_pooler,
                    [self.bos_token_id],
                ),
                "token_classify": SpecialTokenFilterPooler(
                    token_classify_pooler,
                    [self.bos_token_id, self.eos_token_id],
                ),
            }
        )

    def load_weights(self, all_weights: Iterable[tuple[str, torch.Tensor]]):
        secondary_weights, primary_weights = filter_secondary_weights(
            all_weights,
            self.secondary_weight_prefixes,
        )

        super().load_weights(primary_weights)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in secondary_weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

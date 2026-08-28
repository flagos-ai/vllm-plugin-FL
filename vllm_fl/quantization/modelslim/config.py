# Copyright 2026 FlagOS Contributors
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ModelSlim descriptor parsing and vLLM quantization registration.

The implementation follows the ModelSlim contract used by vLLM-Ascend
v0.20.2rc1, while keeping model-specific name mapping and NPU kernels outside
the descriptor layer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.model_executor.models.utils import WeightsMapper

ASCEND_QUANTIZATION_METHOD = "ascend"
MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"

_SUPPORTED_LINEAR_QUANT_TYPES = frozenset({"FLOAT", "W8A8", "W8A8_DYNAMIC"})
_PREFIX_MAPPERS: dict[str, Callable[[str], str]] = {}

logger = init_logger(__name__)


def register_modelslim_prefix_mapper(
    model_type: str,
    mapper: Callable[[str], str],
    *,
    replace: bool = False,
) -> None:
    """Register a model-owned descriptor-prefix adapter.

    The generic quantization implementation does not contain model-name
    branches. Models whose checkpoint names cannot be expressed by vLLM's
    ``WeightsMapper`` can register one focused adapter during model setup.
    """

    if not model_type:
        raise ValueError("model_type must be non-empty")
    if model_type in _PREFIX_MAPPERS and not replace:
        raise ValueError(f"ModelSlim prefix mapper already registered: {model_type}")
    _PREFIX_MAPPERS[model_type] = mapper


def _normalize_quant_description(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("ModelSlim quantization config must be a mapping")

    normalized = dict(config)
    for key, value in tuple(config.items()):
        if not isinstance(key, str):
            raise TypeError("ModelSlim quantization config keys must be strings")
        if "weight_packed" not in key:
            continue
        alias = key.replace("weight_packed", "weight")
        previous = normalized.get(alias, value)
        if previous != value:
            raise ValueError(
                f"Conflicting ModelSlim descriptor aliases: {key!r} and {alias!r}"
            )
        normalized[alias] = value
    return normalized


def _packed_shard_prefixes(
    prefix: str,
    packed_modules_mapping: Mapping[str, list[str]],
) -> list[str]:
    parent, separator, projection = prefix.rpartition(".")
    shards = packed_modules_mapping.get(projection)
    if not shards:
        return [prefix]
    base = f"{parent}{separator}" if separator else ""
    return [f"{base}{shard}" for shard in shards]


def resolve_linear_quant_type(
    quant_description: Mapping[str, Any],
    prefix: str,
    packed_modules_mapping: Mapping[str, list[str]] | None = None,
) -> str:
    """Resolve one linear/MoE prefix to a single ModelSlim quant type."""

    mapping = packed_modules_mapping or {}
    shard_prefixes = _packed_shard_prefixes(prefix, mapping)
    resolved: list[tuple[str, str]] = []
    for shard_prefix in shard_prefixes:
        key = f"{shard_prefix}.weight"
        if key not in quant_description:
            raise ValueError(
                f"ModelSlim descriptor has no quantization entry for {key!r}"
            )
        quant_type = quant_description[key]
        if not isinstance(quant_type, str):
            raise TypeError(f"ModelSlim descriptor value for {key!r} must be a string")
        resolved.append((key, quant_type.upper()))

    quant_types = {quant_type for _, quant_type in resolved}
    if len(quant_types) != 1:
        details = ", ".join(f"{key}={value}" for key, value in resolved)
        raise ValueError(
            f"Packed layer {prefix!r} mixes ModelSlim quantization types: {details}"
        )

    quant_type = resolved[0][1]
    if quant_type not in _SUPPORTED_LINEAR_QUANT_TYPES:
        raise NotImplementedError(
            f"ModelSlim quantization type {quant_type!r} is not supported for "
            f"linear layer {prefix!r}"
        )
    return quant_type


def _get_model_file(
    model: str | Path,
    filename: str,
    revision: str | None = None,
) -> Path | None:
    model_path = Path(model)
    if model_path.exists():
        candidate = model_path / filename
        return candidate if candidate.is_file() else None

    try:
        from vllm import envs

        if envs.VLLM_USE_MODELSCOPE:
            from modelscope.hub.file_download import model_file_download

            return Path(
                model_file_download(
                    model_id=str(model),
                    file_path=filename,
                    revision=revision,
                )
            )

        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=str(model),
                filename=filename,
                revision=revision,
            )
        )
    except Exception as exc:
        logger.warning(
            "Could not resolve ModelSlim descriptor %s from %s: %s",
            filename,
            model,
            exc,
        )
        return None


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
class AscendModelSlimConfig(QuantizationConfig):
    """Configuration adapter for ModelSlim ``quant_model_description.json``."""

    def __init__(self, quant_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.quant_description = _normalize_quant_description(quant_config or {})
        self.model_type: str | None = None
        self._applied_mapper_ids: set[int] = set()

    def __repr__(self) -> str:
        return f"AscendModelSlimConfig(entries={len(self.quant_description)})"

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # Loading in maybe_update_config gives a precise error and also works
        # with ModelScope/Hugging Face repositories.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendModelSlimConfig":
        return cls(config)

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        if hf_config is not None:
            self.model_type = getattr(hf_config, "model_type", None)
        if self.quant_description:
            return

        config_path = _get_model_file(
            model_name,
            MODELSLIM_CONFIG_FILENAME,
            revision=revision,
        )
        if config_path is None:
            raise ValueError(
                f"--quantization {ASCEND_QUANTIZATION_METHOD} requires "
                f"{MODELSLIM_CONFIG_FILENAME!r} in model {model_name!r}"
            )
        try:
            with config_path.open(encoding="utf-8") as descriptor_file:
                descriptor = json.load(descriptor_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Failed to load ModelSlim descriptor {str(config_path)!r}: {exc}"
            ) from exc
        self.quant_description = _normalize_quant_description(descriptor)
        if not self.quant_description:
            raise ValueError(f"ModelSlim descriptor {str(config_path)!r} is empty")

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        mapper_id = id(hf_to_vllm_mapper)
        if mapper_id in self._applied_mapper_ids:
            return
        mapped = hf_to_vllm_mapper.apply_dict(self.quant_description)
        if len(mapped) != len(self.quant_description):
            raise ValueError(
                "vLLM weight mapping produced colliding ModelSlim descriptor keys"
            )
        self.quant_description = _normalize_quant_description(mapped)
        self._applied_mapper_ids.add(mapper_id)

    def _map_prefix(self, prefix: str) -> str:
        if self.model_type is None:
            return prefix
        mapper = _PREFIX_MAPPERS.get(self.model_type)
        return mapper(prefix) if mapper is not None else prefix

    def _get_linear_quant_type(self, prefix: str) -> str:
        return resolve_linear_quant_type(
            self.quant_description,
            self._map_prefix(prefix),
            self.packed_modules_mapping,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
            VocabParallelEmbedding,
        )

        if isinstance(layer, VocabParallelEmbedding):
            quant_type = self._get_linear_quant_type(prefix)
            if quant_type == "FLOAT":
                return UnquantizedEmbeddingMethod()
            raise NotImplementedError(
                f"ModelSlim quantized embedding is not supported: {prefix!r}"
            )

        if isinstance(layer, LinearBase):
            quant_type = self._get_linear_quant_type(prefix)
            if quant_type == "FLOAT":
                return UnquantizedLinearMethod()

            from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import (
                AscendModelSlimLinearMethod,
                get_w8a8_linear_scheme,
            )

            return AscendModelSlimLinearMethod(
                get_w8a8_linear_scheme(quant_type)
            )

        if isinstance(layer, FusedMoE):
            quant_type = self._get_linear_quant_type(prefix)
            if quant_type == "FLOAT":
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if quant_type == "W8A8_DYNAMIC":
                from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import (
                    AscendModelSlimW8A8DynamicMoEMethod,
                )

                return AscendModelSlimW8A8DynamicMoEMethod(layer.moe_config)
            raise NotImplementedError(
                f"ModelSlim {quant_type} MoE is not supported for {prefix!r}"
            )

        return None

"""ModelSlim dynamic W8A8 checkpoint support for FL backends."""

from __future__ import annotations

from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.parameter import ChannelQuantScaleParameter
from vllm.model_executor.utils import set_weight_attrs


class _ModelSlimW8A8LinearScheme(CompressedTensorsW8A8Int8):
    """Retain ModelSlim's symmetric offset tensor while using FL scaled-mm."""

    def create_weights(self, layer: torch.nn.Module, **kwargs: Any) -> None:
        super().create_weights(layer, **kwargs)
        weight_loader = kwargs["weight_loader"]
        output_size = sum(kwargs["output_partition_sizes"])
        weight_offset = ChannelQuantScaleParameter(
            data=torch.empty((output_size, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_offset", weight_offset)


class _ModelSlimW8A8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):
    """Add ModelSlim offset parameters to the standard dynamic INT8 MoE."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        attrs = dict(extra_weight_attrs)
        attrs["quant_method"] = FusedMoeWeightScaleSupported.CHANNEL.value
        w13_offset = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_offset = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_offset)
        layer.register_parameter("w2_weight_offset", w2_offset)
        set_weight_attrs(w13_offset, attrs)
        set_weight_attrs(w2_offset, attrs)


@register_quantization_config("fl_modelslim_w8a8")
class ModelSlimW8A8Config(QuantizationConfig):
    """Read ``quant_model_description.json`` produced by ModelSlim."""

    def __init__(self, description: dict[str, Any]) -> None:
        super().__init__()
        self.description = description
        self.weight_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        )
        self.input_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        )

    @classmethod
    def get_name(cls) -> str:
        return "fl_modelslim_w8a8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelSlimW8A8Config:
        return cls(config)

    def _checkpoint_prefixes(self, prefix: str) -> list[str]:
        if prefix.startswith("model."):
            prefix = prefix[len("model.") :]
        if prefix.endswith(".attn.fused_wqa_wkv"):
            base = prefix.removesuffix(".fused_wqa_wkv")
            return [f"{base}.wq_a", f"{base}.wkv"]
        if prefix.endswith(".experts"):
            return [f"{prefix}.0.w1", f"{prefix}.0.w2", f"{prefix}.0.w3"]
        if prefix.endswith(".gate_up_proj"):
            base = prefix.removesuffix(".gate_up_proj")
            return [f"{base}.w1", f"{base}.w3"]
        if prefix.endswith(".down_proj"):
            return [f"{prefix.removesuffix('.down_proj')}.w2"]
        return [prefix]

    def _is_dynamic_w8a8(self, prefix: str) -> bool:
        kinds = {
            self.description.get(f"{candidate}.weight")
            for candidate in self._checkpoint_prefixes(prefix)
        }
        kinds.discard(None)
        if not kinds:
            return False
        if len(kinds) != 1:
            raise ValueError(f"ModelSlim packed layer {prefix!r} mixes {sorted(kinds)}")
        return kinds == {"W8A8_DYNAMIC"}

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if not self._is_dynamic_w8a8(prefix):
                return UnquantizedLinearMethod()
            layer.scheme = _ModelSlimW8A8LinearScheme(
                strategy=QuantizationStrategy.CHANNEL,
                is_static_input_scheme=False,
                input_symmetric=True,
            )
            return CompressedTensorsLinearMethod(self)  # type: ignore[arg-type]
        if isinstance(layer, RoutedExperts):
            if not self._is_dynamic_w8a8(prefix):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return _ModelSlimW8A8MoEMethod(
                self.weight_quant,
                self.input_quant,
                layer.moe_config,
                layer_name=prefix,
            )
        return None


__all__ = ["ModelSlimW8A8Config"]

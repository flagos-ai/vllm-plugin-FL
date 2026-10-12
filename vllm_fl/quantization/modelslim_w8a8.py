"""ModelSlim W8A8 checkpoint support for FL backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
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
    CompressedTensorsScheme,
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
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


class _ModelSlimW8A8StaticLinearScheme(CompressedTensorsScheme):
    """Run ModelSlim's static asymmetric W8A8 linear checkpoint format."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs: Any,
    ) -> None:
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "ModelSlim static W8A8 currently requires bfloat16 activations; "
                f"got {params_dtype}."
            )
        output_size = sum(output_partition_sizes)
        num_partitions = len(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        input_scale = PerTensorScaleParameter(
            data=torch.empty(num_partitions, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        input_offset = PerTensorScaleParameter(
            data=torch.empty(num_partitions, dtype=torch.int8),
            weight_loader=weight_loader,
        )
        deq_scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size, dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        quant_bias = ChannelQuantScaleParameter(
            data=torch.empty(output_size, dtype=torch.int32),
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("input_scale", input_scale)
        layer.register_parameter("input_offset", input_offset)
        layer.register_parameter("deq_scale", deq_scale)
        layer.register_parameter("quant_bias", quant_bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_size = layer.weight.data.shape[1]
        input_scale = layer.input_scale.data.to(layer.params_dtype)
        input_offset = layer.input_offset.data.to(layer.params_dtype)
        if input_scale.numel() != len(layer.logical_widths):
            raise ValueError(
                "ModelSlim static W8A8 requires one input scale per logical "
                f"matrix, got {input_scale.numel()} for {layer.logical_widths}"
            )

        layer.aclnn_input_scale_reciprocal = torch.nn.Parameter(
            input_scale.reciprocal().view(-1, 1).repeat(1, input_size),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            input_offset.view(-1, 1).repeat(1, input_size),
            requires_grad=False,
        )
        # npu_quant_matmul consumes [K, N] weights for the ND layout.
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()

    @staticmethod
    def _quantize(
        x: torch.Tensor,
        input_scale_reciprocal: torch.Tensor,
        input_offset: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_quantize(
            x,
            input_scale_reciprocal,
            input_offset,
            torch.qint8,
            -1,
            False,
        )

    @staticmethod
    def _quant_matmul(
        x: torch.Tensor,
        weight: torch.Tensor,
        deq_scale: torch.Tensor,
        quant_bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_quant_matmul(
            x,
            weight,
            deq_scale,
            bias=quant_bias,
            output_dtype=output_dtype,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.dtype == torch.int8 and len(layer.logical_widths) != 1:
            raise ValueError(
                "Pre-quantized ModelSlim input is only supported for a single "
                "logical matrix"
            )

        is_row_parallel = isinstance(layer, RowParallelLinear)
        tp_rank = getattr(layer, "tp_rank", None) if is_row_parallel else 0
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        outputs: list[torch.Tensor] = []
        output_offset = 0

        for partition, output_width in enumerate(layer.logical_widths):
            quant_x = x
            if quant_x.dtype != torch.int8:
                quant_x = self._quantize(
                    quant_x,
                    layer.aclnn_input_scale_reciprocal[partition],
                    layer.aclnn_input_offset[partition],
                )

            output_slice = slice(output_offset, output_offset + output_width)
            quant_bias = layer.quant_bias[output_slice] if tp_rank == 0 else None
            output = self._quant_matmul(
                quant_x,
                layer.weight[:, output_slice].contiguous(),
                layer.deq_scale[output_slice],
                quant_bias,
                layer.params_dtype,
            )
            if bias is not None and tp_rank == 0:
                output = output + bias[output_slice]
            outputs.append(output)
            output_offset += output_width

        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


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

    @staticmethod
    def _root_prefixes(prefix: str) -> list[str]:
        if prefix.startswith("model."):
            return [prefix, prefix.removeprefix("model.")]
        return [prefix, f"model.{prefix}"]

    def _checkpoint_prefix_groups(self, prefix: str) -> list[tuple[str, ...]]:
        groups: list[tuple[str, ...]] = []
        for rooted_prefix in self._root_prefixes(prefix):
            if rooted_prefix.endswith(".attn.fused_wqa_wkv"):
                base = rooted_prefix.removesuffix(".fused_wqa_wkv")
                groups.append((f"{base}.wq_a", f"{base}.wkv"))
            elif rooted_prefix.endswith(".fused_qkv_a_proj"):
                base = rooted_prefix.removesuffix(".fused_qkv_a_proj")
                groups.append((f"{base}.q_a_proj", f"{base}.kv_a_proj_with_mqa"))
            elif rooted_prefix.endswith(".experts"):
                groups.extend(
                    (
                        (
                            f"{rooted_prefix}.0.w1",
                            f"{rooted_prefix}.0.w2",
                            f"{rooted_prefix}.0.w3",
                        ),
                        (
                            f"{rooted_prefix}.0.gate_proj",
                            f"{rooted_prefix}.0.up_proj",
                            f"{rooted_prefix}.0.down_proj",
                        ),
                    )
                )
            elif rooted_prefix.endswith(".gate_up_proj"):
                base = rooted_prefix.removesuffix(".gate_up_proj")
                groups.extend(
                    (
                        (f"{base}.w1", f"{base}.w3"),
                        (f"{base}.gate_proj", f"{base}.up_proj"),
                    )
                )
            elif rooted_prefix.endswith(".down_proj"):
                base = rooted_prefix.removesuffix(".down_proj")
                groups.extend(((rooted_prefix,), (f"{base}.w2",)))
            else:
                groups.append((rooted_prefix,))
        return list(dict.fromkeys(groups))

    def _quant_kind(self, prefix: str) -> str | None:
        kinds: set[str] = set()
        for group in self._checkpoint_prefix_groups(prefix):
            group_kinds = [
                self.description.get(f"{candidate}.weight") for candidate in group
            ]
            present = [kind for kind in group_kinds if kind is not None]
            if not present:
                continue
            if len(present) != len(group):
                missing = [
                    candidate
                    for candidate, kind in zip(group, group_kinds, strict=True)
                    if kind is None
                ]
                raise ValueError(
                    f"ModelSlim packed layer {prefix!r} is missing quantization "
                    f"entries for {missing}"
                )
            kinds.update(present)
        if not kinds:
            return None
        if len(kinds) != 1:
            raise ValueError(f"ModelSlim packed layer {prefix!r} mixes {sorted(kinds)}")
        kind = kinds.pop()
        if kind not in {"FLOAT", "W8A8", "W8A8_DYNAMIC"}:
            raise NotImplementedError(
                f"ModelSlim quantization {kind!r} is not supported for {prefix!r}"
            )
        return kind

    def _is_dynamic_w8a8(self, prefix: str) -> bool:
        return self._quant_kind(prefix) == "W8A8_DYNAMIC"

    def _is_static_w8a8(self, prefix: str) -> bool:
        return self._quant_kind(prefix) == "W8A8"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            kind = self._quant_kind(prefix)
            if kind in {None, "FLOAT"}:
                return UnquantizedLinearMethod()
            if kind == "W8A8":
                layer.scheme = _ModelSlimW8A8StaticLinearScheme()
            else:
                layer.scheme = _ModelSlimW8A8LinearScheme(
                    strategy=QuantizationStrategy.CHANNEL,
                    is_static_input_scheme=False,
                    input_symmetric=True,
                )
            return CompressedTensorsLinearMethod(self)  # type: ignore[arg-type]
        if isinstance(layer, RoutedExperts):
            kind = self._quant_kind(prefix)
            if kind in {None, "FLOAT"}:
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if kind == "W8A8":
                raise NotImplementedError(
                    "ModelSlim static W8A8 routed experts are not supported"
                )
            return _ModelSlimW8A8MoEMethod(
                self.weight_quant,
                self.input_quant,
                layer.moe_config,
                layer_name=prefix,
            )
        return None


__all__ = ["ModelSlimW8A8Config"]

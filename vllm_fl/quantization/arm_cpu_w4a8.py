# Copyright (c) 2026 BAAI. All rights reserved.

"""Packed W4A8 checkpoint adapter for vLLM's ARM CPU linear kernels."""

from __future__ import annotations

import threading

import torch

_INSTALL_LOCK = threading.Lock()


def install_arm_cpu_packed_w4a8() -> bool:
    """Teach vLLM 0.24 to retain packed W4 G128 checkpoint parameters."""
    with _INSTALL_LOCK:
        from compressed_tensors.compressors.pack_quantized.helpers import (
            unpack_from_int32,
        )
        from compressed_tensors.config import CompressionFormat

        from vllm.model_executor.kernels.linear import (
            MPLinearLayerConfig,
            choose_mp_linear_kernel,
        )
        from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
            Dynamic4bitLinearKernel,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors import (
            compressed_tensors as config_module,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w4a8_int as scheme_module,
        )
        from vllm.model_executor.layers.quantization.utils import replace_parameter
        from vllm.model_executor.parameter import (
            BasevLLMParameter,
            ChannelQuantScaleParameter,
            GroupQuantScaleParameter,
            PackedvLLMParameter,
        )

        scheme_cls = scheme_module.CompressedTensorsW4A8Int
        config_cls = config_module.CompressedTensorsConfig
        scheme_installed = getattr(scheme_cls, "_vllm_fl_arm_packed_w4a8", False)
        kernel_installed = getattr(
            Dynamic4bitLinearKernel, "_vllm_fl_arm_w4a8", False
        )
        if scheme_installed != kernel_installed:
            raise RuntimeError("incomplete ARM W4A8 runtime installation detected")
        if scheme_installed:
            return False

        original_init = scheme_cls.__init__
        original_create_weights = scheme_cls.create_weights
        original_get_scheme = config_cls._get_scheme_from_parts
        original_process = Dynamic4bitLinearKernel.process_weights_after_loading
        original_apply = Dynamic4bitLinearKernel.apply_weights

        def scheme_init(
            self,
            strategy: str,
            num_bits: int,
            group_size: int | None = None,
            is_static_input_scheme: bool = False,
            input_symmetric: bool = True,
            packed: bool = False,
        ) -> None:
            original_init(
                self,
                strategy=strategy,
                num_bits=num_bits,
                group_size=group_size,
                is_static_input_scheme=is_static_input_scheme,
                input_symmetric=input_symmetric,
            )
            self._vllm_fl_checkpoint_packed = packed
            self._vllm_fl_pack_factor = 32 // num_bits

        def create_weights(
            self,
            layer: torch.nn.Module,
            output_size: int,
            input_size: int,
            output_partition_sizes: list[int],
            input_size_per_partition: int,
            params_dtype: torch.dtype,
            weight_loader,
            **kwargs,
        ) -> None:
            if not getattr(self, "_vllm_fl_checkpoint_packed", False):
                original_create_weights(
                    self,
                    layer,
                    output_size,
                    input_size,
                    output_partition_sizes,
                    input_size_per_partition,
                    params_dtype,
                    weight_loader,
                    **kwargs,
                )
                return

            output_size_per_partition = sum(output_partition_sizes)
            row_parallel = input_size != input_size_per_partition
            effective_group_size = (
                input_size_per_partition
                if self.group_size == -1 and row_parallel
                else input_size
                if self.group_size == -1
                else self.group_size
            )
            if input_size_per_partition % effective_group_size:
                raise ValueError(
                    f"input partition {input_size_per_partition} is not "
                    f"divisible by W4 group size {effective_group_size}"
                )

            kernel_config = MPLinearLayerConfig(
                full_weight_shape=(input_size, output_size),
                partition_weight_shape=(
                    input_size_per_partition,
                    output_size_per_partition,
                ),
                weight_type=self.quant_type,
                act_type=params_dtype,
                group_size=effective_group_size,
                zero_points=False,
                has_g_idx=False,
            )
            kernel_type = choose_mp_linear_kernel(kernel_config)

            weight = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // self._vllm_fl_pack_factor,
                    dtype=torch.int32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
                packed_factor=self._vllm_fl_pack_factor,
                packed_dim=1,
            )
            layer.register_parameter("weight_packed", weight)

            scale_args = {
                "weight_loader": weight_loader,
                "data": torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // effective_group_size,
                    dtype=params_dtype,
                ),
            }
            if self.group_size == -1 and row_parallel:
                weight_scale = ChannelQuantScaleParameter(
                    output_dim=0, **scale_args
                )
            else:
                weight_scale = GroupQuantScaleParameter(
                    output_dim=0, input_dim=1, **scale_args
                )
            layer.register_parameter("weight_scale", weight_scale)
            layer.register_parameter(
                "weight_shape",
                BasevLLMParameter(
                    data=torch.empty(2, dtype=torch.int64),
                    weight_loader=weight_loader,
                ),
            )
            self.kernel = kernel_type(
                kernel_config,
                w_q_param_name="weight_packed",
                w_s_param_name="weight_scale",
                w_zp_param_name=None,
                w_gidx_param_name=None,
            )
            # Limit the runtime hooks below to kernels created for the packed
            # checkpoint path.  Stock vLLM W4A8 kernels must keep their
            # original loading and execution behavior.
            self.kernel._vllm_fl_arm_packed_checkpoint = True
            return

        def get_scheme_from_parts(
            self,
            weight_quant,
            input_quant,
            output_quant=None,
            format: str | None = None,
            layer_name: str | None = None,
        ):
            resolved_format = format if format is not None else self.quant_format
            if (
                resolved_format == CompressionFormat.pack_quantized.value
                and self._is_dynamic_token_w4a8_int(weight_quant, input_quant)
            ):
                return scheme_cls(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    is_static_input_scheme=False,
                    input_symmetric=input_quant.symmetric,
                    packed=True,
                )
            return original_get_scheme(
                self,
                weight_quant,
                input_quant,
                output_quant=output_quant,
                format=format,
                layer_name=layer_name,
            )

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            if not getattr(self, "_vllm_fl_arm_packed_checkpoint", False):
                original_process(self, layer)
                return

            config = self.config
            if config.group_size != 128 or config.zero_points:
                original_process(self, layer)
                return

            quantized = getattr(layer, self.w_q_name)
            scales = getattr(layer, self.w_s_name)
            k, n = config.partition_weight_shape
            packed_checkpoint = quantized.dtype == torch.int32 and quantized.shape == (
                n,
                k // 8,
            )
            if packed_checkpoint:
                weight_shape = getattr(layer, "weight_shape", None)
                if weight_shape is not None:
                    loaded_shape = tuple(int(value) for value in weight_shape.tolist())
                    if (
                        len(loaded_shape) != 2
                        or loaded_shape[1] != k
                        or not 0 < loaded_shape[0] <= n
                    ):
                        raise RuntimeError(
                            "unexpected packed-Q4 weight_shape: "
                            f"loaded={loaded_shape}, expected=(*,{k})"
                        )
                quantized_nk = unpack_from_int32(
                    quantized.detach(), 4, torch.Size((n, k))
                ).contiguous()
            else:
                quantized_nk = quantized

            if (
                quantized_nk.dtype != torch.int8
                or quantized_nk.shape != (n, k)
                or scales.shape != (n, k // 128)
            ):
                raise RuntimeError("unexpected compressed-tensors G128 parameter shape")

            from flag_gems.quantized_linear import pack_rhs_qsi4c128p

            packed_weight = pack_rhs_qsi4c128p(
                quantized_nk,
                scales.detach().to(torch.bfloat16).contiguous(),
            )
            replace_parameter(
                layer,
                self.w_q_name,
                torch.nn.Parameter(packed_weight, requires_grad=False),
            )

            source_alias = getattr(layer, "weight", None)
            if (
                self.w_q_name != "weight"
                and source_alias is not None
                and source_alias.untyped_storage().data_ptr()
                == quantized.untyped_storage().data_ptr()
            ):
                replace_parameter(
                    layer,
                    "weight",
                    torch.nn.Parameter(
                        torch.empty(
                            0,
                            dtype=quantized.dtype,
                            device=quantized.device,
                        ),
                        requires_grad=False,
                    ),
                )

            setattr(layer, self.w_s_name, None)
            if packed_checkpoint:
                layer.weight_shape = None
            self._vllm_fl_arm_w4a8_shape = (n, k)
            return

        def apply_weights(
            self,
            layer: torch.nn.Module,
            input: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            shape = getattr(self, "_vllm_fl_arm_w4a8_shape", None)
            if shape is None:
                return original_apply(self, layer, input, bias)

            from flag_gems.quantized_linear import w4a8_g128_linear

            n, k = shape
            source_dtype = input.dtype
            input_bf16 = (
                input if source_dtype == torch.bfloat16 else input.to(torch.bfloat16)
            )
            output = w4a8_g128_linear(
                input_bf16.contiguous(),
                getattr(layer, self.w_q_name),
                n,
                k,
            )
            if source_dtype != torch.bfloat16:
                output = output.to(source_dtype)
            if bias is not None:
                output = output + bias.to(output.dtype)
            return output

        scheme_cls.__init__ = scheme_init
        scheme_cls.create_weights = create_weights
        config_cls._get_scheme_from_parts = get_scheme_from_parts
        Dynamic4bitLinearKernel.process_weights_after_loading = (
            process_weights_after_loading
        )
        Dynamic4bitLinearKernel.apply_weights = apply_weights
        scheme_cls._vllm_fl_arm_packed_w4a8 = True
        scheme_cls._vllm_fl_arm_original_init = original_init
        scheme_cls._vllm_fl_arm_original_create_weights = original_create_weights
        config_cls._vllm_fl_arm_original_get_scheme = original_get_scheme
        Dynamic4bitLinearKernel._vllm_fl_arm_w4a8 = True
        Dynamic4bitLinearKernel._vllm_fl_arm_original_process = original_process
        Dynamic4bitLinearKernel._vllm_fl_arm_original_apply = original_apply
        return True

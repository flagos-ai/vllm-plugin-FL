# Copyright 2026 FlagOS Contributors
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
"""Hardware scaled-mm adapters for dynamic-token/per-channel W8A8."""

from __future__ import annotations

import inspect

import torch

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import (
    is_oot_enabled,
    use_flaggems_op,
)

FLAGGEMS_W8A8_LINEAR_OP = "w8a8_dynamic_per_token_linear"

_dynamic_per_token_quant_int8 = CachedOp("dynamic_per_token_quant_int8")
_prepare_ascend_weight = CachedOp("w8a8_prepare_weight")
_ascend_dynamic_linear = CachedOp("w8a8_dynamic_linear")


class FLW8A8DynamicLinearKernel(Int8ScaledMMLinearKernel):
    """Use FlagGems scaled-mm with the vLLM dynamic-W8A8 contract.

    Runtime operands are ``A=[M,K] int8``, ``B=[K,N] int8``,
    ``scale_a=[M,1] fp32`` and ``scale_b=[N] fp32``.
    """

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not is_oot_enabled():
            return False, "FL W8A8 kernels are disabled"
        if not use_flaggems_op(FLAGGEMS_W8A8_LINEAR_OP):
            return False, "FlagGems W8A8 linear is disabled by policy"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: Int8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        if not config.is_channelwise:
            return False, "requires per-channel weights"
        if config.is_static_input_scheme:
            return False, "requires dynamic input quantization"
        if not config.input_symmetric:
            return False, "requires symmetric INT8 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        weight = getattr(layer, w_q_name)
        weight_scale = getattr(layer, w_s_name)
        if weight.ndim != 2 or weight.dtype != torch.int8:
            raise ValueError("FL W8A8 expects checkpoint weight as 2D int8")
        if weight_scale.numel() != weight.shape[0]:
            raise ValueError("FL W8A8 requires one scale per output channel")

        # FlagGems scaled_mm consumes B in [K, N] layout. Keep scales as [N].
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(weight.t().contiguous(), requires_grad=False),
        )
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(
                weight_scale.reshape(-1).to(torch.float32).contiguous(),
                requires_grad=False,
            ),
        )
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from flag_gems import scaled_mm

        weight, weight_scale, _, _, _ = self._get_layer_params(layer)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        x_q, x_scale = _dynamic_per_token_quant_int8(x_2d)
        output = scaled_mm(
            x_q,
            weight,
            x_scale,
            weight_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return output.reshape(*original_shape[:-1], weight.shape[1])


class AscendInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """Use upstream checkpoint parameters and native NPU dynamic W8A8 math.

    CompressedTensorsW8A8Int8 owns allocation and checkpoint loading of
    logical INT8 [N,K] weights and FP32 channel scales [N,1]. Preparation
    transposes weights to native [K,N], optionally casts NZ, and flattens scales.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        del compute_capability
        if (
            getattr(current_platform, "vendor_name", None) != "ascend"
            or current_platform.device_type != "npu"
        ):
            return False, "requires the Ascend NPU platform"
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not config.is_channelwise:
            return False, "requires per-channel weights"
        if config.is_static_input_scheme:
            return False, "requires dynamic input quantization"
        if not config.input_symmetric:
            return False, "requires symmetric INT8 activations"
        return True, None

    def make_grouped_method(self, method):
        """Keep grouped preparation in the kernel-selected device adapter."""
        return AscendGroupedW8A8LinearMethod(method.quantization_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        weight, scale, _, _, _ = self._get_layer_params(layer)
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("Expected W8A8 [N,K] int8 weight")
        if scale.shape != (weight.shape[0], 1):
            raise ValueError("Expected one W8A8 scale per output channel")
        if weight.shape[0] >= 65536:
            raise NotImplementedError(
                "Ascend W8A8 does not support unchunked output width "
                f"N={weight.shape[0]} (requires N < 65536)"
            )
        weight_nz_mode = _weight_nz_mode(layer)
        prepared_weight, prepared_scale = _prepare_ascend_weight(
            weight.data.transpose(0, 1).contiguous(),
            scale.data.flatten(),
            weight_nz_mode,
        )
        getattr(layer, w_q_name).data = prepared_weight
        getattr(layer, w_s_name).data = prepared_scale
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dynamically quantize inputs and execute native INT8 GEMM with bias."""
        weight, scale, _, _, _ = self._get_layer_params(layer)
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Dynamic W8A8 Linear requires BF16/FP16 input")
        if x.ndim < 2 or x.shape[-1] != weight.shape[0]:
            raise ValueError("Expected [..., K] activations matching prepared [K,N]")
        if bias is not None and bias.shape != (weight.shape[1],):
            raise ValueError("Expected bias [N]")
        shape = (*x.shape[:-1], weight.shape[1])
        return _ascend_dynamic_linear(
            x.reshape(-1, x.shape[-1]).contiguous(), weight, scale, bias
        ).reshape(shape)


def _is_ascend_npu():
    from vllm.platforms import current_platform

    return (
        getattr(current_platform, "vendor_name", None) == "ascend"
        and current_platform.device_type == "npu"
    )


def _weight_nz_mode(layer):
    from vllm.config import get_current_vllm_config

    mode = getattr(layer, "_fl_ascend_weight_nz_mode", None)
    if mode is None:
        config = get_current_vllm_config()
        mode = (config.additional_config or {}).get("weight_nz_mode", 1)
    if mode not in (0, 1, 2):
        raise ValueError("weight_nz_mode must be 0, 1 or 2")
    return mode


class AscendW8A8Int8(CompressedTensorsW8A8Int8):
    """Keep the NPU activation contract around upstream weight allocation."""

    def create_weights(
        self,
        layer,
        output_partition_sizes,
        input_size_per_partition,
        params_dtype,
        weight_loader,
        **kwargs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Dynamic W8A8 Linear requires BF16/FP16")
        return super().create_weights(
            layer,
            output_partition_sizes,
            input_size_per_partition,
            params_dtype,
            weight_loader,
            **kwargs,
        )


_prepare_weight = _prepare_ascend_weight


_linear = _ascend_dynamic_linear


class AscendGroupedW8A8LinearMethod(CompressedTensorsLinearMethod):
    """Project independent output groups; this is not quantization group_size.

    Upstream allocates logical INT8 weights [G*O,I] and FP32 scales [G*O,1].
    Each input group [T,I] uses its own weight [O,I], prepared as [I,O] with
    optional NZ layout and flat scales; concatenate the G outputs to [T,G*O].
    """

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.int8 or layer.weight.ndim != 2:
            raise ValueError("Expected W8A8 [N,K] int8 weight")
        if layer.weight_scale.shape != (layer.weight.shape[0], 1):
            raise ValueError("Expected one W8A8 scale per output channel")
        if getattr(self, "output_groups", None) is None:
            raise ValueError(
                "Grouped W8A8 requires configure_grouped before preparation"
            )
        weight_nz_mode = _weight_nz_mode(layer)
        groups = self.output_groups
        if groups < 1 or layer.weight.shape[0] % groups:
            raise ValueError("Grouped W8A8 outputs must divide the weight rows")
        output_width = layer.weight.shape[0] // groups
        if output_width >= 65536:
            raise NotImplementedError(
                "Ascend W8A8 does not support per-group output width "
                f"N={output_width} (requires N < 65536)"
            )
        self.group_input_size = layer.weight.shape[1]
        for group, (weight, scale) in enumerate(
            zip(
                layer.weight.data.chunk(groups, dim=0),
                layer.weight_scale.data.chunk(groups, dim=0),
            )
        ):
            weight, scale = _prepare_weight(
                weight.transpose(0, 1).contiguous(),
                scale.flatten(),
                weight_nz_mode,
            )
            layer.register_buffer(
                f"_grouped_w8a8_weight_{group}", weight, persistent=False
            )
            layer.register_buffer(
                f"_grouped_w8a8_scale_{group}", scale, persistent=False
            )
        # Keep parameter ownership while releasing the unused logical bank.
        # Other groups remain module-owned buffers, moved with the layer.
        layer.weight.data = layer._grouped_w8a8_weight_0

    def configure_grouped(self, groups):
        """Set the independent output groups before post-load preparation."""
        if groups < 1:
            raise ValueError("Grouped W8A8 requires positive groups")
        self.output_groups = groups

    def apply(self, layer, x, bias=None):
        """Run dynamic quantization/GEMM per input group and concatenate outputs."""
        groups = self.output_groups
        tokens = x.shape[0]
        grouped = x.reshape(tokens, groups, self.group_input_size)
        outputs = [
            _linear(
                grouped[:, group].contiguous(),
                getattr(layer, f"_grouped_w8a8_weight_{group}"),
                getattr(layer, f"_grouped_w8a8_scale_{group}"),
            )
            for group in range(groups)
        ]
        output = torch.cat(outputs, dim=-1)
        return output if bias is None else output + bias

    def apply_grouped(self, layer, x):
        """Compatibility entrypoint for existing component callers."""
        return self.apply(layer, x)


_grouped_project = CachedOp("deepseek_v4_grouped_project")


class UnquantizedGroupedLinearMethod(UnquantizedLinearMethod):
    """Reuse upstream allocation for independent floating-point projections."""

    def configure_grouped(self, groups):
        self.output_groups = groups

    def process_weights_after_loading(self, layer):
        # The provider consumes the logical checkpoint layout on every device.
        pass

    def apply(self, layer, x, bias=None):
        output = _grouped_project(x, layer.weight, self.output_groups)
        return output if bias is None else output + bias


def configure_grouped_projection(layer, groups):
    """Select a standard Linear method through its kernel's grouped capability."""
    if not isinstance(groups, int) or groups < 1:
        raise ValueError("Grouped projection requires positive integer groups")
    method = layer.quant_method
    if callable(getattr(method, "configure_grouped", None)):
        method.configure_grouped(groups)
        layer.is_bmm = True
        layer.bmm_batch_size = groups
        return
    factory = getattr(method, "make_grouped_method", None)
    if factory is None:
        kernel = getattr(getattr(layer, "scheme", None), "kernel", None)
        factory = getattr(kernel, "make_grouped_method", None)
    if factory is not None:
        method = factory(method)
    elif isinstance(method, UnquantizedLinearMethod):
        method = UnquantizedGroupedLinearMethod()
    else:
        raise NotImplementedError(
            "Selected Linear kernel does not support grouped projection"
        )
    method.configure_grouped(groups)
    layer.quant_method = method
    layer.is_bmm = True
    layer.bmm_batch_size = groups


def _native_dynamic_scheme(scheme):
    from vllm_fl.quantization.w8a8.int8_mode import is_dynamic_token_int8

    weight = scheme.get("weights")
    return (
        weight is not None
        and weight.num_bits == 8
        and weight.type == "int"
        and weight.strategy == "channel"
        and weight.symmetric
        and not weight.dynamic
        and weight.group_size is None
        and is_dynamic_token_int8(scheme.get("input_activations"))
        and scheme.get("output_activations") is None
    )


def _int8_weight(weight):
    return weight is not None and weight.num_bits == 8 and weight.type == "int"


def install_ascend_w8a8_linear_adapter():
    """Compose device execution hooks around the current upstream methods."""
    if not _is_ascend_npu():
        return
    marker = "_fl_ascend_w8a8_linear"
    # Resolve and validate every dependency before changing upstream objects.
    current_method = CompressedTensorsConfig.get_quant_method
    current_parts = CompressedTensorsConfig._get_scheme_from_parts
    token = object()
    inspect.signature(current_method).bind(token, token, token)
    inspect.signature(CompressedTensorsConfig.get_scheme_dict).bind(token, token, token)
    inspect.signature(current_parts).bind(
        token,
        weight_quant=token,
        input_quant=token,
        output_quant=None,
        format=None,
        layer_name=None,
    )
    method_installed = getattr(current_method, marker, False)
    parts_installed = getattr(current_parts, marker, False)
    if method_installed and parts_installed:
        return

    updates = {}
    if not method_installed:

        def get_quant_method(config, layer, prefix):
            if _is_ascend_npu() and isinstance(layer, (LinearBase, ParallelLMHead)):
                # Validate before upstream's LMHead fallback catches ValueError.
                scheme = config.get_scheme_dict(layer, prefix)
                if scheme is not None:
                    fmt = scheme.get("format") or config.quant_format
                    native = fmt == "int-quantized" and _int8_weight(
                        scheme.get("weights")
                    )
                    if native and not _native_dynamic_scheme(scheme):
                        raise NotImplementedError(
                            f"Unsupported Ascend INT8 Linear scheme at {prefix}"
                        )
                    packed = False
                    if fmt == "pack-quantized":
                        from vllm_fl.quantization.w8a8.int8_mode import (
                            should_use_packed_w8a8,
                        )

                        packed = should_use_packed_w8a8(
                            scheme.get("weights"), scheme.get("input_activations"), fmt
                        )
                    if (native or packed) and (
                        config.kv_cache_scheme or config.transform_config
                    ):
                        raise NotImplementedError(
                            "Ascend W8A8 does not implement KV quantization or transforms"
                        )
            return current_method(config, layer, prefix)

        setattr(get_quant_method, marker, True)
        updates["get_quant_method"] = get_quant_method

    if not parts_installed:

        def get_scheme_from_parts(
            config,
            weight_quant,
            input_quant,
            output_quant=None,
            format=None,
            layer_name=None,
        ):
            fmt = format if format is not None else config.quant_format
            if (
                _is_ascend_npu()
                and fmt == "int-quantized"
                and _int8_weight(weight_quant)
            ):
                scheme = {
                    "weights": weight_quant,
                    "input_activations": input_quant,
                    "output_activations": output_quant,
                }
                if not _native_dynamic_scheme(scheme):
                    raise NotImplementedError(
                        f"Unsupported Ascend INT8 Linear scheme at {layer_name}"
                    )
                return AscendW8A8Int8(weight_quant.strategy, False, True)
            return current_parts(
                config,
                weight_quant=weight_quant,
                input_quant=input_quant,
                output_quant=output_quant,
                format=format,
                layer_name=layer_name,
            )

        setattr(get_scheme_from_parts, marker, True)
        updates["_get_scheme_from_parts"] = get_scheme_from_parts

    missing = object()
    originals = {
        name: CompressedTensorsConfig.__dict__.get(name, missing) for name in updates
    }
    try:
        for name, value in updates.items():
            setattr(CompressedTensorsConfig, name, value)
    except Exception:
        # Restore descriptors and existing wrappers exactly, including markers.
        for name, original in reversed(tuple(originals.items())):
            if original is missing:
                if name in CompressedTensorsConfig.__dict__:
                    delattr(CompressedTensorsConfig, name)
            else:
                setattr(CompressedTensorsConfig, name, original)
        raise


def register_fl_w8a8_linear_kernel(registry: dict) -> bool:
    """Prefer native Ascend on NPU, retaining the existing other-vendor order."""
    from vllm.platforms import PlatformEnum

    install_ascend_w8a8_linear_adapter()
    # PlatformFL always selects through the OOT registry. On NVIDIA that list
    # has already inherited CUDA's native Cutlass/Triton candidates, so
    # prepending here keeps them available as ordered fallbacks.
    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if FLW8A8DynamicLinearKernel not in candidates:
        candidates.insert(0, FLW8A8DynamicLinearKernel)
    if (
        AscendInt8ScaledMMLinearKernel.is_supported()[0]
        and AscendInt8ScaledMMLinearKernel not in candidates
    ):
        candidates.insert(0, AscendInt8ScaledMMLinearKernel)
    return True


__all__ = [
    "AscendInt8ScaledMMLinearKernel",
    "AscendW8A8Int8",
    "AscendGroupedW8A8LinearMethod",
    "UnquantizedGroupedLinearMethod",
    "configure_grouped_projection",
    "install_ascend_w8a8_linear_adapter",
    "FLAGGEMS_W8A8_LINEAR_OP",
    "FLW8A8DynamicLinearKernel",
    "register_fl_w8a8_linear_kernel",
]

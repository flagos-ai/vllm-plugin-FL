# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Route vLLM 0.24 W8A8 INT8 MoE by hardware backend."""

from importlib import import_module
from inspect import signature
from threading import RLock

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8MoEMethod,
)

from vllm_fl.dispatch import CachedOp

_ADAPTER_MARKER = "_vllm_fl_w8a8_int8_moe_adapter"
_CONFIG_BUILDER_MARKER = "_vllm_fl_dynamic_w8a8_config"
_ORACLE_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
_SCHEME_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
# Reuse the repository's existing MoE policy key so current platform
# blacklists/whitelists keep governing the whole FL MoE pipeline.
FLAGGEMS_W8A8_MOE_OP = "fused_moe"
logger = init_logger(__name__)
_INSTALL_LOCK = RLock()


def install_fl_w8a8_moe_selector() -> bool:
    """Install dependent MoE hooks atomically, retaining existing wrappers."""
    with _INSTALL_LOCK:
        oracle = import_module(_ORACLE_MODULE)
        scheme = import_module(_SCHEME_MODULE)
        from vllm.platforms import current_platform

        targets = [
            (scheme, "make_int8_moe_quant_config"),
            (oracle, "select_int8_moe_backend"),
            (scheme, "select_int8_moe_backend"),
        ]
        token = object()
        signature(scheme.make_int8_moe_quant_config).bind(
            w1_scale=token,
            w2_scale=token,
            a1_scale=None,
            a2_scale=None,
            w1_bias=None,
            w2_bias=None,
            per_act_token_quant=True,
        )
        for owner in (oracle, scheme):
            signature(owner.select_int8_moe_backend).bind(
                token, weight_key=None, activation_key=None
            )
        if (
            getattr(current_platform, "vendor_name", None) == "ascend"
            and current_platform.device_type == "npu"
        ):
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
                CompressedTensorsMoEMethod,
            )

            signature(CompressedTensorsMoEMethod.get_moe_method).bind(
                token, token, token
            )
            targets.append((CompressedTensorsMoEMethod, "get_moe_method"))
        missing = object()
        originals = [
            (owner, name, vars(owner).get(name, missing)) for owner, name in targets
        ]
        try:
            return _install_fl_w8a8_moe_selector()
        except Exception:
            for owner, name, original in reversed(originals):
                if original is missing:
                    if name in vars(owner):
                        delattr(owner, name)
                else:
                    setattr(owner, name, original)
            raise


def _install_fl_w8a8_moe_selector() -> bool:
    """Prefer FlagGems W8A8 MoE and retain the native NVIDIA fallback."""
    oracle_module = import_module(_ORACLE_MODULE)
    scheme_module = import_module(_SCHEME_MODULE)

    # vLLM 0.24 treats absent checkpoint activation scales as W8A16 before
    # consulting per_act_token_quant. Dynamic-token W8A8 intentionally stores
    # no activation scales, so retain the scheme's explicit signal.
    current_builder = scheme_module.make_int8_moe_quant_config
    if not getattr(current_builder, _CONFIG_BUILDER_MARKER, False):

        def make_int8_moe_quant_config_fl(
            w1_scale,
            w2_scale,
            a1_scale=None,
            a2_scale=None,
            w1_bias=None,
            w2_bias=None,
            per_act_token_quant=False,
        ):
            if not per_act_token_quant:
                return current_builder(
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    w1_bias=w1_bias,
                    w2_bias=w2_bias,
                    per_act_token_quant=False,
                )

            from vllm.model_executor.layers.fused_moe.config import (
                int8_w8a8_moe_quant_config,
            )

            return int8_w8a8_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                per_act_token_quant=True,
            )

        setattr(
            make_int8_moe_quant_config_fl,
            _CONFIG_BUILDER_MARKER,
            True,
        )
        scheme_module.make_int8_moe_quant_config = make_int8_moe_quant_config_fl

    _install_ascend_moe_method_factory()

    current_selector = oracle_module.select_int8_moe_backend
    if getattr(current_selector, _ADAPTER_MARKER, False):
        scheme_module.select_int8_moe_backend = current_selector
        return True

    def select_int8_moe_backend_fl(
        config,
        weight_key=None,
        activation_key=None,
    ):
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt8DynamicTokenSym,
            kInt8StaticChannelSym,
        )
        from vllm.platforms import current_platform

        from vllm_fl.utils import (
            is_oot_enabled,
            use_flaggems_op,
        )

        canonical_w8a8 = weight_key in (
            None,
            kInt8StaticChannelSym,
        ) and activation_key in (None, kInt8DynamicTokenSym)

        # Ascend owns the dynamic quantization inside its native experts on
        # either FlagGems setting. Other OOT vendors retain the common policy.
        if (
            getattr(current_platform, "vendor_name", None) == "ascend"
            and current_platform.device_type == "npu"
        ):
            from vllm_fl.quantization.w8a8.moe_experts import (
                AscendW8A8DynamicExperts,
            )

            AscendW8A8DynamicExperts.validate_config(config)
            supported, reason = AscendW8A8DynamicExperts.is_supported_config(
                AscendW8A8DynamicExperts,
                config,
                weight_key or kInt8StaticChannelSym,
                activation_key or kInt8DynamicTokenSym,
                AscendW8A8DynamicExperts.activation_format(),
            )
            if not supported:
                raise NotImplementedError(f"Unsupported Ascend W8A8 MoE: {reason}")
            logger.info_once("Using native Ascend dynamic W8A8 MoE experts.")
            return oracle_module.Int8MoeBackend.TRITON, AscendW8A8DynamicExperts

        use_fl = (
            canonical_w8a8
            and (current_platform.is_cuda() or current_platform.is_out_of_tree())
            and is_oot_enabled()
            and use_flaggems_op(FLAGGEMS_W8A8_MOE_OP)
        )
        if use_fl:
            if getattr(config, "is_lora_enabled", False):
                raise NotImplementedError(
                    "The FlagGems W8A8 MoE adapter does not support LoRA"
                )
            if config.moe_parallel_config.use_batched_activation_format:
                raise ValueError(
                    "FL W8A8 MoE currently requires the standard activation "
                    "format; batched-experts dispatch is not supported"
                )

            from vllm_fl.quantization.w8a8.moe_experts import FlagGemsW8A8Experts

            logger.info_once("Using FlagGems W8A8 MoE experts.")
            return (
                oracle_module.Int8MoeBackend.TRITON,
                FlagGemsW8A8Experts,
            )

        # Keep NVIDIA's native functional path as the fallback when the common
        # FlagGems gate is disabled. The bridge corrects the modular input
        # contract: prepare must keep the first activation floating-point so
        # fused_experts can perform both dynamic quantization steps itself.
        if current_platform.is_cuda() and canonical_w8a8:
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation

            # The functional fused_experts API drops the SwiGLU clamp/alpha/
            # beta fields from the quant config. Let upstream select modular
            # experts for configurable activations so both parameter passing
            # and activation-input quantization remain under its control.
            activation = getattr(config, "activation", MoEActivation.SILU)
            if activation is not MoEActivation.SILU or any(
                getattr(config, name, None) is not None
                for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
            ):
                logger.info_once(
                    "Using upstream modular INT8 MoE experts to preserve "
                    "activation parameters."
                )
                backend, experts_cls = current_selector(
                    config,
                    weight_key=weight_key,
                    activation_key=activation_key,
                )
                from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
                    TritonExperts,
                )

                from vllm_fl.quantization.w8a8.moe_experts import (
                    VllmFunctionalW8A8Experts,
                )

                if experts_cls is TritonExperts:
                    experts_cls = VllmFunctionalW8A8Experts
                return backend, experts_cls

            if getattr(config, "is_lora_enabled", False):
                raise NotImplementedError(
                    "The vLLM functional W8A8 MoE bridge does not support LoRA"
                )
            if config.moe_parallel_config.use_batched_activation_format:
                raise ValueError(
                    "The vLLM functional W8A8 MoE bridge requires the standard "
                    "activation format; batched-experts dispatch is not supported"
                )

            from vllm_fl.quantization.w8a8.moe_experts import (
                VllmFunctionalW8A8Experts,
            )

            logger.info_once("Using native vLLM functional W8A8 MoE experts on NVIDIA.")
            return (
                oracle_module.Int8MoeBackend.TRITON,
                VllmFunctionalW8A8Experts,
            )

        return current_selector(
            config,
            weight_key=weight_key,
            activation_key=activation_key,
        )

    setattr(select_int8_moe_backend_fl, _ADAPTER_MARKER, True)
    oracle_module.select_int8_moe_backend = select_int8_moe_backend_fl
    # The compressed-tensors scheme imported the selector by name.
    scheme_module.select_int8_moe_backend = select_int8_moe_backend_fl
    return True


_METHOD_FACTORY_MARKER = "_fl_ascend_w8a8_method_factory"


def _install_ascend_moe_method_factory():
    from vllm.platforms import current_platform

    if (
        getattr(current_platform, "vendor_name", None) != "ascend"
        or current_platform.device_type != "npu"
    ):
        return
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )

    current = CompressedTensorsMoEMethod.get_moe_method
    if getattr(current, _METHOD_FACTORY_MARKER, False):
        return

    def get_moe_method(config, layer, layer_name):
        # The framework still parses and matches the checkpoint scheme. Only
        # the selected native INT8 method needs a different post-load contract.
        method = current(config, layer, layer_name)
        from vllm.platforms import current_platform

        if (
            getattr(current_platform, "vendor_name", None) == "ascend"
            and current_platform.device_type == "npu"
            and type(method) is CompressedTensorsW8A8Int8MoEMethod
        ):
            return AscendW8A8DynamicMoEMethod(
                method.weight_quant,
                method.input_quant,
                layer.moe_config,
                layer_name=layer_name,
            )
        return method

    setattr(get_moe_method, _METHOD_FACTORY_MARKER, True)
    CompressedTensorsMoEMethod.get_moe_method = staticmethod(get_moe_method)


_prepare = CachedOp("w8a8_prepare_experts")


class AscendW8A8DynamicMoEMethod(CompressedTensorsW8A8Int8MoEMethod):
    """Reuse upstream loading and modular lifecycle with native Ascend math."""

    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        # Also support direct component construction outside platform startup.
        # One idempotent installer owns both selector and dynamic-config seams.
        install_fl_w8a8_moe_selector()
        from vllm_fl.quantization.w8a8.moe_experts import (
            AscendW8A8DynamicExperts,
        )

        AscendW8A8DynamicExperts.validate_config(moe)
        super().__init__(weight_quant, input_quant, moe, layer_name)

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Dynamic W8A8 experts require BF16/FP16")
        # The upstream allocator replaces params_dtype with INT8. Preparation
        # still needs the original activation dtype to select GMM scale dtype.
        self._activation_dtype = params_dtype
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer):
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")
            if weight.dtype != torch.int8 or weight.ndim != 3:
                raise ValueError("Expected W8A8 INT8 [experts, output, input] weights")
            if scale.shape != (*weight.shape[:2], 1):
                raise ValueError("Expected W8A8 per-channel expert scales")
        prepared = _prepare(
            layer.w13_weight.data,
            layer.w2_weight.data,
            layer.w13_weight_scale.data,
            layer.w2_weight_scale.data,
            output_dtype=self._activation_dtype,
        )
        for name, tensor in zip(
            ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"), prepared
        ):
            getattr(layer, name).data = tensor
        super().process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer):
        config = super().get_fused_moe_quant_config(layer)
        config.gemm1_clamp_limit = self.moe.swiglu_limit
        return config


__all__ = [
    "AscendW8A8DynamicMoEMethod",
    "FLAGGEMS_W8A8_MOE_OP",
    "install_fl_w8a8_moe_selector",
]

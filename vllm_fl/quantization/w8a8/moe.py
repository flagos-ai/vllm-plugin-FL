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

from vllm.logger import init_logger

_ADAPTER_MARKER = "_vllm_fl_w8a8_int8_moe_v024"
_CONFIG_BUILDER_MARKER = "_vllm_fl_dynamic_w8a8_config_v024"
_ORACLE_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
_SCHEME_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
# Reuse the repository's existing MoE policy key so current platform
# blacklists/whitelists keep governing the whole FL MoE pipeline.
FLAGGEMS_W8A8_MOE_OP = "fused_moe"
logger = init_logger(__name__)


def install_fl_w8a8_moe_selector() -> bool:
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


__all__ = [
    "FLAGGEMS_W8A8_MOE_OP",
    "install_fl_w8a8_moe_selector",
]

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
"""Route vLLM's W8A8 INT8 MoE scheme to the FL experts implementation."""

from importlib import import_module

_ADAPTER_MARKER = "_vllm_fl_w8a8_int8_moe"
_CONFIG_BUILDER_MARKER = "_vllm_fl_dynamic_w8a8_config"
_ORACLE_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
_SCHEME_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)


def install_fl_w8a8_moe_selector() -> bool:
    """Let canonical W8A8 MoE checkpoints reach FlagGems on OOT devices."""
    oracle_module = import_module(_ORACLE_MODULE)
    scheme_module = import_module(_SCHEME_MODULE)

    # vLLM 0.20.2 treats missing activation scales as W8A16 in the shared
    # helper. Dynamic per-token W8A8 intentionally has no checkpoint scale, so
    # preserve the scheme's explicit per_act_token_quant signal instead.
    current_builder = scheme_module.make_int8_moe_quant_config
    if not getattr(current_builder, _CONFIG_BUILDER_MARKER, False):

        def make_int8_moe_quant_config_fl(
            w1_scale,
            w2_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=False,
        ):
            if not per_act_token_quant:
                return current_builder(
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
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
                per_act_token_quant=True,
            )

        setattr(
            make_int8_moe_quant_config_fl,
            _CONFIG_BUILDER_MARKER,
            True,
        )
        scheme_module.make_int8_moe_quant_config = (
            make_int8_moe_quant_config_fl
        )

    current_selector = oracle_module.select_int8_moe_backend
    if getattr(current_selector, _ADAPTER_MARKER, False):
        # Keep the scheme module in sync even if it was imported after a prior
        # installation.
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

        from vllm_fl.utils import is_oot_enabled, use_flaggems_op

        canonical_w8a8 = (
            weight_key in (None, kInt8StaticChannelSym)
            and activation_key in (None, kInt8DynamicTokenSym)
        )
        use_fl = (
            current_platform.is_out_of_tree()
            and is_oot_enabled()
            and use_flaggems_op("fused_moe")
            and canonical_w8a8
        )
        if not use_fl:
            return current_selector(
                config,
                weight_key=weight_key,
                activation_key=activation_key,
            )

        if config.moe_parallel_config.use_batched_activation_format:
            raise ValueError(
                "FL W8A8 MoE currently requires the standard activation "
                "format; batched-experts dispatch is not supported"
            )

        from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

        return oracle_module.Int8MoeBackend.TRITON, TritonExpertsFL

    setattr(select_int8_moe_backend_fl, _ADAPTER_MARKER, True)
    oracle_module.select_int8_moe_backend = select_int8_moe_backend_fl
    # compressed_tensors_w8a8_int8 imports the selector by name, so patch its
    # module-local binding as well.
    scheme_module.select_int8_moe_backend = select_int8_moe_backend_fl
    return True


__all__ = ["install_fl_w8a8_moe_selector"]

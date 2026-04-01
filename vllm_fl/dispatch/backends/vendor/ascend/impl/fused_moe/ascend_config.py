#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
import os
import warnings
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        additional_config = (
            vllm_config.additional_config
            if vllm_config.additional_config is not None
            else {}
        )

        finegrained_tp_config = {}
        self.finegrained_tp_config = FinegrainedTPConfig(
            finegrained_tp_config, vllm_config
        )

        eplb_config = {}
        self.eplb_config = EplbConfig(eplb_config)

        # Dump / PrecisionDebugger configuration
        self.dump_config_path = None
        self._construct_weight_prefetch_config(additional_config)
        self.layer_sharding = None
        logger.info_once(
            f"Linear layer sharding enabled with config: {self.layer_sharding}. "
            "Note: This feature works optimally with FLASHCOMM2 and DSA-CP enabled; "
            "using it without these features may result in significant performance degradation."
        )

        self.enable_shared_expert_dp = (
            additional_config.get("enable_shared_expert_dp", False)
            and vllm_config.parallel_config.enable_expert_parallel
            and vllm_config.parallel_config.tensor_parallel_size > 1
        )
        from .utils import enable_sp

        if self.enable_shared_expert_dp:
            assert enable_sp(vllm_config=vllm_config, enable_shared_expert_dp=True)

        if vllm_config.parallel_config.prefill_context_parallel_size > 1 and enable_sp(
            vllm_config=vllm_config
        ):
            tp_pcp_size = (
                vllm_config.parallel_config.tensor_parallel_size
                * vllm_config.parallel_config.prefill_context_parallel_size
            )
            if vllm_config.scheduler_config.max_num_batched_tokens % tp_pcp_size != 0:
                vllm_config.scheduler_config.max_num_batched_tokens = (
                    cdiv(
                        vllm_config.scheduler_config.max_num_batched_tokens, tp_pcp_size
                    )
                    * tp_pcp_size
                )
                logger.warning_once(
                    f"When using FLASHCOMM1, the max_num_batched_tokens should be divisible"
                    f"by tp_size * pcp_size ({tp_pcp_size}). It has been adjusted to"
                    f"{vllm_config.scheduler_config.max_num_batched_tokens}."
                )
        self.multistream_overlap_shared_expert = False
        self.multistream_overlap_gate = False
        self.recompute_scheduler_enable = False
        self.enable_cpu_binding = False

        self.pd_tp_ratio = 1
        self.pd_head_ratio = 1
        self.num_head_replica = 1

        self.SLO_limits_for_dynamic_batch = -1
        from .utils import get_flashcomm2_config_and_validate

        self.flashcomm2_oproj_tensor_parallel_size = get_flashcomm2_config_and_validate(
            self, vllm_config
        )

        # We find that _npu_paged_attention still performs better than
        # npu_fused_infer_attention_score in some cases. We allow to execute
        # _npu_paged_attention in this cases. This should be removed once
        # npu_fused_infer_attention_score performs better on all scenarios.
        self.pa_shape_list = additional_config.get("pa_shape_list", [])

        self.enable_async_exponential = False

        self.enable_kv_nz = False
        if self.enable_kv_nz:
            use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")
            if not vllm_config.model_config.is_deepseek_mla or use_sparse:
                raise RuntimeError("enable_kv_nz is only supported for mla currently.")
            if (
                vllm_config.kv_transfer_config is None
                or not vllm_config.kv_transfer_config.is_kv_consumer
            ):
                raise NotImplementedError(
                    "enable_kv_nz is only supported in pd scenario and can only be used in D node."
                )

    def _construct_weight_prefetch_config(self, additional_config):
        weight_prefetch_config = additional_config.get("weight_prefetch_config", {})
        self.weight_prefetch_config = WeightPrefetchConfig(weight_prefetch_config)



_ASCEND_CONFIG: AscendConfig | None = None


class FinegrainedTPConfig:
    """
    Configuration Object for finegrained_tp_config from additional_config
    """

    def __init__(self, finegrained_tp_config: dict, vllm_config):
        self.oproj_tensor_parallel_size = finegrained_tp_config.get(
            "oproj_tensor_parallel_size", 0
        )
        self.lmhead_tensor_parallel_size = finegrained_tp_config.get(
            "lmhead_tensor_parallel_size", 0
        )
        self.embedding_tensor_parallel_size = finegrained_tp_config.get(
            "embedding_tensor_parallel_size", 0
        )
        self.mlp_tensor_parallel_size = finegrained_tp_config.get(
            "mlp_tensor_parallel_size", 0
        )

        enabled_configs = []
        if self.oproj_tensor_parallel_size > 0:
            enabled_configs.append(
                f"oproj_tensor_parallel_size={self.oproj_tensor_parallel_size}"
            )
            # dummy_run does not run the entire attention module in eager mode,
            # so the o_proj tp split can only be used in graph mode.
            if vllm_config.model_config.enforce_eager is True:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in graph mode"
                )
            if (
                vllm_config.kv_transfer_config is None
                or not vllm_config.kv_transfer_config.is_kv_consumer
            ):
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
                )
        if self.lmhead_tensor_parallel_size > 0:
            enabled_configs.append(
                f"lmhead_tensor_parallel_size={self.lmhead_tensor_parallel_size}"
            )
        if self.embedding_tensor_parallel_size > 0:
            enabled_configs.append(
                f"embedding_tensor_parallel_size={self.embedding_tensor_parallel_size}"
            )
        if self.mlp_tensor_parallel_size > 0:
            enabled_configs.append(
                f"mlp_tensor_parallel_size={self.mlp_tensor_parallel_size}"
            )
        module_tp_sizes = [
            self.oproj_tensor_parallel_size,
            self.lmhead_tensor_parallel_size,
            self.embedding_tensor_parallel_size,
            self.mlp_tensor_parallel_size,
        ]
        for module_tp_size in module_tp_sizes:
            if (
                module_tp_size > 0
                and vllm_config.parallel_config.data_parallel_size % module_tp_size != 0
            ):
                raise AssertionError("module tp sizes must divide data_parallel_size")
        if any(size > 0 for size in module_tp_sizes) and enabled_configs:
            logger.info(f"finegrained_tp_config enabled: {', '.join(enabled_configs)}")


class WeightPrefetchConfig:
    """
    Configuration Object for weight_prefetch_config from additional_config
    """

    mlp_pre_version_compatibale_config: dict = {}

    prefetch_ratio: dict = {
        "attn": {
            "qkv": 1.0,
            "o": 1.0,
        },
        "moe": {"gate_up": 0.8},
        "mlp": {"gate_up": 1, "down": 1.0},
    }

    def __init__(self, weight_prefetch_config: dict):
        self.enabled = weight_prefetch_config.get("enabled", False)
        self.prefetch_ratio = weight_prefetch_config.get(
            "prefetch_ratio", self.prefetch_ratio
        )

    def set_mlp_pre_version_compatibale_config(
        self, gate_up_prefetch_size: int, down_prefetch_size: int
    ):
        config = {
            "gate_up": gate_up_prefetch_size,
            "down": down_prefetch_size,
        }
        self.mlp_pre_version_compatibale_config = config


class EplbConfig:
    """
    Configuration Object for xlite_graph_config from additional_config
    """

    _defaults = {
        "dynamic_eplb": False,
        "expert_map_path": None,
        "expert_heat_collection_interval": 400,
        "algorithm_execution_interval": 30,
        "expert_map_record_path": None,
        "num_redundant_experts": 0,
        "eplb_policy_type": 1,
    }

    def __init__(self, user_config: dict | None = None):
        if user_config is None:
            user_config = {}
        self.config = self._defaults.copy()
        if user_config and isinstance(user_config, dict):
            for key, value in user_config.items():
                if key in self.config:
                    self.config[key] = value
                else:
                    raise ValueError(f"Config has no attribute '{key}'")

        self._validate_config()

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def _validate_config(self):
        if self.expert_map_path is not None:
            if self.expert_map_path[-5:] != ".json":
                raise TypeError("The expert_map is not json.")
            if not os.path.exists(self.expert_map_path):
                raise ValueError("The expert_map is not exist.")
        if self.expert_map_record_path is not None:
            self.config["dynamic_eplb"] = True
            if self.expert_map_record_path[-5:] != ".json":
                raise TypeError("The expert_map_record_path is not json.")
            dirname = os.path.dirname(self.expert_map_record_path)
            os.makedirs(dirname, exist_ok=True)
        for key in [
            "expert_heat_collection_interval",
            "algorithm_execution_interval",
            "num_redundant_experts",
        ]:
            if not isinstance(self.config[key], int):
                raise TypeError(f"{key} must be an integer")
            if self.config[key] < 0:  # type: ignore
                raise ValueError(
                    f"{key} must greater than 0; got {self.config[key]} instead"
                )
        if self.eplb_policy_type not in [0, 1, 2, 3]:
            raise ValueError("eplb_policy_type must in [0, 1, 2, 3]")


def init_ascend_config(vllm_config):
    additional_config = (
        vllm_config.additional_config
        if vllm_config.additional_config is not None
        else {}
    )
    refresh = additional_config.get("refresh", False) if additional_config else False
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is not None and not refresh:
        return _ASCEND_CONFIG
    _ASCEND_CONFIG = AscendConfig(vllm_config)
    return _ASCEND_CONFIG


def clear_ascend_config():
    global _ASCEND_CONFIG
    _ASCEND_CONFIG = None


def get_ascend_config():
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is None:
        raise RuntimeError(
            "Ascend config is not initialized. Please call init_ascend_config first."
        )
    return _ASCEND_CONFIG

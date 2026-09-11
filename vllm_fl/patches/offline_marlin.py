# Copyright (c) 2026 BAAI. All rights reserved.
"""Load rank-sharded checkpoints whose MoE weights are already Marlin packed."""

import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)

_OFFLINE_MARLIN_ENV = "VLLM_FL_OFFLINE_MARLIN"


def offline_marlin_enabled() -> bool:
    return os.environ.get(_OFFLINE_MARLIN_ENV, "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _checkpoint_key_to_parameter(key: str) -> str:
    """Map processed Marlin aliases back to create_weights parameter names."""
    marker = ".routed_experts."
    if marker in key and key.endswith((".w13_weight", ".w2_weight")):
        return f"{key}_packed"
    return key


def _finalize_prepacked_marlin(method, layer) -> None:
    """Build the modular kernel without repacking checkpoint tensors again."""
    from vllm.model_executor.layers.fused_moe import FusedMoEExpertsModular
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        WNA16MoEBackend,
        make_wna16_moe_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    if method.wna16_backend not in (
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ):
        raise ValueError(
            f"{_OFFLINE_MARLIN_ENV}=1 requires a Marlin WNA16 backend, "
            f"got {method.wna16_backend.value}"
        )

    layer.w13_weight = layer.w13_weight_packed
    layer.w2_weight = layer.w2_weight_packed

    assert method.experts_cls is not None
    method.moe_quant_config = method.get_fused_moe_quant_config(layer)
    assert method.moe_quant_config is not None

    if issubclass(method.experts_cls, FusedMoEExpertsModular):
        layer.workspace = marlin_make_workspace_new(layer.w13_weight.device, 4)

    method.moe_kernel = make_wna16_moe_kernel(
        moe_quant_config=method.moe_quant_config,
        moe_config=method.moe,
        experts_cls=method.experts_cls,
        routing_tables=layer._expert_routing_tables(),
        w13_g_idx=layer.w13_weight_g_idx,
        w2_g_idx=layer.w2_weight_g_idx,
        w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
        w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
        is_k_full=method.is_k_full,
    )


def patch_offline_marlin_loader() -> bool:
    """Install opt-in sharded-loader and quant-method adapters."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (  # noqa: E501
        CompressedTensorsWNA16MarlinMoEMethod,
    )
    from vllm.model_executor.model_loader import ShardedStateLoader

    current_iter = ShardedStateLoader.iterate_over_files
    if getattr(current_iter, "_vllm_fl_offline_marlin", False):
        return False

    @wraps(current_iter)
    def iterate_over_files(self, paths):
        for key, tensor in current_iter(self, paths):
            if offline_marlin_enabled():
                key = _checkpoint_key_to_parameter(key)
            yield key, tensor

    iterate_over_files._vllm_fl_offline_marlin = True
    iterate_over_files._vllm_fl_original = current_iter
    ShardedStateLoader.iterate_over_files = iterate_over_files

    current_process = CompressedTensorsWNA16MarlinMoEMethod.process_weights_after_loading

    @wraps(current_process)
    def process_weights_after_loading(self, layer):
        if not offline_marlin_enabled():
            return current_process(self, layer)
        return _finalize_prepacked_marlin(self, layer)

    process_weights_after_loading._vllm_fl_offline_marlin = True
    process_weights_after_loading._vllm_fl_original = current_process
    CompressedTensorsWNA16MarlinMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    logger.info("Enabled opt-in offline Marlin sharded-state loader support")
    return True


__all__ = [
    "offline_marlin_enabled",
    "patch_offline_marlin_loader",
]

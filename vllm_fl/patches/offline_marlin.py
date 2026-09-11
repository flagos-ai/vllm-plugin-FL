# Copyright (c) 2026 BAAI. All rights reserved.
"""Restore rank-local MoE tensors from an offline Marlin sidecar."""

import glob
import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)

_OFFLINE_MARLIN_ENV = "VLLM_FL_OFFLINE_MARLIN"
_OFFLINE_MARLIN_PATH_ENV = "VLLM_FL_OFFLINE_MARLIN_PATH"
_KEY_INDEX: dict[tuple[str, int], dict[str, str]] = {}


def offline_marlin_enabled() -> bool:
    return os.environ.get(_OFFLINE_MARLIN_ENV, "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _rank_key_index(path: str, rank: int) -> dict[str, str]:
    cache_key = (path, rank)
    if cache_key in _KEY_INDEX:
        return _KEY_INDEX[cache_key]

    from safetensors import safe_open

    files = sorted(
        glob.glob(os.path.join(path, f"model-rank-{rank}-part-*.safetensors"))
    )
    if not files:
        raise FileNotFoundError(
            f"No offline Marlin files for rank {rank} under {path}"
        )
    index: dict[str, str] = {}
    for filename in files:
        with safe_open(filename, framework="pt") as handle:
            index.update((key, filename) for key in handle.keys())
    _KEY_INDEX[cache_key] = index
    return index


def _layer_checkpoint_prefix(layer_name: str, index: dict[str, str]) -> str:
    candidates = (layer_name, f"model.{layer_name}")
    for prefix in candidates:
        if f"{prefix}.w13_weight" in index:
            return prefix
    raise KeyError(f"No offline Marlin MoE tensors found for layer {layer_name}")


def _restore_prepacked_marlin(method, layer) -> None:
    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank

    path = os.environ.get(_OFFLINE_MARLIN_PATH_ENV)
    if not path:
        raise ValueError(
            f"{_OFFLINE_MARLIN_PATH_ENV} must point to the TP/EP sharded "
            "Marlin sidecar"
        )
    layer_name = getattr(method, "_vllm_fl_layer_name", None)
    if not layer_name:
        raise ValueError("Missing layer name for offline Marlin MoE restore")

    rank = get_tensor_model_parallel_rank()
    index = _rank_key_index(path, rank)
    prefix = _layer_checkpoint_prefix(layer_name, index)
    attrs = {
        "w13_weight": "w13_weight_packed",
        "w2_weight": "w2_weight_packed",
        "w13_weight_scale": "w13_weight_scale",
        "w2_weight_scale": "w2_weight_scale",
    }
    for saved_name, attr_name in attrs.items():
        key = f"{prefix}.{saved_name}"
        target = getattr(layer, attr_name)
        with safe_open(index[key], framework="pt") as handle:
            source = handle.get_tensor(key)
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(
                f"Offline Marlin tensor {key} has {source.shape}/{source.dtype}, "
                f"expected {target.shape}/{target.dtype}"
            )
        target.copy_(source.to(device=target.device, non_blocking=False))


def patch_offline_marlin_loader() -> bool:
    """Make WNA16 MoE consume a rank-local, already-Marlin-packed sidecar."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (  # noqa: E501
        CompressedTensorsWNA16MarlinMoEMethod,
    )
    from vllm.model_executor.model_loader import base_loader

    current_process = CompressedTensorsWNA16MarlinMoEMethod.process_weights_after_loading
    if getattr(current_process, "_vllm_fl_offline_marlin", False):
        return False

    current_init = CompressedTensorsWNA16MarlinMoEMethod.__init__

    @wraps(current_init)
    def init(self, *args, **kwargs):
        current_init(self, *args, **kwargs)
        layer_name = kwargs.get("layer_name")
        if layer_name is None and len(args) >= 4:
            layer_name = args[3]
        self._vllm_fl_layer_name = layer_name

    @wraps(current_process)
    def process_weights_after_loading(self, layer):
        result = current_process(self, layer)
        if offline_marlin_enabled():
            _restore_prepacked_marlin(self, layer)
        return result

    process_weights_after_loading._vllm_fl_offline_marlin = True
    process_weights_after_loading._vllm_fl_original = current_process
    CompressedTensorsWNA16MarlinMoEMethod.__init__ = init
    CompressedTensorsWNA16MarlinMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )

    current_model_process = base_loader.process_weights_after_loading

    @wraps(current_model_process)
    def process_model_weights(model, model_config, target_device):
        if offline_marlin_enabled():
            for name, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if isinstance(quant_method, CompressedTensorsWNA16MarlinMoEMethod):
                    quant_method._vllm_fl_layer_name = name
        return current_model_process(model, model_config, target_device)

    process_model_weights._vllm_fl_offline_marlin = True
    process_model_weights._vllm_fl_original = current_model_process
    base_loader.process_weights_after_loading = process_model_weights
    logger.info("Enabled opt-in offline Marlin MoE sidecar loading")
    return True


__all__ = ["offline_marlin_enabled", "patch_offline_marlin_loader"]

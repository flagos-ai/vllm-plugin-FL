# Copyright (c) 2026 BAAI. All rights reserved.
"""Restore rank-local MoE tensors from an offline Marlin sidecar."""

import glob
import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)

_OFFLINE_MARLIN_ENV = "VLLM_FL_OFFLINE_MARLIN"
_OFFLINE_MARLIN_PATH_ENV = "VLLM_FL_OFFLINE_MARLIN_PATH"
_FLAGGEMS_FUSED_MARLIN_ENV = "VLLM_FL_FLAGGEMS_FUSED_MARLIN"
_FLAGGEMS_FUSED_MARLIN_PATH_ENV = "VLLM_FL_FLAGGEMS_FUSED_MARLIN_PATH"
_KEY_INDEX: dict[tuple[str, int], dict[str, str]] = {}
_FLAGGEMS_KEY_INDEX: dict[tuple[str, int], dict[str, str]] = {}
_FLAGGEMS_TENSOR_NAMES = (
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
)


def offline_marlin_enabled() -> bool:
    return os.environ.get(_OFFLINE_MARLIN_ENV, "0").lower() in (
        "1",
        "true",
        "yes",
    )


def flaggems_fused_marlin_enabled() -> bool:
    """Use PR #5437's GPTQ-packed input contract for Marlin MoE."""
    return os.environ.get(_FLAGGEMS_FUSED_MARLIN_ENV, "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _prepare_flaggems_packed_tensors(layer) -> dict[str, object]:
    """Convert vLLM's pre-repack tensors to FlagGems' packed layout.

    vLLM creates transposed int32 checkpoint parameters for its Marlin
    repacker.  FlagGems PR #5437 consumes the same nibbles as uint8 tensors in
    output-major order.  Preserve that representation before vLLM replaces
    the parameters with its CUDA-Marlin layout.
    """
    import torch

    def packed_u8(value):
        return value.transpose(1, 2).contiguous().view(torch.uint8)

    return {
        "w13_weight_packed": packed_u8(layer.w13_weight_packed),
        "w2_weight_packed": packed_u8(layer.w2_weight_packed),
        "w13_weight_scale": layer.w13_weight_scale.transpose(1, 2).contiguous(),
        "w2_weight_scale": layer.w2_weight_scale.transpose(1, 2).contiguous(),
    }


def _install_flaggems_packed_tensors(method, layer, tensors=None) -> None:
    from vllm.model_executor.layers.quantization.utils import replace_parameter

    if tensors is None:
        tensors = getattr(method, "_vllm_fl_flaggems_packed_tensors", None)
    if tensors is None:
        raise ValueError("Missing pre-repack tensors for FlagGems fused Marlin MoE")
    for name, tensor in tensors.items():
        replace_parameter(layer, name, tensor)
    layer.w13_weight = layer.w13_weight_packed
    layer.w2_weight = layer.w2_weight_packed

    # PR #5437 does not implement activation-order metadata or quantized
    # activation input.  DSV4's symmetric group-128 checkpoint uses neither;
    # vLLM still materializes empty Marlin metadata, so remove it from the
    # already-created expert object.
    experts = method.moe_kernel.fused_experts
    experts.w13_g_idx = None
    experts.w2_g_idx = None
    experts.w13_g_idx_sort_indices = None
    experts.w2_g_idx_sort_indices = None
    experts.input_dtype = None


def _flaggems_rank_files(path: str, rank: int) -> list[str]:
    return sorted(
        glob.glob(os.path.join(path, f"model-rank-{rank}-part-*.safetensors"))
    )


def _flaggems_key_index(path: str, rank: int) -> dict[str, str]:
    cache_key = (path, rank)
    if cache_key in _FLAGGEMS_KEY_INDEX:
        return _FLAGGEMS_KEY_INDEX[cache_key]

    from safetensors import safe_open

    index: dict[str, str] = {}
    for filename in _flaggems_rank_files(path, rank):
        with safe_open(filename, framework="pt") as handle:
            index.update((key, filename) for key in handle.keys())
    _FLAGGEMS_KEY_INDEX[cache_key] = index
    return index


def _flaggems_sidecar_complete(model, method_type, path: str, rank: int) -> bool:
    index = _flaggems_key_index(path, rank)
    modules = [
        name
        for name, module in model.named_modules()
        if isinstance(getattr(module, "quant_method", None), method_type)
    ]
    return bool(modules) and all(
        f"{name}.{tensor_name}" in index
        for name in modules
        for tensor_name in _FLAGGEMS_TENSOR_NAMES
    )


def _load_flaggems_packed_tensors(method, layer) -> dict[str, object]:
    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank

    path = os.environ.get(_FLAGGEMS_FUSED_MARLIN_PATH_ENV)
    if not path:
        raise ValueError(f"{_FLAGGEMS_FUSED_MARLIN_PATH_ENV} is not configured")
    layer_name = getattr(method, "_vllm_fl_layer_name", None)
    if not layer_name:
        raise ValueError("Missing layer name for FlagGems Marlin sidecar restore")

    rank = get_tensor_model_parallel_rank()
    index = _flaggems_key_index(path, rank)
    device = layer.w13_weight_packed.device
    tensors = {}
    for tensor_name in _FLAGGEMS_TENSOR_NAMES:
        key = f"{layer_name}.{tensor_name}"
        filename = index.get(key)
        if filename is None:
            raise KeyError(f"Missing FlagGems Marlin sidecar tensor {key}")
        with safe_open(filename, framework="pt") as handle:
            tensor = handle.get_tensor(key)
        tensors[tensor_name] = tensor.to(device=device, non_blocking=False)
    return tensors


def _export_flaggems_packed_tensors(model, method_type, path: str, rank: int) -> None:
    """Persist one rank's FlagGems layout in bounded, layer-sized files."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    part = 0
    for name, module in model.named_modules():
        if not isinstance(getattr(module, "quant_method", None), method_type):
            continue
        tensors = {
            f"{name}.{tensor_name}": getattr(module, tensor_name)
            .detach()
            .to(device="cpu")
            .contiguous()
            for tensor_name in _FLAGGEMS_TENSOR_NAMES
        }
        filename = os.path.join(
            path, f"model-rank-{rank}-part-{part:05d}.safetensors"
        )
        temporary = f"{filename}.tmp"
        save_file(tensors, temporary, metadata={"format": "flaggems-fused-marlin-v1"})
        os.replace(temporary, filename)
        part += 1
    _FLAGGEMS_KEY_INDEX.pop((path, rank), None)
    logger.info(
        "Exported %d FlagGems fused-Marlin MoE layers for rank %d to %s",
        part,
        rank,
        path,
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
        if flaggems_fused_marlin_enabled():
            if getattr(self, "_vllm_fl_flaggems_sidecar_ready", False):
                tensors = _load_flaggems_packed_tensors(self, layer)
            else:
                tensors = None
            _install_flaggems_packed_tensors(self, layer, tensors)
        elif offline_marlin_enabled():
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
        use_flaggems_layout = flaggems_fused_marlin_enabled()
        flaggems_path = os.environ.get(_FLAGGEMS_FUSED_MARLIN_PATH_ENV)
        flaggems_sidecar_ready = False
        rank = None
        if use_flaggems_layout and flaggems_path:
            from vllm.distributed import get_tensor_model_parallel_rank

            rank = get_tensor_model_parallel_rank()
            flaggems_sidecar_ready = _flaggems_sidecar_complete(
                model,
                CompressedTensorsWNA16MarlinMoEMethod,
                flaggems_path,
                rank,
            )
        if offline_marlin_enabled() or use_flaggems_layout:
            for name, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if isinstance(quant_method, CompressedTensorsWNA16MarlinMoEMethod):
                    quant_method._vllm_fl_layer_name = name
                    quant_method._vllm_fl_flaggems_sidecar_ready = (
                        flaggems_sidecar_ready
                    )
                    if use_flaggems_layout and not flaggems_sidecar_ready:
                        quant_method._vllm_fl_flaggems_packed_tensors = (
                            _prepare_flaggems_packed_tensors(module)
                        )
        result = current_model_process(model, model_config, target_device)
        if use_flaggems_layout and flaggems_path and not flaggems_sidecar_ready:
            _export_flaggems_packed_tensors(
                model,
                CompressedTensorsWNA16MarlinMoEMethod,
                flaggems_path,
                rank,
            )
        return result

    process_model_weights._vllm_fl_offline_marlin = True
    process_model_weights._vllm_fl_original = current_model_process
    base_loader.process_weights_after_loading = process_model_weights
    logger.info("Enabled opt-in offline Marlin MoE sidecar loading")
    return True


__all__ = [
    "flaggems_fused_marlin_enabled",
    "offline_marlin_enabled",
    "patch_offline_marlin_loader",
]

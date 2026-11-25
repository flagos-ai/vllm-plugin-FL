# Copyright (c) 2026 BAAI. All rights reserved.
"""Load GLM channel-INT8 WK into vLLM 0.24's fused BF16 indexer.

Checkpoint files stay unchanged. Conversion happens only during loading, not
during forward/Graph capture. Other models and FP8/BF16 weights pass through.
"""

from functools import wraps
import logging

import torch

logger = logging.getLogger(__name__)


def _convert_int8_wk(weights, quantization_config):
    pending = {}
    for name, tensor in weights:
        if ".indexer.wk." not in name:
            yield name, tensor
            continue
        prefix, suffix = name.rsplit(".wk.", 1)
        is_weight = suffix == "weight" and tensor.dtype in (torch.int8, torch.float8_e4m3fn)
        is_scale = suffix == "weight_scale"
        if not (is_weight or is_scale):
            yield name, tensor
            continue

        # Fail closed: only the verified symmetric per-output-channel format.
        groups = (quantization_config or {}).get("config_groups", {})
        specs = [group.get("weights", {}) for group in groups.values()]
        if len(specs) != 1 or specs[0].get("type") not in ("int", "float") or not all(
            specs[0].get(key) == value
            for key, value in {
                "num_bits": 8, "symmetric": True,
                "strategy": "channel",
            }.items()
        ):
            raise ValueError("GLM WK adapter requires symmetric channel INT8")
        entry = pending.setdefault(prefix, {})
        key = "weight" if is_weight else "scale"
        if key in entry:
            raise ValueError(f"Duplicate GLM WK {key}: {prefix}")
        entry[key] = tensor
        if len(entry) < 2:
            continue
        weight, scale = entry["weight"], entry["scale"]
        expected_dtype = torch.int8 if specs[0]["type"] == "int" else torch.float8_e4m3fn
        if weight.dtype != expected_dtype:
            raise ValueError("GLM WK dtype does not match the declared channel quantization")
        if weight.ndim != 2 or tuple(scale.shape) != (weight.shape[0], 1):
            raise ValueError(f"Invalid GLM WK weight/scale shapes: {prefix}")
        if not scale.is_floating_point() or not torch.isfinite(scale).all():
            raise ValueError(f"Invalid GLM WK scale dtype/values: {prefix}")
        if not (scale > 0).all():
            raise ValueError(f"GLM WK scale must be positive: {prefix}")
        if weight.device != scale.device:
            raise ValueError(f"GLM WK weight/scale device mismatch: {prefix}")
        converted = (weight.float() * scale.float()).to(torch.bfloat16)
        if not torch.isfinite(converted).all():
            raise ValueError(f"GLM WK dequantization overflow: {prefix}")
        del pending[prefix]
        yield f"{prefix}.wk.weight", converted
    if pending:
        raise ValueError(f"Incomplete GLM INT8 WK weight/scale pairs: {sorted(pending)}")


def _install_glm_int8_indexer_loader(model_class):
    original = model_class.load_weights
    if getattr(original, "_fl_glm_int8_wk", False):
        return

    @wraps(original)
    def load_weights(self, weights):
        if getattr(self.config, "model_type", None) == "glm_moe_dsa":
            weights = _convert_int8_wk(
                weights, getattr(self.config, "quantization_config", None) or getattr(self.config, "compression_config", None)
            )
        return original(self, weights)

    load_weights._fl_glm_int8_wk = True
    model_class.load_weights = load_weights


def patch_glm_int8_indexer_loader():
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

    _install_glm_int8_indexer_loader(DeepseekV2Model)
    logger.info("Patched GLM symmetric INT8 WK loading for fused BF16 indexer")

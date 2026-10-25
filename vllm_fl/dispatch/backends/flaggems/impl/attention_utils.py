# Copyright 2026 FlagOS Contributors

from vllm.platforms import current_platform

import vllm_fl.envs as fl_envs


def attention_descales(layer, kv_cache_dtype, shape=None):
    """Omit unused scales for the release MetaX unquantized FA2 contract."""
    if (
        fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM
        and getattr(current_platform, "vendor_name", None) in ("metax", "maca")
        and kv_cache_dtype == "auto"
    ):
        return None, None, None
    scales = (layer._q_scale, layer._k_scale, layer._v_scale)
    return tuple(scale.expand(shape) for scale in scales) if shape else scales

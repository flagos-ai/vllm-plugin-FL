# Copyright (c) 2026 BAAI. All rights reserved.
"""BGE-M3 provider boundaries; compute kernels stay in FG/FGV."""

from functools import partial
from importlib import import_module

import vllm_fl.envs as fl_envs
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _varlen():
    return import_module("flaggems_vllm.ops.attention").flash_attn_varlen_func


def encoder_attention(query, key, value, **kwargs):
    if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        raise RuntimeError("BGE-M3 encoder attention requires FlagGems-vllm")
    return _varlen()(query, key, value, **kwargs)


def linear(native_apply, layer, x, bias=None):
    """Default policy keeps the original layer method and active FG dispatch."""
    return native_apply(layer, x, bias)


def register_bge_m3_operators(registry, *, allow):
    registry.register_many(
        [
            OpImpl(
                "bge_m3_linear",
                "default.flagos",
                BackendImplKind.DEFAULT,
                linear,
                priority=BackendPriority.DEFAULT,
            )
        ]
    )
    if not allow("bge_m3_encoder_attention"):
        return

    def available():
        return bool(fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM) and callable(_varlen())

    selected_provider = partial(encoder_attention)
    selected_provider._is_available = available
    registry.register_many(
        [
            OpImpl(
                "bge_m3_encoder_attention",
                "default.flagos",
                BackendImplKind.DEFAULT,
                selected_provider,
                priority=BackendPriority.DEFAULT,
            )
        ]
    )

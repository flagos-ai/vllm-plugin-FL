# Copyright (c) 2026 BAAI. All rights reserved.
"""Bound Qwen vision TP reduction rounding without changing decoder collectives."""
from contextvars import ContextVar
from functools import wraps

_vision_forward = ContextVar('iluvatar_vision_forward', default=False)


def in_vision_forward() -> bool:
    return _vision_forward.get()


def _wrap_vision_forward(forward):
    if getattr(forward, '_iluvatar_fp32_reduce', False):
        return forward

    @wraps(forward)
    def wrapped(*args, **kwargs):
        token = _vision_forward.set(True)
        try:
            return forward(*args, **kwargs)
        finally:
            _vision_forward.reset(token)

    wrapped._iluvatar_fp32_reduce = True
    return wrapped


def patch_vision_precision() -> None:
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    Qwen3_VisionTransformer.forward = _wrap_vision_forward(
        Qwen3_VisionTransformer.forward
    )

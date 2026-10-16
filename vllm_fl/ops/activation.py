# Copyright (c) 2025 BAAI. All rights reserved.

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.activation import (
    GeluAndMul,
    SiluAndMul,
    SiluAndMulWithClamp,
)

from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")
_silu_and_mul_with_clamp = CachedOp("silu_and_mul_with_clamp")


@torch.library.custom_op("vllm_fl::silu_and_mul", mutates_args=())
def _silu_and_mul_graph(x: torch.Tensor) -> torch.Tensor:
    return _silu_and_mul(None, x)


@_silu_and_mul_graph.register_fake
def _silu_and_mul_graph_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


@torch.library.custom_op("vllm_fl::gelu_and_mul", mutates_args=())
def _gelu_and_mul_graph(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return _gelu_and_mul(SimpleNamespace(approximate=approximate), x)


@_gelu_and_mul_graph.register_fake
def _gelu_and_mul_graph_fake(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


@torch.library.custom_op("vllm_fl::silu_and_mul_with_clamp", mutates_args=())
def _silu_and_mul_with_clamp_graph(x: torch.Tensor, limit: float) -> torch.Tensor:
    obj = SimpleNamespace(swiglu_limit=limit, alpha=1.0, beta=0.0)
    return _silu_and_mul_with_clamp(obj, x)


@_silu_and_mul_with_clamp_graph.register_fake
def _silu_and_mul_with_clamp_graph_fake(x: torch.Tensor, limit: float) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def enable_fused_activations_for_compile(compilation_config) -> None:
    """Keep selected OOT activations opaque to Inductor and CUDA graph capture."""
    import vllm_fl.envs as fl_envs
    from vllm_fl.utils import (
        get_oot_blacklist,
        get_oot_whitelist,
        is_oot_enabled,
        use_flaggems_op,
    )

    if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM or not is_oot_enabled():
        return
    whitelist, blacklist = get_oot_whitelist(), get_oot_blacklist() or []
    for op, name in (
        ("silu_and_mul", "SiluAndMul"),
        ("gelu_and_mul", "GeluAndMul"),
        ("silu_and_mul_with_clamp", "SiluAndMulWithClamp"),
    ):
        if not use_flaggems_op(op) or op in blacklist:
            continue
        if whitelist is not None and op not in whitelist:
            continue
        if any(f"-{alias}" in compilation_config.custom_ops for alias in (op, name)):
            continue
        selected = f"+{op if op == 'silu_and_mul_with_clamp' else name}"
        if selected not in compilation_config.custom_ops:
            compilation_config.custom_ops.append(selected)


class SiluAndMulFL(SiluAndMul):
    def __init__(self):
        super().__init__()

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return _silu_and_mul_graph(x)
        return _silu_and_mul(self, x)


class GeluAndMulFL(GeluAndMul):
    def __init__(self, approximate: str = "none"):
        super().__init__(approximate=approximate)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return _gelu_and_mul_graph(x, self.approximate)
        return _gelu_and_mul(self, x)


__all__ = ["SiluAndMulFL", "GeluAndMulFL", "SiluAndMulWithClampFL"]


class SiluAndMulWithClampFL(SiluAndMulWithClamp):
    """Dispatch the standard clamp; preserve the native parameterized formula."""

    @classmethod
    def enabled(cls) -> bool:
        from vllm.config import get_cached_compilation_config

        controls = get_cached_compilation_config().custom_ops
        names = ("silu_and_mul_with_clamp", "SiluAndMulWithClamp")
        enabled = any(f"+{name}" in controls for name in names)
        disabled = any(f"-{name}" in controls for name in names)
        assert not (enabled and disabled), "Cannot enable and disable clamped SwiGLU"
        return (cls.default_on() or enabled) and not disabled

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # v0.2.0 only implements alpha=1,beta=0. Reject that capability before
        # entering the fused provider, rather than falling back after execution.
        if self.alpha != 1.0 or self.beta != 0.0:
            return self.forward_native(x)
        if torch.compiler.is_compiling():
            return _silu_and_mul_with_clamp_graph(x, self.swiglu_limit)
        return _silu_and_mul_with_clamp(self, x)

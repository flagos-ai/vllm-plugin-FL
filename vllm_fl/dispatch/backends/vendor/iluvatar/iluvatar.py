# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar CoreX (`ivcore11`) thin vendor backend and FlagTree/Inductor target adapter.

Registers `IluvatarAttentionBackend` onto `AttentionBackendEnum.TRITON_ATTN`,
adapts TorchInductor's hardcoded `"cuda"` `GPUTarget` to FlagTree's `"corex"`
compiler backend, delegates core elementwise/norm/rotary ops to `FlagGems`, and
activates declarative runtime hooks in `vendor.iluvatar.patches`.
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional, Union

import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_fl.dispatch.backends.base import Backend

logger = logging.getLogger(__name__)


def register_attention_backends() -> None:
    """Bind `AttentionBackendEnum.TRITON_ATTN` to `IluvatarAttentionBackend`."""
    register_backend(
        AttentionBackendEnum.TRITON_ATTN,
        class_path="vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention.IluvatarAttentionBackend",
    )


def patch_triton_chained_or_for_iluvatar() -> None:
    """No-op compatibility stub retained for `PlatformFL.pre_register_and_update` (`FlagTree >= 0.7 / Triton >= 3.6` supports chained boolean `or` natively)."""
    return None


def patch_sampler_compile_for_iluvatar() -> None:
    """Unwrap `@torch.compile` decorators on stock vLLM sampler helpers when targeting FlagTree CoreX."""
    for mod_name, attr_name in (
        ("vllm.v1.sample.ops.topk_topp_sampler", "compiled_random_sample"),
        ("vllm.v1.sample.ops.logprobs", "batched_count_greater_than"),
    ):
        try:
            mod = importlib.import_module(mod_name)
            if not getattr(mod, "_iluvatar_compile_patched", False):
                orig = getattr(mod, attr_name)
                setattr(mod, attr_name, getattr(orig, "__wrapped__", orig))
                setattr(mod, "_iluvatar_compile_patched", True)
        except Exception as exc:
            logger.warning("patch_sampler_compile_for_iluvatar (%s): %s", mod_name, exc)


def patch_torch_inductor_for_iluvatar(_bind_worker_hooks: bool = True) -> None:
    """Remap `torch._inductor` `GPUTarget('cuda', ...)` to FlagTree `'corex'` target and optionally bind worker hooks.

    PyTorch on Iluvatar exposes `device.type == 'cuda'`, causing TorchInductor's
    `triton_heuristics` to instantiate `GPUTarget('cuda', ...)`, whereas FlagTree
    registers its Triton compiler backend under `'corex'` (or `'iluvatar'`).
    Probes registered `triton.backends` once and translates `'cuda'` to the
    active FlagTree target name transparently.
    """
    if _bind_worker_hooks:
        try:
            from .patches import install_worker_runtime_hooks

            install_worker_runtime_hooks()
        except Exception as exc:
            logger.warning("install_worker_runtime_hooks: %s", exc)

    try:
        import triton.backends as tb
        from triton.backends.compiler import GPUTarget as OrigGPUTarget

        registered = list(getattr(tb, "backends", {}).keys())
        if "cuda" in registered or not registered:
            return
        target = None
        for probe in ("cuda", "corex", "iluvatar") + tuple(registered):
            try:
                t = object.__new__(OrigGPUTarget)
                OrigGPUTarget.__init__(t, probe, 90, False)
                if any(tb.backends[b].compiler.supports_target(t) for b in registered):
                    target = probe
                    break
            except Exception:
                continue
        target = target or "corex"
        if target == "cuda":
            return

        import torch._inductor.runtime.triton_heuristics as th

        if getattr(th, "_iluvatar_gputarget_patched", False):
            return

        class _IluvatarGPUTarget(OrigGPUTarget):
            """GPUTarget proxy rewriting TorchInductor's hardcoded `'cuda'` backend to FlagTree's active target."""

            def __new__(cls, backend, *args, **kwargs):
                return super().__new__(cls)

            def __init__(self, backend, *args, **kwargs):
                super().__init__(target if backend == "cuda" else backend, *args, **kwargs)

        th.GPUTarget = _IluvatarGPUTarget
        th._iluvatar_gputarget_patched = True
    except Exception as exc:
        logger.warning("Failed to patch torch._inductor for iluvatar: %s", exc)


# Activate attention registration, sampler unwrap, Inductor target adapter, and runtime hooks at import time.
register_attention_backends()
patch_sampler_compile_for_iluvatar()
patch_torch_inductor_for_iluvatar(_bind_worker_hooks=False)
import vllm_fl.dispatch.backends.vendor.iluvatar.patches  # noqa: E402, F401


class IluvatarBackend(Backend):
    """Iluvatar vendor backend for operator dispatch."""

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "iluvatar"

    @property
    def vendor(self) -> Optional[str]:
        return "iluvatar"

    def is_available(self) -> bool:
        if IluvatarBackend._available is None:
            try:
                from vllm.platforms import current_platform

                if getattr(current_platform, "vendor_name", None) == "iluvatar":
                    IluvatarBackend._available = True
                elif torch.cuda.is_available():
                    IluvatarBackend._available = "iluvatar" in torch.cuda.get_device_name(0).lower()
                else:
                    IluvatarBackend._available = False
            except Exception:
                IluvatarBackend._available = False
        return IluvatarBackend._available

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """Dispatch SiLU-and-Mul activation to FlagGems via `silu_and_mul_iluvatar`."""
        from .impl.activation import silu_and_mul_iluvatar

        return silu_and_mul_iluvatar(obj, x)

    def rms_norm(self, obj, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Dispatch standalone or residual-fused RMSNorm to FlagGems via `rms_norm_iluvatar`."""
        from .impl.normalization import rms_norm_iluvatar

        return rms_norm_iluvatar(obj, x, residual)

    def rotary_embedding(
        self, obj, query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        position_ids: torch.Tensor, rotary_interleaved: bool = False, inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch NeoX or interleaved rotary embedding to FlagGems via `rotary_embedding_iluvatar`."""
        from .impl.rotary import rotary_embedding_iluvatar

        return rotary_embedding_iluvatar(obj, query, key, cos, sin, position_ids, rotary_interleaved=rotary_interleaved, inplace=inplace)

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """Return the fully qualified attention backend class path (`TRITON_ATTN` or MLA)."""
        register_attention_backends()
        if use_mla:
            return AttentionBackendEnum.FLASHMLA_SPARSE.get_path() if use_sparse else AttentionBackendEnum.FLASHMLA.get_path()
        return AttentionBackendEnum.TRITON_ATTN.get_path()

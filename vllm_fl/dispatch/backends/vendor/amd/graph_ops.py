# Copyright (c) 2026 BAAI. All rights reserved.

"""
AMD graph-mode custom op registrations for torch.compile / CUDA-graph capture.

Dynamo cannot inline FlagGems kernel bodies (they contain ``logging.debug()``
and other non-traceable Python), so we register each operator as an opaque
``torch.library.custom_op`` with a ``register_fake`` meta implementation that
only propagates shapes.  The ``is_compiling()`` bypass paths in
``vllm_fl/ops/{layernorm,activation,rotary_embedding}.py`` call these wrappers
instead of going through the dispatch stack during tracing.

Library namespaces are AMD-specific to avoid collisions with other vendor
backends that may register the same op names under their own namespaces.
"""

from __future__ import annotations

import torch
import torch.library

# ---------------------------------------------------------------------------
# rms_norm
# Split into two ops because gems_rms_forward returns a bare Tensor when
# residual=None but a (Tensor, Tensor) tuple otherwise; a single custom op
# cannot have a polymorphic return type.
# ---------------------------------------------------------------------------

_rms_norm_lib = torch.library.Library("vllm_fl_amd", "DEF")
_rms_norm_lib.define("rms_norm_nr(Tensor x, Tensor weight, float epsilon) -> Tensor")
_rms_norm_lib.define(
    "rms_norm_wr(Tensor x, Tensor residual, Tensor weight, float epsilon)"
    " -> (Tensor, Tensor)"
)


@torch.library.impl(_rms_norm_lib, "rms_norm_nr", "CUDA")
def _rms_norm_nr_cuda(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    from flag_gems.modules.normalization import gems_rms_forward

    result = gems_rms_forward(x, None, weight, epsilon)
    # gems_rms_forward may return a tuple when residual=None on some versions.
    return result[0] if isinstance(result, tuple) else result


@torch.library.register_fake("vllm_fl_amd::rms_norm_nr")
def _rms_norm_nr_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    return x.clone()


@torch.library.impl(_rms_norm_lib, "rms_norm_wr", "CUDA")
def _rms_norm_wr_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flag_gems.modules.normalization import gems_rms_forward

    result = gems_rms_forward(x, residual, weight, epsilon)
    if isinstance(result, tuple):
        return result
    return (result, residual)


@torch.library.register_fake("vllm_fl_amd::rms_norm_wr")
def _rms_norm_wr_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.clone(), residual.clone()


def rms_norm_custom_op(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compile-safe rms_norm call site.  Returns ``(output, residual_or_None)``
    so callers can unpack uniformly regardless of whether residual was passed."""
    if residual is None:
        return (torch.ops.vllm_fl_amd.rms_norm_nr(x, weight, epsilon), None)
    return torch.ops.vllm_fl_amd.rms_norm_wr(x, residual, weight, epsilon)


# ---------------------------------------------------------------------------
# silu_and_mul / gelu_and_mul
# ---------------------------------------------------------------------------

_act_lib = torch.library.Library("vllm_fl_amd_act", "DEF")
_act_lib.define("silu_and_mul(Tensor x) -> Tensor")
_act_lib.define("gelu_and_mul(Tensor x, str approximate) -> Tensor")


@torch.library.impl(_act_lib, "silu_and_mul", "CUDA")
def _silu_cuda(x: torch.Tensor) -> torch.Tensor:
    from flaggems_vllm import silu_and_mul as gems_silu_and_mul

    d = x.shape[-1] // 2
    return gems_silu_and_mul(x[..., :d], x[..., d:])


@torch.library.register_fake("vllm_fl_amd_act::silu_and_mul")
def _silu_fake(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return x[..., :d].clone()


@torch.library.impl(_act_lib, "gelu_and_mul", "CUDA")
def _gelu_cuda(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    from flaggems_vllm import gelu_and_mul

    d = x.shape[-1] // 2
    return gelu_and_mul(x[..., :d], x[..., d:], approximate)


@torch.library.register_fake("vllm_fl_amd_act::gelu_and_mul")
def _gelu_fake(x: torch.Tensor, approximate: str) -> torch.Tensor:
    d = x.shape[-1] // 2
    return x[..., :d].clone()


def silu_and_mul_custom_op(x: torch.Tensor) -> torch.Tensor:
    """Compile-safe silu_and_mul call site."""
    return torch.ops.vllm_fl_amd_act.silu_and_mul(x)


def gelu_and_mul_custom_op(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Compile-safe gelu_and_mul call site."""
    return torch.ops.vllm_fl_amd_act.gelu_and_mul(x, approximate)


# ---------------------------------------------------------------------------
# rotary_embedding
# ---------------------------------------------------------------------------

_rot_lib = torch.library.Library("vllm_fl_amd_rot", "DEF")
_rot_lib.define(
    "rotary_embedding("
    "Tensor query, Tensor key, Tensor cos, Tensor sin, Tensor position_ids,"
    " bool rotary_interleaved) -> (Tensor, Tensor)"
)


@torch.library.impl(_rot_lib, "rotary_embedding", "CUDA")
def _rot_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.dispatch.backends.flaggems.impl.rotary import rotary_embedding_flaggems

    return rotary_embedding_flaggems(
        None,
        query.clone(memory_format=torch.contiguous_format),
        key.clone(memory_format=torch.contiguous_format),
        cos,
        sin,
        position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=True,
    )


@torch.library.register_fake("vllm_fl_amd_rot::rotary_embedding")
def _rot_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )


def rotary_embedding_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh rotary outputs from the opaque AMD kernel."""
    return torch.ops.vllm_fl_amd_rot.rotary_embedding(
        query, key, cos, sin, position_ids, rotary_interleaved
    )

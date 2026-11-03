# Copyright (c) 2026 BAAI. All rights reserved.
"""Opaque inference boundaries for compiled MUSA FlagGems operators.

The runtime kernels are the same FlagGems implementations used by eager FL.
Dynamo only sees tensor schemas and fake metadata, not device guards or dtype
promotion inside the Python kernel launcher.
"""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Keep the library alive, with a private namespace and native MUSA dispatch key.
_LIB = torch.library.Library("vllm_fl_musa", "FRAGMENT")


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    from flag_gems.modules.normalization import gems_rms_forward

    return gems_rms_forward(x, None, weight, epsilon)


def _rms_norm_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    # FlagGems makes x contiguous before allocating its fresh output.
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    from flag_gems.modules.normalization import gems_rms_forward

    gems_rms_forward(x, residual, weight, epsilon)


def _fused_add_rms_norm_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    return None


def _silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    from flag_gems.fused.silu_and_mul import silu_and_mul_out

    d = x.shape[-1] // 2
    # The out variant launches the same FlagGems pointwise kernel and gives
    # the dispatcher an explicit mutation contract for its output buffer.
    silu_and_mul_out(x[..., :d], x[..., d:], out)


def _silu_and_mul_fake(out: torch.Tensor, x: torch.Tensor) -> None:
    return None


def _rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    rotary_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    return gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=False,
    )


def _rotary_embedding_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    rotary_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Match the original Gems fresh-output allocation, including strided views.
    query_view = query.view(-1, query.shape[-2], query.shape[-1])
    key_view = key.view(-1, key.shape[-2], key.shape[-1])
    return torch.empty_like(query_view).view(query.shape), torch.empty_like(
        key_view
    ).view(key.shape)


def _rotary_embedding_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    rotary_interleaved: bool,
) -> None:
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=True,
    )


def _rotary_embedding_inplace_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    rotary_interleaved: bool,
) -> None:
    return None


direct_register_custom_op(
    "rms_norm",
    _rms_norm,
    fake_impl=_rms_norm_fake,
    target_lib=_LIB,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    "fused_add_rms_norm",
    _fused_add_rms_norm,
    mutates_args=["x", "residual"],
    fake_impl=_fused_add_rms_norm_fake,
    target_lib=_LIB,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    "silu_and_mul",
    _silu_and_mul,
    mutates_args=["out"],
    fake_impl=_silu_and_mul_fake,
    target_lib=_LIB,
    dispatch_key="PrivateUse1",
)


direct_register_custom_op(
    "rotary_embedding",
    _rotary_embedding,
    fake_impl=_rotary_embedding_fake,
    target_lib=_LIB,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    "rotary_embedding_inplace",
    _rotary_embedding_inplace,
    mutates_args=["query", "key"],
    fake_impl=_rotary_embedding_inplace_fake,
    target_lib=_LIB,
    dispatch_key="PrivateUse1",
)


def rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        return torch.ops.vllm_fl_musa.rms_norm(x, weight, epsilon)
    # Match FlagGems: mutate original contiguous inputs, copies otherwise.
    # Return the aliases outside the custom op so its schema returns no aliases.
    x = x.contiguous()
    residual = residual.contiguous()
    torch.ops.vllm_fl_musa.fused_add_rms_norm(x, residual, weight, epsilon)
    return x, residual


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty_like(x[..., :d])
    torch.ops.vllm_fl_musa.silu_and_mul(out, x)
    return out


def rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    rotary_interleaved: bool,
    inplace: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inplace:
        # Return input aliases outside the void custom op's explicit mutation.
        torch.ops.vllm_fl_musa.rotary_embedding_inplace(
            query, key, cos, sin, position_ids, rotary_interleaved
        )
        return query.view(query.shape), key.view(key.shape)
    return torch.ops.vllm_fl_musa.rotary_embedding(
        query, key, cos, sin, position_ids, rotary_interleaved
    )

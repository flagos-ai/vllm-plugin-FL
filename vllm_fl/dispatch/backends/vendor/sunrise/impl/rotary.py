# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise (PTPU) rotary embedding operator implementations.

Routes RoPE to ``torch_ptpu.sgl_kernel.apply_rope_with_cos_sin_cache``.
"""

from __future__ import annotations

import torch

from torch_ptpu.sgl_kernel import apply_rope_with_cos_sin_cache


_CACHED_F32_CACHE_ATTR = "_vllm_fl_sunrise_cos_sin_cache_f32"


def _get_cos_sin_cache_f32(obj, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Return a float32 ``[max_pos, head_dim]`` cos_sin_cache for PTPU sgl.

    Prefers ``obj.cos_sin_cache`` (the RotaryEmbedding layer typically owns
    the original cache) so we avoid recomputing it. Falls back to
    reconstructing from the dispatched ``cos`` / ``sin`` halves when the
    caller does not expose the cache.

    The float32 view is cached as a private attribute on ``obj`` so that we
    only pay the dtype-cast cost once per layer for the lifetime of the
    process.
    """
    cached = getattr(obj, _CACHED_F32_CACHE_ATTR, None)
    if cached is not None:
        return cached

    raw = getattr(obj, "cos_sin_cache", None)
    if raw is None:
        # Reconstruct from the dispatched halves; ``cos`` / ``sin`` are slices
        # of a contiguous ``[..., 2 * rot_dim_half]`` buffer so concatenation
        # along the last dim restores the original layout.
        raw = torch.cat([cos, sin], dim=-1)

    cache_f32 = raw.to(dtype=torch.float32)
    try:
        setattr(obj, _CACHED_F32_CACHE_ATTR, cache_f32)
    except (AttributeError, TypeError):
        # ``obj`` might be a frozen / slot-based object; that is fine, we just
        # eat the cast cost on every call in that case.
        pass
    return cache_f32


def rotary_embedding_sunrise(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE on PTPU.

    Args:
        obj: The calling RotaryEmbedding layer (used to fetch the original
            ``cos_sin_cache`` and to memoize the float32 version).
        query: ``[num_tokens, num_heads, rotary_dim]`` (3D, matching the
            shape produced by ``RotaryEmbeddingFL.forward_oot``).
        key: ``[num_tokens, num_kv_heads, rotary_dim]`` (3D).
        cos, sin: ``[max_seq_len, rotary_dim // 2]`` each; redundant when
            ``obj`` exposes ``cos_sin_cache`` but kept for fallback.
        position_ids: ``[num_tokens]`` flat tensor of absolute positions.
        rotary_interleaved: ``False`` for neox-style (Qwen3); ``True`` for
            GPT-J style interleaved layout.
        inplace: Honoured implicitly -- PTPU sgl operates in-place on the
            flat 2D view and returns the same tensors.

    Returns:
        Tuple ``(q_embed, k_embed)`` with the original 3D shapes restored.
    """
    num_tokens = query.shape[0]
    rotary_dim = query.shape[-1]
    head_size = getattr(obj, "head_size", rotary_dim)

    if rotary_dim != head_size:
        # Partial rotary: the dispatched ``query`` only contains the rotated
        # slice ``[..., :rotary_dim]``, but PTPU's ``apply_rope_with_cos_sin_cache``
        # expects a flat ``[tokens, num_heads * head_size]`` buffer covering
        # the full head and rotates only the leading ``rotary_dim`` channels
        # internally. Reconstructing the full head here would require knowing
        # the unrotated pass-through half, which we do not have. Fall back to
        # the FlagGems path for this less-common model topology.
        from vllm_fl.dispatch.backends.flaggems.impl.rotary import (
            rotary_embedding_flaggems,
        )

        return rotary_embedding_flaggems(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    query_shape = query.shape
    key_shape = key.shape

    # PTPU sgl wants 2D [tokens, num_heads * head_size]. ``.contiguous()`` is
    # required because the upstream wrapper produces a strided slice via
    # ``query[..., :rotary_dim]``.
    query_flat = query.contiguous().view(num_tokens, -1)
    key_flat = key.contiguous().view(num_tokens, -1)

    cos_sin_cache_f32 = _get_cos_sin_cache_f32(obj, cos, sin)

    q_out, k_out = apply_rope_with_cos_sin_cache(
        positions=position_ids,
        query=query_flat,
        key=key_flat,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache_f32,
        is_neox=not rotary_interleaved,
    )

    return q_out.view(query_shape), k_out.view(key_shape)

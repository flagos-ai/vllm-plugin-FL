# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""Gather MRoPE cache rows without the TXDA advanced-index CPU fallback.

The native rotation below is adapted from vLLM v0.32.0
vllm/model_executor/layers/rotary_embedding/mrope.py. Only cache lookup changes.
"""

import functools

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.fl.txda.mrope")


def gather_rows(cache: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Copy selected rows on-device, transferring only the small index tensor."""
    if cache.ndim != 2 or cache.layout != torch.strided:
        raise ValueError("MRoPE cache must be a strided 2D tensor")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("MRoPE positions must have dtype int32 or int64")
    if positions.device.type != "cpu" and positions.device != cache.device:
        raise ValueError("MRoPE positions must be on CPU or the cache device")
    rows = cache.shape[0]
    indices = positions.cpu().reshape(-1).tolist()
    normalized = []
    for index in indices:
        if not -rows <= index < rows:
            raise IndexError(f"MRoPE position {index} is outside the cache")
        normalized.append(index + rows if index < 0 else index)

    output = torch.empty(
        (*positions.shape, cache.shape[1]), dtype=cache.dtype, device=cache.device
    )
    if not normalized or cache.shape[1] == 0:
        return output
    flat = output.view(-1, cache.shape[1])
    start = 0
    while start < len(normalized):
        first = normalized[start]
        end = start + 1
        while end < len(normalized) and normalized[end] == first + end - start:
            end += 1
        flat[start:end].copy_(cache[first : first + end - start])
        start = end
    return output


def _forward_native(self, positions, query, key=None, offsets=None):
    """Retain the upstream native MRoPE formula and partial-rotation tail."""
    assert positions.ndim == 1 or positions.ndim == 2
    assert key is not None

    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    num_tokens = positions.shape[-1]
    cos_sin = gather_rows(cos_sin_cache, positions)
    if not getattr(self, "_fl_txda_mrope_lookup_reported", False):
        logger.info(
            "TXDA MRoPE lookup active: provider=%s cache_shape=%s positions_shape=%s "
            "cache_device=%s positions_device=%s",
            f"{gather_rows.__module__}.{gather_rows.__name__}",
            tuple(cos_sin_cache.shape),
            tuple(positions.shape),
            cos_sin_cache.device,
            positions.device,
        )
        self._fl_txda_mrope_lookup_reported = True
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        assert self.mrope_section
        if self.mrope_interleaved:
            from vllm.model_executor.layers.rotary_embedding.mrope import (
                apply_interleaved_rope,
            )

            cos = apply_interleaved_rope(cos, self.mrope_section)
            sin = apply_interleaved_rope(sin, self.mrope_section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    query_rot = self.apply_rotary_emb.forward_native(query_rot, cos, sin)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)
    key_rot = key[..., : self.rotary_dim]
    key_pass = key[..., self.rotary_dim :]
    key_rot = self.apply_rotary_emb.forward_native(key_rot, cos, sin)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def patch_txda_mrope() -> bool:
    """Install before model construction; leave non-TXDA calls unchanged."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "txda":
        return False
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    original = MRotaryEmbedding.forward_native
    if getattr(original, "_fl_txda_mrope_lookup", False):
        return True

    @functools.wraps(original)
    def forward_native(self, positions, query, key=None, offsets=None):
        if query.device.type != "txda":
            return original(self, positions, query, key, offsets)
        return _forward_native(self, positions, query, key, offsets)

    forward_native._fl_txda_mrope_lookup = True
    MRotaryEmbedding.forward_native = forward_native
    return True

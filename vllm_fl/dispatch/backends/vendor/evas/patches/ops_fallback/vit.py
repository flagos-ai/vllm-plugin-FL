# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU ViT pos-embed interpolation patch.

``Qwen3_VisionTransformer.fast_pos_embed_interpolate`` picks the Triton branch
because ``HAS_TRITON`` is True, and that kernel cannot compile on LPU
(``math.floor``), so initialization fails before the vision tower runs.
Upstream's pure-PyTorch fallback clamps indices on an int64 tensor, which
LPU's clamp rejects; this patch ports it with the clamp in fp32 (plus
``linspace`` as arange+scale) — same arithmetic. Only applied on LPU.

"""

import torch

_PATCHED = False


def _pos_embed_interpolate_lpu(
    embed_weight,
    t,
    h,
    w,
    num_grid_per_side,
    m_size,
    dtype,
):
    """LPU-safe port of upstream ``pos_embed_interpolate_native``.

    Only divergence from upstream: the index-bounds clamp runs in fp32 then
    casts back to long (LPU's clamp rejects int64), and linspace is replaced by
    arange+scale (LPU's linspace rejects fp32 with "unsupported dtypes: [Long]").
    All other ops (meshgrid, indexing, embed gather, weighted sum, reshape/permute)
    are unchanged and already LPU-compatible.
    """
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    # LPU: torch.linspace is unsupported (aten "received unsupported dtypes: [Long]").
    # Use arange + scale instead: linspace(0, N-1, h) = arange(h) * (N-1) / (h-1).
    h_idxs = torch.arange(h, device=device, dtype=torch.float32) * (num_grid_per_side - 1) / (h - 1)
    w_idxs = torch.arange(w, device=device, dtype=torch.float32) * (num_grid_per_side - 1) / (w - 1)

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    # LPU: clamp rejects int64. Clamp in fp32 then cast back to long.
    _cap = float(num_grid_per_side - 1)
    h_ceil = torch.clamp(h_idxs + 1.0, max=_cap).to(torch.long)
    w_ceil = torch.clamp(w_idxs + 1.0, max=_cap).to(torch.long)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    weights = weights.to(dtype=dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)


def apply_lpu_vit_patch() -> None:
    """Force Qwen3-VL pos-embed interpolation onto the LPU-safe native port.

    Monkey-patches ``Qwen3_VisionTransformer.fast_pos_embed_interpolate`` to
    call ``_pos_embed_interpolate_lpu`` (an LPU-safe port of the upstream
    native fallback) instead of the ``HAS_TRITON``-gated Triton kernel that
    crashes on LPU.

    ALSO redirects ``torch.nn.functional.scaled_dot_product_attention`` to
    FlagGems' evas-backend SDPA override. Rationale: the vision attention path
    (``Qwen2_5_VisionAttention`` → ``MMEncoderAttention.forward_native`` →
    ``vit_torch_sdpa_wrapper`` → ``apply_sdpa``) calls ``F.scaled_dot_product_attention``
    directly — bypassing FlagGems. LPU's native SDPA only supports
    ``head_size==128`` (it fails its parameter check and reports ``V_d`` as
    unsupported for other head sizes); Qwen3 ViT head_dim ≠ 128, so native
    SDPA crashes. FlagGems' evas SDPA override is head-size-agnostic
    (pure-PyTorch bmm, fp32 softmax), but is registered only under the
    ``flag_gems.*`` namespace (consumed by the evas GQA/MLA attention backend
    via ``flag_gems.scaled_dot_product_attention``),
    NOT under ``F.scaled_dot_product_attention``. We close that gap here —
    mirroring what the FlagGems ARM backend does in
    ``_arm/ops/__init__.py::_register_sdpa``.

    Safety: the evas GQA/MLA backends call ``flag_gems.scaled_dot_product_attention``
    explicitly (attention.py:162), NOT ``F.scaled_dot_product_attention``, so
    redirecting ``F.sdpa`` does not double-route or recurse those paths. The
    vision path is the only ``F.sdpa`` caller on LPU, and its shapes
    (``[1, H, Sq, D]`` after rearrange, ``enable_gqa=False`` since ViT is
    non-GQA) match the evas override's contract exactly. Idempotent.
    """
    global _PATCHED
    if _PATCHED:
        return

    import torch.nn.functional as _F
    from flag_gems import scaled_dot_product_attention as _evas_sdpa
    from vllm.model_executor.models import qwen3_vl

    def _fast_pos_embed_interpolate_lpu(self, grid_thw):
        """LPU-safe replacement: use the fp32-clamp port, bypassing HAS_TRITON."""
        outputs = []
        for t, h, w in grid_thw:
            outputs.append(
                _pos_embed_interpolate_lpu(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.dtype,
                )
            )
        return torch.cat(outputs, dim=0)

    qwen3_vl.Qwen3_VisionTransformer.fast_pos_embed_interpolate = (
        _fast_pos_embed_interpolate_lpu
    )
    # Redirect F.sdpa -> evas SDPA (head-size-agnostic bmm). See docstring for
    # why this is safe (GQA/MLA paths use flag_gems.sdpa explicitly, not F.sdpa).
    _F.scaled_dot_product_attention = _evas_sdpa
    _PATCHED = True

# Copyright (c) 2026 BAAI. All rights reserved.
"""Fused Triton M-RoPE (multimodal rotary embedding) for GCU.

``MRotaryEmbedding`` is looked up by its **exact** class name in
``vllm.model_executor.custom_op.CustomOp.__new__`` -> ``op_registry_oot``, so
registering the OOT impl for ``RotaryEmbedding`` does *not* cover it.  Without
an OOT entry it runs ``forward_native``, which for ``[3, num_tokens]`` positions
costs ~20 aten launches (``index`` gather, ``chunk``, ``split`` + ``cat`` per
section, two ``apply_rotary_emb`` torch decompositions, two ``cat`` + reshape).

This impl reproduces ``forward_cuda``'s fast path: one gather for
``cos_sin_cache[positions]`` plus **one** Triton kernel that

* selects the t/h/w section per lane (no ``split``/``cat`` at all -- the three
  stacked sections are read with three masked loads that are summed),
* applies the rotation **in place** on the rotary prefix of q/k, leaving the
  pass-through tail untouched (so no ``cat`` on the output side either).

GCU300 caveat: ``tl.arange``-derived load/store masks may only use ``<`` /
``~(a < b)`` (``>=`` / ``<=`` / ``>`` are mis-lowered), hence the section masks
below.  ``NUM_SPC`` is the constant grid size (48), so a changing batch size
never triggers a fresh JIT compilation.
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

GCU_NUM_GRID = 48
GCU_MAX_WARPS = 4
# Per-program tile = next_pow2(num_heads) * rotary_dim/2 lanes.  Beyond this the
# GCU300 register allocator gives up ("ran out of registers"), so we fall back
# to the torch path instead of failing the run.
_MAX_TILE_LANES = 4096
# Measured crossover on GCU300 (bf16, head=128, rotary=128; identical for
# 8/16/32 q-heads per rank): fused vs torch path
#   n=1   6.1x   n=16  6.6x   n=64  3.1x   n=128 1.6x   n=192 1.0x
#   n=256 0.79x  n=512 0.78x  n=1024 0.78x
# The kernel is DTE-descriptor bound (one descriptor set per row), so long
# prefill batches regress; keep the torch path there.
_FUSED_MAX_TOKENS = 192
_SEL_CACHE: dict = {}  # (section, half_rd, device) -> int8 selection LUT


@triton.jit
def _triton_mrope_forward_gcu(
    q_ptr, k_ptr, cos, sin, sel_ptr, num_tokens,
    n_qh: tl.constexpr, n_kh: tl.constexpr,
    hd: tl.constexpr, rd: tl.constexpr,
    pad_n_qh: tl.constexpr, pad_n_kh: tl.constexpr, pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr, mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr, is_interleaved: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    half_rd = rd // 2
    cos_offsets = tl.arange(0, pad_hd // 2)
    # !!! GCU300 triton quirk !!!
    # Masks derived from ``tl.arange`` must NOT use ``>=`` / ``<=`` / ``>``:
    # the backend mis-lowers those predicates when the tensor is used as a
    # ``tl.load``/``tl.store`` mask (``>=`` turns into "all true", ``>`` turns
    # into ``<``).  Always express bounds with ``<`` or ``~(a < b)``.
    if is_interleaved:
        # ``cos_offsets % 3`` masks blow the GCU300 register allocator; read the
        # t/h/w selection from a tiny precomputed LUT instead (masks derived from
        # loaded data are safe on this backend).
        sv = tl.load(sel_ptr + cos_offsets, mask=cos_offsets < half_rd, other=0)
        t_mask = sv == 0
        h_mask = sv == 1
        w_mask = sv == 2
    else:
        lt_t = cos_offsets < mrope_section_t
        lt_h = cos_offsets < (mrope_section_t + mrope_section_h)
        t_mask = lt_t
        h_mask = lt_h & ~lt_t
        w_mask = (cos_offsets < half_rd) & ~lt_h
    fqo = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    fko = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    fqm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < rd // 2)
    fkm = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < rd // 2)
    for row in range(pid, num_tokens, NUM_SPC):
        qr = q_ptr + row * (n_qh * hd)
        kr = k_ptr + row * (n_kh * hd)
        tc = cos + row * half_rd
        hc = tc + num_tokens * half_rd
        wc = hc + num_tokens * half_rd
        ts = sin + row * half_rd
        hs = ts + num_tokens * half_rd
        ws = hs + num_tokens * half_rd
        cr = tl.load(tc + cos_offsets, mask=t_mask, other=0) + tl.load(hc + cos_offsets, mask=h_mask, other=0) + tl.load(wc + cos_offsets, mask=w_mask, other=0)
        sr = tl.load(ts + cos_offsets, mask=t_mask, other=0) + tl.load(hs + cos_offsets, mask=h_mask, other=0) + tl.load(ws + cos_offsets, mask=w_mask, other=0)
        q1 = tl.load(qr + fqo, mask=fqm, other=0).to(sr.dtype)
        k1 = tl.load(kr + fko, mask=fkm, other=0).to(sr.dtype)
        q2 = tl.load(qr + fqo + rd // 2, mask=fqm, other=0).to(sr.dtype)
        k2 = tl.load(kr + fko + rd // 2, mask=fkm, other=0).to(sr.dtype)
        tl.store(qr + fqo, q1 * cr - q2 * sr, mask=fqm)
        tl.store(qr + fqo + rd // 2, q2 * cr + q1 * sr, mask=fqm)
        tl.store(kr + fko, k1 * cr - k2 * sr, mask=fkm)
        tl.store(kr + fko + rd // 2, k2 * cr + k1 * sr, mask=fkm)

def _interleaved_sel_lut(mrope_section, half_rd, device):
    """0 = t, 1 = h, 2 = w per lane -- mirrors ``apply_interleaved_rope``."""
    key = (tuple(mrope_section), half_rd, str(device))
    lut = _SEL_CACHE.get(key)
    if lut is None:
        sel = torch.zeros(half_rd, dtype=torch.int8)
        t, h, w = mrope_section
        # vLLM's ``x[..., 1 : h*3 : 3]`` slice is clamped to the tensor width,
        # so the index range must be clamped the same way.
        sel[torch.arange(1, min(h * 3, half_rd), 3)] = 1
        sel[torch.arange(2, min(w * 3, half_rd), 3)] = 2
        lut = sel.to(device)
        _SEL_CACHE[key] = lut
    return lut


def triton_mrope_gcu(
    q, k, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved
):
    """In-place rotary on the first ``rotary_dim`` lanes of every head."""
    n_row, n_qhh = q.shape
    n_qh = n_qhh // head_size
    n_kh = k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_nq = triton.next_power_of_2(n_qh)
    pad_nk = triton.next_power_of_2(n_kh)
    q, k = q.contiguous(), k.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    if mrope_interleaved:
        sel = _interleaved_sel_lut(mrope_section, rotary_dim // 2, cos.device)
    else:
        sel = cos  # unused by the kernel
    _triton_mrope_forward_gcu[(min(n_row, GCU_NUM_GRID),)](
        q,
        k,
        cos,
        sin,
        sel,
        n_row,
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        pad_nq,
        pad_nk,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
        NUM_SPC=GCU_NUM_GRID,
        num_warps=GCU_MAX_WARPS,
    )
    return q, k


def mrope_gcu(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
):
    """Dispatch entry for the ``mrope`` op (mirrors ``MRotaryEmbedding.forward_cuda``)."""
    if key is None or positions.ndim != 2 or not getattr(obj, "mrope_section", None):
        # text-only positions / no section split / CPU: upstream torch path
        return _forward_native(obj, positions, query, key)
    if query.device.type not in ("gcu", "cuda"):
        return _forward_native(obj, positions, query, key)

    if positions.shape[-1] > _FUSED_MAX_TOKENS:
        return _forward_native(obj, positions, query, key)

    n_qh = query.shape[-1] // obj.head_size
    n_kh = key.shape[-1] // obj.head_size
    lanes = max(triton.next_power_of_2(n_qh), triton.next_power_of_2(n_kh))
    lanes *= triton.next_power_of_2(obj.head_size) // 2
    if lanes > _MAX_TILE_LANES:
        # too many heads per rank for one fused tile (e.g. TP1 with 64 heads)
        return _forward_native(obj, positions, query, key)

    cos_sin_cache = obj._match_cos_sin_cache_dtype(query)
    cos_sin = cos_sin_cache[positions]  # [3, num_tokens, rotary_dim]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query_shape = query.shape
    key_shape = key.shape
    q, k = triton_mrope_gcu(
        query,
        key,
        cos,
        sin,
        obj.mrope_section,
        obj.head_size,
        obj.rotary_dim,
        bool(getattr(obj, "mrope_interleaved", False)),
    )
    return q.reshape(query_shape), k.reshape(key_shape)


# ---------------------------------------------------------------------------
# monkey-patch entry
# ---------------------------------------------------------------------------
_ROPE_MODULE = "vllm.model_executor.layers.rotary_embedding.mrope"

_patched = False
# Saved before we overwrite the method: the fallbacks above must call the
# *original* torch implementation, otherwise patching forward_native would make
# them recurse forever.
_orig_forward_native = None


def _forward_native(obj, positions, query, key):
    fn = _orig_forward_native
    if fn is None:
        return obj.forward_native(positions, query, key)
    return fn(obj, positions, query, key)


def apply_mrope_gcu_patch() -> None:
    """Route ``MRotaryEmbedding.forward_native`` to the fused GCU Triton kernel.

    Patching the class method (rather than registering an OOT layer) also covers
    every subclass of ``MRotaryEmbedding``, which the exact-class-name OOT
    lookup in ``CustomOp.__new__`` would otherwise miss.
    """
    global _patched, _orig_forward_native
    if _patched:
        return
    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return
    try:
        mod = importlib.import_module(_ROPE_MODULE)
        cls = getattr(mod, "MRotaryEmbedding", None)
        if cls is None:
            logger.debug("mrope patch deferred (partial import)")
            return
        if cls.__dict__.get("forward_native") is not mrope_forward_gcu:
            _orig_forward_native = cls.forward_native
            cls.forward_native = mrope_forward_gcu
        _patched = True
        logger.info(
            "Patched MRotaryEmbedding.forward_native -> fused GCU Triton kernel "
            "(grid<=%d, fused for num_tokens<=%d)",
            GCU_NUM_GRID,
            _FUSED_MAX_TOKENS,
        )
    except Exception as exc:
        logger.debug("mrope patch deferred (will retry): %s", exc)


def mrope_forward_gcu(self, positions, query, key=None, offsets=None):
    return mrope_gcu(self, positions, query, key, offsets)


__all__ = ["mrope_gcu", "mrope_forward_gcu", "triton_mrope_gcu", "apply_mrope_gcu_patch"]

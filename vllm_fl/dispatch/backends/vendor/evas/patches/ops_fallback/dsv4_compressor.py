# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU patch: DSV4 compressor / sparse-MLA metadata pure-PyTorch stand-ins.

The compressor's Triton kernels are compile dead-ends under evas-triton-opt
and the compressed-slot-mapping kernel silently never writes, so this module
rebinds them (plus the C128A topk metadata) to pure-PyTorch stand-ins — the
slot mapping via a sys.modules-wide alias sweep, because upstream modules
import the name. Applied unconditionally on LPU; every rebinding is inert
unless the DSV4 attention path runs.

"""

from __future__ import annotations

from typing import Any

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Captured once, before the first rebind, so the post-sweep self-check can tell
# a real upstream alias apart from our stand-in even if the installer runs
# again (re-apply / module reload).
_UPSTREAM_GCS = None
_SWEEP_RECHECKED = False

# Guards apply_lpu_dsv4_compressor_patch() against re-install.
_GENERIC_PATCHED = False


def _lpu_build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch stand-in for ``flashmla_sparse.build_c128a_topk_metadata``.

    The upstream Triton kernel ``_build_c128a_topk_metadata_kernel`` carries a
    ``scf.for`` loop with ``tt.reduce`` (``count += tl.sum(is_valid, axis=0)``),
    the known evas-Triton dead path (``evas-triton-opt`` fails: 'scf.for' op
    iter_arg and yielded value have different type).  The computation itself is
    trivial metadata and has no multi-stream/custom-op content, so it is
    re-expressed with basic int32 tensor ops (LPU aten dtype constraint):

    - decode tokens: for each offset < max_compressed_tokens,
      slot = block_table[req_idx, offset // block_size] * block_size
             + offset % block_size, masked to offset < num_compressed (else -1);
      decode_lens = num_compressed if slot_mapping[token] >= 0 else 0.
    - prefill tokens: local indices [0, ..., num_compressed-1, -1, ...].

    num_compressed = min((position + 1) // compress_ratio, max_compressed_tokens).
    """
    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens

    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    pos = positions.to(torch.int32)
    num_compressed = (pos + 1) // compress_ratio
    num_compressed = torch.minimum(
        num_compressed,
        torch.full_like(num_compressed, max_compressed_tokens),
    )
    offsets = torch.arange(
        max_compressed_tokens, dtype=torch.int32, device=positions.device
    )

    if num_decode_tokens > 0:
        req_idx = token_to_req_indices[:num_decode_tokens]
        block_indices = offsets // block_size
        # Triton masks invalid loads; clamp the tail reads to the last column
        # so they never OOB, then where(is_valid, slot, -1) nulls them out --
        # identical final values to the masked-load + where sequence upstream.
        block_numbers = block_table[
            req_idx[:, None], block_indices.clamp(max=block_table.shape[1] - 1)
        ]
        slot_ids = block_numbers * block_size + offsets % block_size
        is_valid = offsets.unsqueeze(0) < num_compressed[:num_decode_tokens].unsqueeze(1)
        slot_ids = torch.where(
            is_valid,
            slot_ids,
            torch.tensor(-1, dtype=torch.int32, device=positions.device),
        )
        global_decode_buffer[:num_decode_tokens, :max_compressed_tokens].copy_(slot_ids)

        is_valid_token = slot_mapping[:num_decode_tokens] >= 0
        decode_lens_buffer[:num_decode_tokens].copy_(
            torch.where(is_valid_token, num_compressed[:num_decode_tokens], 0)
        )

    if num_prefill_tokens > 0:
        pfx_num_compressed = num_compressed[num_decode_tokens:]
        pfx_ids = torch.where(
            offsets.unsqueeze(0) < pfx_num_compressed.unsqueeze(1),
            offsets.unsqueeze(0),
            torch.tensor(-1, dtype=torch.int32, device=positions.device),
        )
        prefill_buffer[:num_prefill_tokens, :max_compressed_tokens].copy_(pfx_ids)

    return global_decode, decode_lens, prefill_local


def _install_c128a_topk_metadata_override() -> None:
    """Rebind sparse_mla.build_c128a_topk_metadata to the pure-PyTorch op.

    The C128A (compress_ratio == 128) decode/prefill topk metadata is the only
    direct Triton kernel in ``vllm/models/deepseek_v4/sparse_mla.py``
    (``_build_c128a_topk_metadata_kernel``, a ``scf.for`` + ``tt.reduce`` dead
    path on evas).  Patching the module-level name is picked up by the sparse
    MLA metadata builder at call time.
    """
    from vllm.models.deepseek_v4 import sparse_mla as _mod

    _mod.build_c128a_topk_metadata = _lpu_build_c128a_topk_metadata


def _lpu_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch stand-in for ``compressor_utils.get_compressed_slot_mapping``.

    The upstream Triton ``_compressed_slot_mapping_kernel`` loads
    ``block_table_ptr + batch_idx * stride + block_ids`` (block-ids vector from
    ``pos_after_compress // block_size``) -- the block-table multi-block
    liveness pattern that is a known evas pointwise compile dead-end (see
    memory ``project-fl-has-triton-trap-status``).  On LPU the kernel never
    populates ``out``, leaving the caller's ``fill_(-1)`` in place, so the
    compressor's ``kv_slot_mapping`` is all -1 and the compressed KV cache is
    never written.

    The computation is trivial per-request metadata: token ``t`` of request
    ``r`` at position ``pos`` maps to compressed slot
    ``block_table[r, pos//ratio // bs] * bs + pos//ratio % bs`` when
    ``(pos+1) % ratio == 0``, else ``-1`` (PAD).  Re-expressed with basic int32
    tensor ops (LPU aten dtype constraint); ``query_start_loc`` / ``seq_lens``
    are read on-device, matching the upstream kernel's per-request loop.

    Same signature/buffer contracts as upstream; writes the ``out[:num_tokens]``
    slice.
    """
    global _SWEEP_RECHECKED
    if not _SWEEP_RECHECKED:
        _SWEEP_RECHECKED = True
        try:
            _sweep_and_check_gcs(context="first-use")
        except Exception as _exc:  # noqa: BLE001 - never break the op path
            logger.error("[dsv4-attn] first-use alias re-check failed: %s", _exc)

    if out is not None:
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )
    num_reqs = block_table.shape[0]
    if num_tokens == 0 or num_reqs == 0:
        return slot_mapping

    qsl = query_start_loc  # [num_reqs+1]
    q_lens = qsl[1:] - qsl[:-1]  # [num_reqs]
    req_ids = torch.arange(num_reqs, dtype=torch.int32, device=qsl.device)
    req_per_token = torch.repeat_interleave(req_ids, q_lens)[:num_tokens]
    if req_per_token.numel() == 0:
        return slot_mapping
    start_pos = seq_lens - q_lens  # [num_reqs] int32
    local_idx = (
        torch.arange(req_per_token.numel(), dtype=torch.int32, device=qsl.device)
        - qsl[req_per_token]
    )
    pos = start_pos[req_per_token] + local_idx  # [T] int32

    is_valid = (pos + 1) % compress_ratio == 0
    pos_after = pos // compress_ratio
    block_ids = pos_after // block_size
    # Triton masks invalid loads (other=0); clamp the tail reads so they never
    # OOB, then where(is_valid, slot, -1) nulls them out -- identical final
    # values to the masked-load + where sequence upstream.
    bn = block_table[
        req_per_token, block_ids.clamp(min=0).clamp(max=block_table.shape[1] - 1)
    ]
    slots = bn.to(torch.int64) * block_size + (pos_after % block_size).to(torch.int64)
    slots = torch.where(
        is_valid, slots, torch.full_like(slots, -1, dtype=torch.int64)
    )
    slot_mapping[: req_per_token.numel()].copy_(slots)
    return slot_mapping


# Per-row slice stores beat one batched index_put_ while the valid-row count
# stays small; measured crossover on LPU is V ~= 4 (V=1: 0.53 vs 1.45 ms,
# V=4: 2.07 vs 1.87 ms, V=16: 8.15 vs 2.22 ms).  Decode runs at V <= a few.
_SLICE_STORE_MAX_ROWS = 4

# The state cache is bound from the KV pool, so its stride(0) is the pool group
# page (1,039,680 B on the full 43-layer model) even though one compressor block
# only needs 32,768 B.  On LPU every op that touches such a view pays for the
# whole spanned buffer: reading one element or storing one row costs
# milliseconds at that span, against sub-millisecond for the same ops on a
# dense tensor of identical shape.  A dense stand-in removes that tax at
# identical values, so it is safe to compare arm against arm.
_DENSE_STATE_CACHE = os.environ.get("VLLM_FL_DSV4_CMPR_STATE_DENSE", "0") == "1"
# id(pooled tensor) -> (dense stand-in, owner).  Holding the owner keeps the id
# unique for the layer lifetime and lets the save and fused call sites of one
# layer share a single buffer.  Entries are never evicted: a worker that
# re-creates the cache (a KV pool refit) would add one dense copy per layer per
# refit and keep the old pool alive, which is acceptable because the knob is off
# by default and refitting is not a supported in-process flow.
_DENSE_STATE_STORE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
# Call sites allowed to obtain the stand-in.  The stand-in detaches the cache
# from the KV pool, so a reader that does NOT come through here would keep
# reading the pooled view and silently see stale data.  That list is therefore a
# contract: extend it only after auditing the new reader, and never hand the raw
# pooled tensor to a kernel while this knob is on.
_DENSE_STATE_SITES = ("save_partial_states", "fused_kv_compress_norm_rope_insert")


def _dense_state_cache(state_cache: torch.Tensor, site: str) -> torch.Tensor:
    """Return a dense stand-in for a page-padded state-cache view (or itself).

    Only the two audited call sites above may ask for it; any other caller is a
    code bug (the contract is documented in the block comment).  An unexpected
    layout falls back to the pooled tensor instead of raising: the run stays
    correct and only loses the speedup.
    """
    if not _DENSE_STATE_CACHE or state_cache.is_contiguous():
        return state_cache
    if site not in _DENSE_STATE_SITES:
        raise AssertionError(
            "dense state cache requested by unaudited site %r; readers must be "
            "added to _DENSE_STATE_SITES with a write/read audit" % (site,))
    if state_cache.dim() != 3 or state_cache.dtype != torch.float32:
        logger.warning(
            "[cmpr-dense] unexpected state_cache layout dim=%d dtype=%s at %s;"
            " keeping the pooled view", state_cache.dim(), state_cache.dtype,
            site)
        return state_cache
    entry = _DENSE_STATE_STORE.get(id(state_cache))
    if entry is None:
        dense = torch.zeros(state_cache.shape, dtype=state_cache.dtype,
                            device=state_cache.device)
        entry = (dense, state_cache)
        _DENSE_STATE_STORE[id(state_cache)] = entry
        esz = state_cache.element_size()
        span_mb = (state_cache.stride(0) * state_cache.shape[0] * esz) / (1 << 20)
        logger.info(
            "[cmpr-dense] state_cache shape=%s stride=%s esz=%d bytes=%.1f MB"
            " span=%.1f MB -> dense %.1f MB",
            tuple(state_cache.shape), tuple(state_cache.stride()), esz,
            state_cache.numel() * esz / (1 << 20),
            span_mb, dense.numel() * esz / (1 << 20))
    return entry[0]


def _write_packed(cache: torch.Tensor, write) -> bool:
    """Commit ``write(target)`` through a host mirror when ``cache`` is packed.

    vLLM >= 0.25 packs every KV page (compressor state rows included) into ONE
    allocation and hands each layer a page-strided window of it.  On LPU any
    strict sub-interval write of such a window is staged as the whole backing
    (advanced-index forms up to 2x, a per-row slice 1x), which is an OOM at
    headroom -- measured 2026-10-26 on a 1.74 GiB pool with 0.98 GiB free.  So
    the caller's row write runs against a host copy and is pushed back with one
    whole-buffer H2D: same bytes, zero device transient.  Slow by construction;
    the size lever is
    ``VLLM_FL_KV_CPU_DEQUANT_THRESHOLD``.

    ``write(target)`` must do all of its work relative to ``target`` -- move
    payload and index tensors with ``.to(target.device)`` -- so the same closure
    serves both routes.  Returns True when the host route ran and committed;
    False means the caller must still write in place.
    """
    # Host-mirror machinery is platform-local (vendor/evas/local/, 02doc
    # §12.5): _evas/fused is the core-op mirror surface and must not carry
    # non-mirroring files.
    from ...local import packed_cache as _packed_cache

    if not _packed_cache.is_oversized(cache):
        return False
    host = _packed_cache.host_view_of_storage(cache)
    write(host)
    _packed_cache.write_back_from_host(cache, host)
    return True


def _flat_element_view(t: torch.Tensor) -> torch.Tensor:
    """1-D view of ``t`` over its WHOLE backing, starting at ``t``'s offset.

    ``reshape(-1)`` on a page-strided (packed) cache COPY -- writes into the
    copy are silently lost, and the copy itself stages the pool -- so the flat
    index space must come from the backing storage.  Offsets stay in ``t``'s
    own element units: ``block_idx * stride(0) + pos * token_stride``.
    """
    storage = t.untyped_storage()
    n = storage.nbytes() // t.element_size()
    flat = torch.empty(0, dtype=t.dtype, device=t.device).set_(storage, 0, (n,), (1,))
    return flat[t.storage_offset():]


def _lpu_save_partial_states(
    kv: torch.Tensor,
    kv_stride: int,
    score: torch.Tensor,
    score_stride: int,
    ape: torch.Tensor,
    ape_stride: int,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    state_cache_stride0: int,
    state_cache_stride1: int,
    slot_mapping: torch.Tensor,
    block_size: int,
    HEAD_SIZE: int,
    TRITON_BLOCK_SIZE: int = 0,
    STATE_WIDTH: int = 0,
    COMPRESS_RATIO: int = 0,
    **kwargs,
) -> None:
    """Pure-PyTorch stand-in for ``_save_partial_states_kernel``.

    The upstream Triton kernel (deepseek_compressor.py) carries a runtime
    ``if slot_id < 0: return`` guard (the vLLM PAD sentinel) -- a
    ``cf.cond_br`` the evas evofc-opt rejects ('failed to legalize operation
    cf.cond_br').  The computation is a trivial per-token scatter into the
    compressor state cache, so it is re-expressed with basic int32/int64
    tensor ops (LPU aten dtype constraint):

    - invalid tokens (slot_id < 0): no write (the Triton kernel returns
      before any store).
    - valid tokens: ``state_cache[block_idx, pos_in_block, 0:HEAD_SIZE]``
      = ``kv[token, :HEAD_SIZE]`` and
      ``state_cache[block_idx, pos_in_block, STATE_WIDTH:STATE_WIDTH+HEAD_SIZE]``
      = ``score[token, :HEAD_SIZE] + ape[position % COMPRESS_RATIO, :HEAD_SIZE]``
      (fused APE addition), where ``block_idx = slot_id // block_size``,
      ``pos_in_block = slot_id % block_size``.

    Same signature/buffer contracts as upstream; the ``*_stride`` / constexpr
    args are accepted and mostly unused (the flat writes use torch's native
    stride-aware indexing).
    """
    num_actual = slot_mapping.shape[0]
    h = int(HEAD_SIZE)
    sw = int(STATE_WIDTH)
    bs = int(block_size)
    if num_actual == 0 or h == 0:
        return
    state_cache = _dense_state_cache(state_cache, "save_partial_states")
    device = state_cache.device

    # LPU aten clamp rejects int64; the metadata arrives as int64 (torch
    # zeros/arange defaults) -- convert to int32 before any clamp/arith.
    slot = slot_mapping[:num_actual].to(torch.int32)
    valid = slot >= 0
    # One device->host sync for the whole call (the two writes share the mask,
    # so asking twice only buys another full drain on LPU).
    if not bool(valid.any()):
        return
    # Clamp so bogus invalid-token entries never OOB; only valid rows write.
    slot_safe = slot.clamp(min=0)
    block_idx = slot_safe // bs
    pos_in_block = slot_safe % bs

    bidx = block_idx[valid].to(torch.int64)  # [V]
    pbidx = pos_in_block[valid].to(torch.int64)  # [V]
    kv_v = kv[:num_actual, :h][valid]
    # Fused: score += ape[position % compress_ratio]
    pos_i32 = positions[:num_actual].to(torch.int32)
    ape_row = (pos_i32 % COMPRESS_RATIO).clamp(max=ape.shape[0] - 1)
    ape_v = ape[ape_row.to(torch.int64), :h]
    score_v = (
        score[:num_actual, :h].to(torch.float32)
        + ape_v.to(torch.float32)
    )[valid]

    # The production 0.24 compressor always has STATE_WIDTH ==
    # kv.shape[-1] == coff*head_dim, i.e. the kv half and the score half
    # together are exactly one state row.  Store them as one contiguous row
    # instead of two column-ranged writes: on LPU a single strided row copy is
    # ~0.5 ms while each 3D advanced-index write costs ~1-2 ms (measured,
    # num_actual=1, page-aligned cache), so the whole call drops from ~4.1 ms
    # to ~1.7 ms.
    row = (
        torch.cat([kv_v.to(state_cache.dtype), score_v.to(state_cache.dtype)],
                  dim=1)
        if sw == h
        else None
    )

    def _write(target):
        dev = target.device
        _store_state_rows(
            target, bidx.to(dev), pbidx.to(dev),
            None if row is None else row.to(dev),
            kv_v.to(dev), score_v.to(dev), h, sw,
        )

    # vLLM >= 0.25 packs every KV page (and the compressor state rows) into ONE
    # allocation and hands each layer a page-strided window of it: all three
    # store branches below -- the decode per-row slice write included -- then
    # stage the whole backing and OOM at headroom.  ``_write_packed`` reroutes
    # them through a host mirror; on a standalone 0.24 cache it is a no-op.
    if not _write_packed(state_cache, _write):
        _write(state_cache)


def _store_state_rows(
    cache: torch.Tensor,
    bidx: torch.Tensor,
    pbidx: torch.Tensor,
    row: torch.Tensor | None,
    kv_v: torch.Tensor,
    score_v: torch.Tensor,
    h: int,
    sw: int,
) -> None:
    """Scatter partial states into ``cache`` (the real one or a host mirror).

    Every tensor argument must live on ``cache``'s device.  ``row`` is the
    combined ``[kv | score]`` payload of the ``STATE_WIDTH == kv width``
    layout; the other layouts pass ``None`` and the two halves separately.
    """
    device = cache.device
    if row is not None:
        n = row.shape[0]
        if n <= _SLICE_STORE_MAX_ROWS:
            # Decode case: a handful of rows.  Per-row slice assignment is a
            # plain copy kernel; the batched index_put_ below is only faster
            # once V grows past ~4 (crossover measured on LPU).
            bl = bidx.tolist()
            pl = pbidx.tolist()
            for i in range(n):
                cache[bl[i], pl[i], : 2 * h] = row[i]
        else:
            cols = torch.arange(2 * h, dtype=torch.int64, device=device)
            cache.index_put_(
                (bidx[:, None], pbidx[:, None], cols[None, :]), row
            )
        return

    # Generic layout (STATE_WIDTH != kv width, e.g. the pre-0.24 kernel
    # contract): two column-ranged writes.
    cols = torch.arange(h, dtype=torch.int64, device=device)
    # cache can be a STRIDED (non-contiguous) view on LPU
    # (stride(0) > shape[1]*shape[2], the cache is a padded sub-view);
    # reshape(-1) would COPY it, so a flat index_put_ writes into the copy
    # and the real cache is never touched.  Write through 3D advanced
    # indexing into the original buffer instead (verified on LPU).
    cache.index_put_(
        (bidx[:, None], pbidx[:, None], cols[None, :]),
        kv_v.to(cache.dtype),
    )
    cache.index_put_(
        (bidx[:, None], pbidx[:, None], (sw + cols)[None, :]),
        score_v.to(cache.dtype),
    )


def _lpu_fused_kv_compress_norm_rope_insert(
    # state cache (compressor internal state)
    state_cache: torch.Tensor,
    state_cache_stride0: int,
    state_cache_stride1: int,
    # metadata
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    # RMSNorm
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    # RoPE
    cos_sin_cache: torch.Tensor,
    cos_sin_stride: int,
    # KV cache output
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    # constexprs
    HEAD_SIZE: int,
    TRITON_BLOCK_SIZE: int = 0,
    STATE_WIDTH: int = 0,
    COMPRESS_RATIO: int = 0,
    OVERLAP: int = 0,
    ROPE_HEAD_DIM: int = 0,
    TOKEN_STRIDE: int = 0,
    KV_BLOCK_STRIDE: int = 0,
    fp8_cache: bool = False,
    **kwargs,
) -> None:
    """Pure-PyTorch stand-in for the fused compressor Triton kernels
    (``_fused_kv_compress_norm_rope_insert_{sparse,indexer}_attn_bf16`` and
    the FP8 variants ``_fused_kv_compress_norm_rope_insert_{sparse,indexer}_attn``).

    Both upstream kernels carry three runtime guards -- ``slot_id < 0`` (PAD
    sentinel), ``(position + 1) % COMPRESS_RATIO != 0`` (not a compression
    boundary), ``kv_slot_idx < 0`` -- each a ``cf.cond_br`` dead path on evas.
    The computation is one softmax-weighted compression + RMSNorm + GPT-J
    RoPE per boundary token, re-expressed with basic tensor ops (LPU aten
    dtype constraint):

    - per token: skip unless ``slot_id >= 0`` AND ``(position+1) % CR == 0``.
    - gather the ``(1+OVERLAP)*CR``-long window ending at ``position`` from
      the state cache (``head_offset = (tokens >= CR) * HEAD_SIZE`` shifts the
      overlap rows); softmax the score half along the window axis (-inf for
      masked rows), weighted-sum with the kv half.
    - RMSNorm the compressed vector (fp32).
    - GPT-J forward RoPE (fp32) on the last ``ROPE_HEAD_DIM`` elements using
      ``cos_sin_cache[compressed_pos]`` (first half cos, second half sin,
      per-pair).

    Store (``fp8_cache=False``): write the full bf16 ``result`` into
    ``k_cache`` at ``kv_slot_idx // kv_cache_block_size`` / ``%`` rows (the
    bf16 wrapper path, quant_config=None).

    Store (``fp8_cache=True``, real-FP8 path): the compressed nope half is
    E4M3-block-quantized and the rope half stays bf16, packed into the MODEL1
    SEGREGATED layout (data region ``bs*576`` then scale region ``bs*8``) that
    ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert._insert_fp8_model1`` and the
    upstream Triton kernel use -- indexed through flat ``[num_blocks, bs*584]``
    views, NOT ``k_cache[b, p, :]``.  Quant convention matches the upstream
    ``fused_compress_quant_cache`` kernels exactly (fp32->bf16->fp32 round
    trip, ``amax = max(|x|, 1e-4)``, ``ceil(log2(amax / 448))``, E8M0 byte
    ``exp + 127``, E4M3 encode) via ``quant_e4m3_blocks``.

    Same signature/buffer contracts as upstream; the ``*_stride`` args are
    accepted and the flat writes replicate the kernel's pointer arithmetic
    (``num_warps`` / ``launch_pdl`` ignored).
    """
    num_actual = slot_mapping.shape[0]
    h = int(HEAD_SIZE)
    sw = int(STATE_WIDTH)
    cr = int(COMPRESS_RATIO)
    ov = int(OVERLAP)
    rope_h = int(ROPE_HEAD_DIM)
    bs = int(block_size)
    if num_actual == 0 or h == 0 or cr == 0:
        return
    state_cache = _dense_state_cache(
        state_cache, "fused_kv_compress_norm_rope_insert")
    device = state_cache.device

    # LPU aten clamp rejects int64; the metadata arrives as int64 (torch
    # zeros/arange defaults) -- convert to int32 before any clamp/arith.
    slot = slot_mapping[:num_actual].to(torch.int32)
    pos = positions[:num_actual].to(torch.int32)
    req = token_to_req_indices[:num_actual].to(torch.int32)
    kv_slot_all = kv_slot_mapping[:num_actual].to(torch.int32)
    # Guards 1-3 (Triton early-return before any state-cache read), folded into
    # one mask: PAD sentinel, non-boundary position, and a negative KV slot.
    # Folding them lets the common step (non-boundary: 127/128 of the sparse
    # compressor steps, 3/4 of the indexer) exit on a single scalar sync
    # instead of paying torch.nonzero plus two more syncs down the line.
    go = (slot >= 0) & ((pos + 1) % cr == 0) & (kv_slot_all >= 0)
    if not bool(go.any()):
        return
    vi = torch.nonzero(go).squeeze(1)

    pos_v = pos[vi]
    req_v = req[vi]

    # ── Gather state cache entries ────────────────────────────────────
    w = (1 + ov) * cr
    tokens = torch.arange(w, dtype=torch.int32, device=device)  # [W]
    start = pos_v - w + 1  # [V] int32
    win_pos = start.unsqueeze(1) + tokens.unsqueeze(0)  # [V, W]
    mask_pos = win_pos >= 0  # [V, W]
    block_indices = win_pos // bs
    # Triton masks invalid loads (other=0); clamp the tail reads so they
    # never OOB, then mask the final gathers -- identical final values.
    block_nums = block_table[
        req_v.unsqueeze(1),
        block_indices.clamp(min=0).clamp(max=block_table.shape[1] - 1),
    ]  # [V, W] int32
    block_offsets = win_pos % bs
    head_offset = (tokens >= cr).to(torch.int32) * h  # [W]

    cols = torch.arange(h, dtype=torch.int64, device=device)  # [H]

    # Read through 3D advanced indexing, NOT a reshape(-1)+flat gather:
    # state_cache is a STRIDED (non-contiguous) sub-view on LPU, so
    # reshape(-1) copies it and the flat gather would read the stale pre-write
    # copy.  Indexing the original buffer sees the _lpu_save_partial_states
    # writes.
    head_cols = (head_offset.unsqueeze(1) + cols).unsqueeze(0)  # [1, W, H]
    score_cols = (sw + head_offset.unsqueeze(1) + cols).unsqueeze(0)  # [1, W, H]
    idx_3d = (block_nums.unsqueeze(2), block_offsets.unsqueeze(2))  # [V, W, 1]

    # Packed pool (vLLM >= 0.25): the state rows live inside the shared KV
    # backing, so the same advanced-index read stages the whole pool on LPU
    # each call.  Take ONE host mirror, gather the two small [V, W, H] windows
    # there, and bring back only those; the softmax math stays on the device.
    # Host-mirror machinery is platform-local (vendor/evas/local/, 02doc
    # §12.5): _evas/fused is the core-op mirror surface and must not carry
    # non-mirroring files.
    from ...local import packed_cache as _packed_cache

    host_state = (
        _packed_cache.host_view_of_storage(state_cache)
        if _packed_cache.is_oversized(state_cache) else None
    )

    def _gather(col_idx):
        if host_state is None:
            return state_cache[idx_3d[0], idx_3d[1], col_idx]
        out = host_state[idx_3d[0].cpu(), idx_3d[1].cpu(), col_idx.cpu()]
        return out.to(device)

    score = torch.where(
        mask_pos.unsqueeze(2),
        _gather(score_cols).to(torch.float32),
        torch.full((1,), float("-inf"), dtype=torch.float32, device=device),
    )  # [V, W, H]; -inf for masked window rows (Triton other=-inf)
    score = torch.softmax(score, dim=1)  # along the window axis
    kv = torch.where(
        mask_pos.unsqueeze(2),
        _gather(head_cols).to(torch.float32),
        torch.zeros((1,), dtype=torch.float32, device=device),
    )
    compressed_kv = (kv * score).sum(dim=1)  # [V, H] fp32

    # ── RMSNorm (fp32 throughout) ─────────────────────────────────────
    rms_w = rms_norm_weight[:h].to(torch.float32)
    variance = (compressed_kv * compressed_kv).sum(dim=-1) / h
    rrms = torch.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms.unsqueeze(1) * rms_w.unsqueeze(0)  # [V, H]

    # ── KV cache pointers (guard 3 already folded into `go`) ──────────
    kv_slot = kv_slot_all[vi]  # [V], all >= 0 by construction
    kv_block_idx = kv_slot // kv_cache_block_size
    kv_pos_in_block = kv_slot % kv_cache_block_size
    token_off = (
        kv_block_idx.to(torch.int64) * KV_BLOCK_STRIDE
        + kv_pos_in_block.to(torch.int64) * TOKEN_STRIDE
    )  # [V]

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    nope_h = h - rope_h
    num_pairs = (int(TRITON_BLOCK_SIZE) or h) // 2
    nope_pairs = nope_h // 2
    half_rope = rope_h // 2
    pair_idx = torch.arange(num_pairs, dtype=torch.int32, device=device)
    rope_local = pair_idx - nope_pairs  # [P]
    is_rope = rope_local >= 0  # [P]
    cs_idx = rope_local.clamp(min=0)  # [P] (non-rope pairs clamp to 0, in-bounds)
    comp_pos = (pos_v // cr * cr)  # [V] int32
    cs_cos = cos_sin_cache[
        comp_pos.to(torch.int64).unsqueeze(1), cs_idx.to(torch.int64).unsqueeze(0)
    ]  # [V, P]
    cs_sin = cos_sin_cache[
        comp_pos.to(torch.int64).unsqueeze(1),
        (half_rope + cs_idx).to(torch.int64).unsqueeze(0),
    ]  # [V, P]
    cos_v = torch.where(
        is_rope.unsqueeze(0),
        cs_cos.to(torch.float32),
        torch.ones((1,), dtype=torch.float32, device=device),
    )  # [V, P]
    sin_v = torch.where(
        is_rope.unsqueeze(0),
        cs_sin.to(torch.float32),
        torch.zeros((1,), dtype=torch.float32, device=device),
    )  # [V, P]

    normed_2d = normed.reshape(-1, num_pairs, 2)
    even = normed_2d[..., 0]  # [V, P]
    odd = normed_2d[..., 1]
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = torch.stack([new_even, new_odd], dim=-1).reshape(-1, h)  # interleaved

    if fp8_cache:
        _store_fp8(k_cache, kv_block_idx, kv_pos_in_block,
                   kv_cache_block_size, result, rope_h)
    else:
        # ── final store (bf16) ────────────────────────────────────────────
        def _write_flat(target):
            dev = target.device
            idx = (token_off.to(dev).unsqueeze(1) + cols.to(dev).unsqueeze(0))
            flat = _flat_element_view(target)
            flat.index_put_((idx,), result.to(target.dtype).to(dev))

        # ``k_cache.reshape(-1)`` is a copy on a packed (page-strided) window:
        # the store would land in the copy and be lost, after staging the whole
        # pool.  ``_flat_element_view`` indexes the backing instead, and for a
        # packed pool the write goes through a host mirror (``_write_packed``).
        if not _write_packed(k_cache, _write_flat):
            _write_flat(k_cache)


def _store_fp8_model1(
    k_cache: torch.Tensor,
    kv_block_idx: torch.Tensor,
    kv_pos_in_block: torch.Tensor,
    kv_cache_block_size: int,
    result: torch.Tensor,
    rope_h: int,
) -> None:
    """Store ``result[valid]`` into the MODEL1 SEGREGATED FP8 cache layout.

    ``k_cache`` is ``[num_blocks, block_size, 584]`` uint8 (fp8_ds_mla); the
    on-chip layout is SEGREGATED per page -- the page's ``block_size*576``
    token-data bytes come first (row t at flat offset ``t*576``), then its
    ``block_size*8`` E8M0 scale bytes (row t at ``block_size*576 + t*8``) --
    matching ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert._insert_fp8_model1``
    and the upstream Triton compress kernel.  Index through flat
    ``[num_blocks, block_size*584]`` views (NOT ``k_cache[b, p, :]``, whose 584
    stride interleaves scales into the data region the reader cannot see).

    The first ``h - rope_h`` (nope) dims are E4M3-block-quantized
    (``quant_e4m3_blocks``, 64-wide blocks, 7 real scales + 1 pad byte); the
    last ``rope_h`` dims stay bf16.  Every row of ``result`` has a non-negative
    KV slot (the caller folds that guard into one mask), so all rows write.
    """
    from ...local.e4m3 import quant_e4m3_blocks

    rows = result  # [W, h] fp32
    num_blocks, block_size = k_cache.shape[0], k_cache.shape[1]
    h = rows.shape[1]
    nope_h = h - rope_h

    nope = rows[:, :nope_h]          # [W, nope_h] fp32
    rope_bf16 = rows[:, nope_h:].to(torch.bfloat16)  # [W, rope_h]

    # quant_e4m3_blocks: [W, K] -> (q [W, K] uint8, s [W, K//64] uint8).
    q_nope, s_nope = quant_e4m3_blocks(nope, block=64)
    rope_u8 = rope_bf16.contiguous().view(torch.uint8)

    # MODEL1 per-token width: nope fp8 bytes + rope bf16 bytes + 8 scale bytes.
    tok_bytes = nope_h + 2 * rope_h + 8
    data_width = nope_h + 2 * rope_h  # 576 for head=512
    n_scale = s_nope.shape[1]  # 7 real + 1 pad byte

    data_row = torch.empty(
        (rows.shape[0], data_width), dtype=torch.uint8, device=k_cache.device
    )
    data_row[:, :nope_h] = q_nope
    data_row[:, nope_h:] = rope_u8
    # Zero-initialised, not ``empty``: only the real E8M0 scale bytes are
    # written below, so the pad byte would hold whatever the allocator left
    # there -- a different byte per run.  The layout reserves it, the vendored
    # kernel stores 0 into it, and this op own test compares whole rows.
    scale_row = torch.zeros(
        (rows.shape[0], n_scale + 1), dtype=torch.uint8, device=k_cache.device
    )
    scale_row[:, :n_scale] = s_nope

    bs = int(kv_cache_block_size)

    def _write(target):
        dev = target.device
        flat = target.reshape(num_blocks, bs * tok_bytes)
        data_region = flat[:, : bs * data_width].view(num_blocks, bs, data_width)
        scale_region = flat[:, bs * data_width:].view(num_blocks, bs, n_scale + 1)
        b_idx = kv_block_idx.to(torch.int64).to(dev)
        p_idx = kv_pos_in_block.to(torch.int64).to(dev)
        data_region[b_idx, p_idx] = data_row.to(dev)
        scale_region[b_idx, p_idx] = scale_row.to(dev)

    # Packed pool (vLLM >= 0.25): the scatter above stages the whole backing on
    # LPU -- see ``_write_packed``.
    if not _write_packed(k_cache, _write):
        _write(k_cache)


def _store_fp8_indexer(
    k_cache: torch.Tensor,
    kv_block_idx: torch.Tensor,
    kv_pos_in_block: torch.Tensor,
    kv_cache_block_size: int,
    result: torch.Tensor,
) -> None:
    """Store ``result[valid]`` into the FP8 indexer cache (132B/token).

    ``k_cache`` is ``[num_blocks, block_size, 132]`` uint8 (DeepseekV4Indexer
    cache, alignment=576).  The on-chip layout is SEGREGATED per page,
    matching the upstream Triton kernel
    ``_fused_kv_compress_norm_rope_insert_indexer_attn``:

      [0, bs*128)        : FP8 E4M3 data, token ``pos`` at flat offset
                           ``pos*TOKEN_STRIDE`` (TOKEN_STRIDE = head_dim = 128)
      [bs*128, +bs*4)    : one fp32 ``exp2(exponent)`` scale per token at
                           ``bs*128 + pos*4`` (SCALE_DIM = 4)

    Quant is a SINGLE flat block over the full 128-dim head (QUANT_BLOCK ==
    TRITON_BLOCK_SIZE == 128, flat ``tl.max`` reduction in the kernel), with
    the fp32->bf16->fp32 round trip before the absmax.  The scale is the
    single fp32 ``exp2(exponent)`` (NOT the E8M0 byte form the MODEL1 path
    stores) -- exactly what ``_fp8_gather_k_pages`` in ``sparse_attn_indexer``
    reads back as ``[T] fp32``.

    Index through flat ``[num_blocks, bs*132]`` views (NOT ``k_cache[b, p, :]``,
    whose 132 stride interleaves scales into the data region).
    """
    rows = result  # [W, 128] fp32
    num_blocks, block_size = k_cache.shape[0], k_cache.shape[1]
    h = rows.shape[1]
    if h != 128:
        raise NotImplementedError(
            f"FP8 indexer store assumes head_dim=128, got {h}"
        )

    # Match the Triton kernel quant exactly: fp32 -> bf16 -> fp32 round trip,
    # amax = max(|x|, 1e-4), exponent = ceil(log2(amax/448)), then divide by
    # 2^-e and cast directly to E4M3 (the float8 cast rounds to the grid
    # itself -- NO extra torch.round, which would double-round).
    bf16_rt = rows.to(torch.bfloat16).to(torch.float32)
    amax = bf16_rt.abs().amax(dim=-1).clamp_min(1e-4)       # [W]
    exponent = torch.ceil(torch.log2(amax / 448.0))          # [W] fp32
    inv_scale = torch.exp2(-exponent).unsqueeze(1)           # [W, 1]
    x_scaled = torch.clamp(bf16_rt * inv_scale, -448.0, 448.0)
    from ...local.e4m3 import _encode_e4m3

    q_fp8 = _encode_e4m3(x_scaled)  # [W, 128] uint8 (E4M3 grid rounding)
    scale_fp32 = torch.exp2(exponent)                        # [W] fp32
    scale_bytes = scale_fp32.contiguous().view(torch.uint8).reshape(-1, 4)  # [W,4]

    bs = int(kv_cache_block_size)

    def _write(target):
        dev = target.device
        flat = target.reshape(num_blocks, bs * 132)
        data_region = flat[:, : bs * 128].view(num_blocks, bs, 128)
        scale_region = flat[:, bs * 128:].view(num_blocks, bs, 4)
        b_idx = kv_block_idx.to(torch.int64).to(dev)
        p_idx = kv_pos_in_block.to(torch.int64).to(dev)
        data_region[b_idx, p_idx] = q_fp8.to(dev)
        scale_region[b_idx, p_idx] = scale_bytes.to(dev)

    # Packed pool (vLLM >= 0.25): the scatter above stages the whole backing on
    # LPU -- see ``_write_packed``.
    if not _write_packed(k_cache, _write):
        _write(k_cache)


def _store_fp8(
    k_cache: torch.Tensor,
    kv_block_idx: torch.Tensor,
    kv_pos_in_block: torch.Tensor,
    kv_cache_block_size: int,
    result: torch.Tensor,
    rope_h: int,
) -> None:
    """Dispatch the FP8 store by head size.

    - ``h == 128`` (sparse-indexer cache, 132B/token): the whole head is FP8
      with a single flat 128-wide quant block and one fp32 ``exp2(exponent)``
      scale per token (``_store_fp8_indexer``).
    - otherwise (MODEL1 fp8_ds_mla cache, 584B/token): nope E4M3-block-quant +
      bf16 rope (``_store_fp8_model1``).
    """
    h = result.shape[1]
    if h == 128:
        _store_fp8_indexer(
            k_cache, kv_block_idx, kv_pos_in_block, kv_cache_block_size,
            result,
        )
    else:
        _store_fp8_model1(
            k_cache, kv_block_idx, kv_pos_in_block, kv_cache_block_size,
            result, rope_h,
        )


def _lpu_save_partial_states_024(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
) -> None:
    """0.24 high-level replacement for ``common/ops/save_partial_states``.

    Maps the 0.24 function signature onto the triton-kernel-style
    ``_lpu_save_partial_states`` (identical computation; stride/constexpr
    args are accepted and mostly unused).
    """
    del pdl_kwargs
    h = kv.shape[-1]
    _lpu_save_partial_states(
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        block_size,
        HEAD_SIZE=h,
        TRITON_BLOCK_SIZE=h,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
    )


def _lpu_compress_norm_rope_store_triton(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """0.24 replacement for ``common/ops/fused_compress_quant_cache``'s
    ``compress_norm_rope_store_triton`` launcher.

    The 0.24 launcher's kwarg signature maps 1:1 onto the triton-kernel-style
    ``_lpu_fused_kv_compress_norm_rope_insert`` positional contract (same
    upstream kernel); the FP8 vs bf16 store is auto-detected from the cache
    dtype (uint8 MODEL1 → fp8; bf16 → plain-row).  ``launch_pdl`` is ignored.
    """
    del num_actual, pdl_kwargs, use_fp4_cache, quant_block, scale_dim
    fp8_cache = kv_cache.dtype == torch.uint8
    _lpu_fused_kv_compress_norm_rope_insert(
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=head_dim,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        TOKEN_STRIDE=token_stride,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        fp8_cache=fp8_cache,
    )


def _sweep_and_check_gcs(*, context: str) -> None:
    """Rebind every remaining ``get_compressed_slot_mapping`` alias and verify
    that none still points at the upstream Triton function.

    Generic sweep: upstream modules bind the name with
    ``from ... import get_compressed_slot_mapping`` (module-local alias), so a
    hand-written module list silently misses call sites -- e.g.
    ``vllm.models.deepseek_v4.sparse_mla`` (imports it at module top, calls it
    at line ~221).  One un-rebound alias is enough: it leaves the upstream
    Triton kernel launched, and since that kernel is the only Triton launch in
    an otherwise all-evas run, its presence IS the signature of a missed alias.
    It is also the known evas pointwise compile dead-end (block-table
    multi-block liveness; silently never writes on LPU), and that then wedges
    the copy engine for the next TP collective.

    Runs at patch-install time AND lazily on the stand-in's first call: the sweep
    can only see *already imported* modules, so a module imported after patch
    application (upstream import-order drift) would otherwise keep the upstream
    alias silently.  By first real use every module is loaded.

    The identity check uses ``_UPSTREAM_GCS`` (captured once, before the first
    rebind) so re-entry cannot mistake our own stand-in for the upstream one.
    """
    import sys as _sys

    _rebound = []
    for _name, _mod in list(_sys.modules.items()):
        if _mod is None or _name.startswith("torch"):
            continue
        _cur = getattr(_mod, "get_compressed_slot_mapping", None)
        if _cur is None or _cur is _lpu_compressed_slot_mapping:
            continue
        try:
            setattr(_mod, "get_compressed_slot_mapping", _lpu_compressed_slot_mapping)
        except Exception:  # noqa: BLE001 - some modules are not writable
            continue
        _rebound.append(_name)
    print(
        f"[dsv4-attn] get_compressed_slot_mapping rebound in "
        f"{len(_rebound)} modules: {_rebound} (context={context})",
        flush=True,
    )

    # Post-sweep self-check: a leaked alias must be a LOUD failure, not a silent
    # run of the upstream Triton kernel.  Re-scan every module -- including the
    # ``torch`` namespace, which the sweep deliberately does not write into, so a
    # leak there is reported but not fixed.
    if _UPSTREAM_GCS is None or _UPSTREAM_GCS is _lpu_compressed_slot_mapping:
        # No usable capture: either the installer never ran (None) or the
        # installer already rebound before capturing (our own stand-in).  Either
        # way the identity comparison below would be meaningless -- and with a
        # None capture every module lacking the attribute would look like a leak.
        print("[dsv4-attn] self-check skipped (no usable upstream capture)",
              flush=True)
        return
    _leaks = []
    for _name, _mod in list(_sys.modules.items()):
        if _mod is None or _name == __name__:
            continue
        try:
            _cur = getattr(_mod, "get_compressed_slot_mapping", None)
        except Exception:  # noqa: BLE001 - module proxies may raise on getattr
            continue
        if _cur is not None and _cur is _UPSTREAM_GCS:
            _leaks.append(_name)
    if _leaks:
        logger.error(
            "[dsv4-attn] upstream get_compressed_slot_mapping is STILL bound in "
            "%d module(s): %s (context=%s) -- on LPU that Triton kernel silently "
            "never writes and wedges the copy engine for the next TP collective.",
            len(_leaks),
            _leaks,
            context,
        )
        if os.environ.get("VLLM_FL_STRICT_UPSTREAM_REBIND") == "1":
            raise RuntimeError(
                "upstream get_compressed_slot_mapping still bound in "
                f"{_leaks} (VLLM_FL_STRICT_UPSTREAM_REBIND=1)"
            )
    else:
        print(
            f"[dsv4-attn] self-check OK: no upstream "
            f"get_compressed_slot_mapping alias left (context={context})",
            flush=True,
        )


def _install_compressor_kernel_overrides() -> None:
    """Rebind DeepseekCompressor's Triton kernels to pure-PyTorch stand-ins.

    vllm 0.24's ``DeepseekCompressor.forward`` calls two module-level
    functions, not Triton kernels by name:

    - ``vllm.models.deepseek_v4.common.ops.save_partial_states.save_partial_states``
      (writes packed [kv, score+ape] partial states into the compressor cache).
    - ``vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache.compress_norm_rope_store_triton``
      (fused compress → RMSNorm → RoPE → FP8 quant → KV cache insert).

    Both launch upstream Triton kernels that fail to compile on LPU
    (``cf.cond_br`` guards; ``launch_pdl`` kwarg rejected by the evas FlagTree
    JIT); both are rebound to the pure-PyTorch stand-ins here.  The compressed
    slot_mapping Triton kernel (compressor_utils) is a block-table
    multi-block liveness pointwise dead-end on evas; it silently never writes,
    leaving the metadata slot_mapping all -1 and the compressor never storing
    compressed KV -- rebind to the pure-PyTorch stand-in.  flashmla_sparse /
    indexer bind the name via ``from ... import``, so the module-local aliases
    are rebinding targets too.
    """
    import importlib

    from vllm.v1.attention.backends.mla import compressor_utils

    for _mod_name, _attr in (
        (
            "vllm.models.deepseek_v4.common.ops.save_partial_states",
            "save_partial_states",
        ),
        (
            "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache",
            "compress_norm_rope_store_triton",
        ),
        # ``DeepseekCompressor.forward`` imports both names into its own module
        # namespace at import time, so the compressor module aliases must be
        # rebound too (the patch runs after vllm is imported).
        ("vllm.models.deepseek_v4.compressor", "save_partial_states"),
        ("vllm.models.deepseek_v4.compressor", "compress_norm_rope_store_triton"),
    ):
        _mod = importlib.import_module(_mod_name)
        if _attr == "save_partial_states":
            setattr(_mod, _attr, _lpu_save_partial_states_024)
        else:
            setattr(_mod, _attr, _lpu_compress_norm_rope_store_triton)

    global _UPSTREAM_GCS
    if _UPSTREAM_GCS is None:
        _UPSTREAM_GCS = compressor_utils.get_compressed_slot_mapping
    compressor_utils.get_compressed_slot_mapping = _lpu_compressed_slot_mapping
    _sweep_and_check_gcs(context="install")

def apply_lpu_dsv4_compressor_patch() -> None:
    """Install the pure-PyTorch compressor / C128A metadata overrides on LPU.

    DSV4 attention is the upstream ``DeepseekV4Attention`` ABC
    (``vllm/models/deepseek_v4/attention.py``); on LPU the concrete subclass is
    ``EvasDSV4Attention``.  Its sequential ``attention_impl`` rides on the base
    compressor + sparse MLA metadata.  The base ``DeepseekCompressor`` Triton
    kernels and the C128A topk metadata kernel do not compile on LPU
    (evas-triton-opt), so this patch rebinds them to the pure-PyTorch
    stand-ins:

    - ``_install_c128a_topk_metadata_override`` — the C128A
      (compress_ratio == 128) decode/prefill topk metadata in
      ``vllm/models/deepseek_v4/sparse_mla.py``.
    - ``_install_compressor_kernel_overrides`` — ``DeepseekCompressor``'s
      fused compress+quant+cache-insert Triton kernels + the compressed
      slot_mapping in ``compressor_utils``.

    Idempotent.  Applied unconditionally on LPU: the rebindings are a no-op
    unless the DSV4 attention path actually runs.
    """
    global _GENERIC_PATCHED
    if _GENERIC_PATCHED:
        return
    _install_c128a_topk_metadata_override()
    _install_compressor_kernel_overrides()
    _GENERIC_PATCHED = True

# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU metadata-construction shims for the upstream DSV4 sparse-SWA builder.

Four pure-PyTorch stand-ins bound on ``sparse_swa`` / ``flashmla`` /
``indexer``: ``get_mla_metadata``, the SWA indices/lens kernel, the
prefill-metadata kernel, and the indexer prefill chunk-metadata kernel — the
last binds its positional tail by arity, serving 0.24 and 0.25
signatures.  The kernel shims carry a Triton-style launch wrapper so upstream
``build()`` sites work unchanged; index geometry is computed host-side in
int64.  One env-gated debug branch sits here.

"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_META_SHIMS_INSTALLED = False


# =============================================================================
# Pure-PyTorch kernel implementations (installed via _install_metadata_shims)
# =============================================================================
# The upstream builders need three LPU-unavailable dependencies:
#   1. ``flashmla.get_mla_metadata`` — FlashMLA C-ext; on non-CUDA platforms vllm
#      replaces it with ``_raise_flashmla_unavailable`` (RuntimeError).
#   2. ``_compute_swa_indices_and_lens_kernel`` /
#      ``_compute_prefill_metadata_kernel`` — Triton kernels that abort
#      ``evas-triton-opt`` at compile time (SetMemRefScopePass).
#   3. ``_build_prefill_chunk_metadata_kernel`` — Triton kernel that writes OOB
#      on LPU (zeroing adjacent device tensors; see module docstring).
# The kernel replacements are wrapped in a Triton-style launch shim
# (``kernel[(grid,)](args...)``) so upstream ``build()`` call sites work
# unchanged — module-attribute redirect only, no upstream edit.

# ──────────────────────────────────────────────────────────────────────────
# Kernel-1: _compute_swa_indices_and_lens_kernel
# ──────────────────────────────────────────────────────────────────────────


def _swa_windows(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    window_size: int,
):
    """Shared per-token window geometry for the SWA indices/lens computation.

    Returns ``(is_valid, swa_len, start_pos, end_pos)`` all shaped
    ``[num_tokens]`` (int64). Mirrors the kernel body:

        req_idx = token_to_req[token_idx]
        query_start = query_start_loc[req_idx]
        query_end   = query_start_loc[req_idx + 1]
        query_len   = query_end - query_start
        seq_len     = seq_lens[req_idx]
        prefix_len  = seq_len - query_len
        pos         = prefix_len + token_idx - query_start
        start_pos   = max(pos - window_size + 1, 0)
        end_pos     = pos + 1
        swa_len     = end_pos - start_pos
    """
    device = query_start_loc.device
    # CPU metadata offload (module docstring): int64 index math must run on CPU.
    query_start_loc = query_start_loc.cpu()
    seq_lens = seq_lens.cpu()
    token_to_req_indices = token_to_req_indices.cpu()
    is_valid_token = is_valid_token.cpu()
    num_tokens = is_valid_token.shape[0]
    req_idx = token_to_req_indices.to(torch.int64)
    token_arange = torch.arange(num_tokens, dtype=torch.int64)
    qs = query_start_loc[req_idx].to(torch.int64)
    qe = query_start_loc[req_idx + 1].to(torch.int64)
    query_len = qe - qs
    seq_len = seq_lens[req_idx].to(torch.int64)
    prefix_len = seq_len - query_len
    pos = prefix_len + token_arange - qs
    # CPU tensor (offloaded above) — int64 clamp is safe here.
    start_pos = torch.clamp(pos - window_size + 1, min=0)
    end_pos = pos + 1
    swa_len = end_pos - start_pos
    is_valid = is_valid_token
    # The kernel early-returns with lens=0 for invalid tokens; keep a valid-mask
    # lens (used for both the lens output and the -1 slot padding).
    lens_for_valid = torch.where(
        is_valid, swa_len, torch.zeros_like(swa_len)
    )
    return (
        is_valid.to(device),
        lens_for_valid.to(device),
        start_pos.to(device),
        end_pos.to(device),
    )


def _decode_swa_indices_from_windows(
    t2r: torch.Tensor,
    bt: torch.Tensor,
    block_size: int,
    window_size: int,
    is_valid: torch.Tensor,
    lens_for_valid: torch.Tensor,
    start_pos: torch.Tensor,
) -> torch.Tensor:
    """Flat SWA slot ids from precomputed ``_swa_windows`` geometry (CPU).

    Shared with ``_compute_swa_indices_and_lens_evas`` so the window geometry
    is computed once instead of re-derived per output.
    """
    req_idx = t2r.to(torch.int64)

    offsets = torch.arange(window_size, dtype=torch.int64)
    pos_offset = start_pos.unsqueeze(1) + offsets.unsqueeze(0)  # [T, W]
    block_indices = pos_offset // block_size
    # Masked-out entries may exceed the block_table row length; clamp before
    # the gather (their slot ids are overwritten with -1 below).
    max_blocks = bt.shape[1]
    # CPU tensor — int64 clamp is safe here.
    clamped = block_indices.clamp(0, max_blocks - 1)
    block_numbers = bt[req_idx].gather(1, clamped)  # [T, W]
    block_offsets = pos_offset % block_size
    slot_ids = block_numbers.to(torch.int64) * block_size + block_offsets

    # offset < swa_len  <=>  pos_offset < end_pos  (valid window slots).
    in_window = offsets.unsqueeze(0) < lens_for_valid.unsqueeze(1)
    slot_ids = torch.where(in_window, slot_ids, torch.full_like(slot_ids, -1))

    # Invalid tokens: the kernel leaves the row untouched (builder init = 0).
    slot_ids = torch.where(
        is_valid.unsqueeze(1), slot_ids, torch.zeros_like(slot_ids)
    )
    return slot_ids.to(torch.int32)


def _compute_swa_indices_and_lens_evas(
    swa_indices: torch.Tensor,
    swa_indices_stride: int,
    swa_lens: torch.Tensor,
    window_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    *,
    _launch_grid: tuple[int, ...] | None = None,
    **_kwargs,
) -> None:
    """Drop-in for ``sparse_swa._compute_swa_indices_and_lens_kernel``.

    Same positional signature as the Triton kernel; writes ``swa_lens`` and the
    flat ``swa_indices`` rows in place (``swa_indices[token_idx * stride + offset]``).
    The ``TRITON_BLOCK_SIZE`` constexpr kwarg is accepted and ignored.
    """
    del block_table_stride  # rectangular gather indexes columns directly
    device = swa_indices.device

    # The upstream kernel is launched with grid=(num_decode_tokens,) and reads
    # token_idx = program_id(0) in [0, num_decode_tokens). Bound the computation
    # by that grid (NOT is_valid_token.shape[0] = slot_mapping.shape[0], the full
    # token count) so token_arange can never index past token_to_req_indices —
    # matching upstream exactly. Decodes are the first tokens of the batch, so
    # slicing the token-indexed inputs to the grid reproduces the kernel's
    # per-token geometry bit-identically. Direct calls (no launch grid) keep the
    # full-length behaviour.
    if _launch_grid is not None:
        num_tokens = int(_launch_grid[0])
    else:
        num_tokens = is_valid_token.shape[0]
    is_valid_token = is_valid_token[:num_tokens]
    token_to_req_indices = token_to_req_indices[:num_tokens]

    # The upstream builder writes the t2r / qsl / seq_lens device tensors via
    # ``copy_(x, non_blocking=True)`` from a TEMPORARY pinned tensor
    # (sparse_swa.build), and this shim then D2H-reads them immediately.  If the
    # LPU copy path does not stream-order those bytes before the reads, the
    # inputs arrive pre-copy/stale: down the line that surfaces as an
    # inconsistent-metadata IndexError on the host, or as garbage consumed by a
    # device kernel (the upstream Triton kernel would read it silently, no
    # bounds check).  Sync the device so the H2D lands before any read (same
    # idiom as patches/ops_fallback/block_table.py).
    torch.lpu.synchronize()
    if os.environ.get("VLLM_FL_DSV4_DEBUG_ARGS"):
        t2r_v = token_to_req_indices.cpu()

        def _layout(t):
            try:
                return f"split_dim={t.split_dim()}"
            except Exception as e:  # noqa: BLE001
                return f"layout-err({type(e).__name__})"

        print(
            f"[swa-evas] grid={_launch_grid} num_tokens={num_tokens} "
            f"win={window_size} "
            f"qsl={query_start_loc.cpu().tolist()} "
            f"seq_lens={seq_lens.cpu().tolist()} "
            f"t2r[:{min(num_tokens, 24)}]={t2r_v[:24].tolist()} "
            f"t2r_max={int(t2r_v.max())} "
            f"is_valid_n={is_valid_token.shape[0]} "
            f"qsl[{_layout(query_start_loc)}] "
            f"t2r[{_layout(token_to_req_indices)}] "
            f"bt[{_layout(block_table)}]",
            flush=True,
        )

    # CPU metadata offload (module docstring); the helpers are CPU-device safe.
    #
    # H2D is NOT ordered with D2H reads on LPU (torch.lpu.synchronize() does
    # not cover the copy engine): two consecutive .cpu() reads of the SAME
    # device qsl can return different values (e.g. [128, 0] then [0, 5]).  So
    # spin-read until the landed value satisfies the invariants every legit
    # CommonAttentionMetadata qsl has: qsl[0] == 0, non-decreasing, and
    # sum(diff) >= num_tokens (the launch grid covers a subset of the batch
    # tokens).  t2r is then rebuilt from the verified qsl.
    qsl_cpu = query_start_loc.cpu()
    for _attempt in range(32):
        _qsl_sum = (
            int((qsl_cpu[1:] - qsl_cpu[:-1]).sum()) if qsl_cpu.numel() >= 2 else 0
        )
        _ok = (
            qsl_cpu.numel() >= 2
            and int(qsl_cpu[0]) == 0
            and bool((qsl_cpu[1:] >= qsl_cpu[:-1]).all())
            and _qsl_sum >= num_tokens
        )
        if _ok:
            if _attempt:
                logger.warning(
                    "[swa-evas] qsl read raced the H2D (LPU): landed after "
                    "%d retries, qsl=%s", _attempt, qsl_cpu.tolist(),
                )
            break
        torch.lpu.synchronize()
        qsl_cpu = query_start_loc.cpu()
    else:
        logger.error(
            "[swa-evas] qsl never satisfied invariants after 32 reads "
            "(grid=%s, last=%s); proceeding with last read", _launch_grid,
            qsl_cpu.tolist(),
        )
    seq_lens_cpu = seq_lens.cpu()
    t2r_cpu = token_to_req_indices.cpu()
    is_valid_cpu = is_valid_token.cpu()
    bt_cpu = block_table.cpu()

    # The device t2r buffer can be corrupted BETWEEN consecutive shim calls
    # (same worker, same batch: read correct at call N, `[0, 0, 16843009,
    # 16843009, ...]` at call N+1 — AFTER a device sync).  qsl/seq_lens
    # (long-lived pinned sources) are correct in every dump; t2r's source is a
    # TEMPORARY pinned tensor (sparse_swa.build `x.pin_memory()` →
    # `copy_(x, non_blocking=True)`), the only racy H2D in the build path.  t2r
    # is fully determined by qsl (t2r[i] = owning request of token i), so
    # rebuild it on CPU from the always-correct qsl and use that; the device
    # value is only compared (mismatch ⇒ the device copy is untrustworthy on
    # LPU) and never used.
    num_reqs_rc = qsl_cpu.shape[0] - 1
    t2r_rebuilt = torch.repeat_interleave(
        torch.arange(num_reqs_rc, dtype=torch.int64),
        (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int64),
    )
    if t2r_rebuilt.shape[0] < num_tokens:
        # Padded tail (past the last real token): repeat the last request,
        # mirroring the upstream padding semantics of query_start_loc.
        pad = torch.full(
            (num_tokens - t2r_rebuilt.shape[0],),
            num_reqs_rc - 1,
            dtype=torch.int64,
        )
        t2r_rebuilt = torch.cat([t2r_rebuilt, pad])
    t2r_rebuilt = t2r_rebuilt[:num_tokens]
    if not torch.equal(t2r_rebuilt, t2r_cpu.to(torch.int64)):
        logger.warning(
            "[swa-evas] device token_to_req_indices mismatches qsl-derived "
            "values (LPU H2D corruption); using qsl-derived. grid=%s "
            "t2r_dev[:8]=%s t2r_qsl[:8]=%s",
            _launch_grid,
            t2r_cpu[:8].tolist(),
            t2r_rebuilt[:8].tolist(),
        )
    t2r_cpu = t2r_rebuilt

    # Repair the DEVICE buffer, not just the local math: sparse_swa.build puts
    # this same (corrupted) device tensor into the attention metadata (build's
    # `token_to_req_indices=...`), so downstream kernels would consume the bad
    # H2D data even with the shim math corrected. Blocking copy — the source is
    # materialized eagerly, unlike the temporary-pinned `non_blocking` pattern
    # this shim works around.
    token_to_req_indices.copy_(
        t2r_rebuilt.to(token_to_req_indices.dtype).to(token_to_req_indices.device)
    )

    # Single window-geometry pass, shared by the indices and lens outputs.
    is_valid, lens_for_valid, start_pos, _ = _swa_windows(
        qsl_cpu, seq_lens_cpu, t2r_cpu, is_valid_cpu, window_size,
    )
    idx = _decode_swa_indices_from_windows(
        t2r_cpu, bt_cpu, block_size, window_size,
        is_valid, lens_for_valid, start_pos,
    )  # [T, W] int32 CPU
    lens = lens_for_valid.to(torch.int32)  # [T] int32 CPU

    swa_lens[:num_tokens].copy_(lens.to(device))

    # Faithful pointer-math write: swa_indices_ptr + token_idx*stride + offset;
    # rows are distinct (stride >= window_size), so the flat scatter is
    # unambiguous. 1D int64 index + int32 payload are the on-device pattern.
    flat = swa_indices.reshape(-1)
    token_off = torch.arange(
        num_tokens, dtype=torch.int64
    ) * swa_indices_stride
    flat_offsets = token_off.unsqueeze(1) + torch.arange(
        window_size, dtype=torch.int64
    ).unsqueeze(0)  # [T, W] CPU
    # The kernel leaves invalid-token rows untouched (keeps whatever was there).
    if is_valid_cpu.any():
        sel_indices = flat_offsets[is_valid_cpu].reshape(-1).to(device)
        flat[sel_indices] = idx[is_valid_cpu].reshape(-1).to(device)


# ──────────────────────────────────────────────────────────────────────────
# Kernel-2: _compute_prefill_metadata_kernel
# ──────────────────────────────────────────────────────────────────────────


def prefill_gather_lens(
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
    window_size: int,
) -> torch.Tensor:
    """``[num_prefills]`` int32 prefill gather length per prefill request.

    Pure-PyTorch equivalent of ``_compute_prefill_metadata_kernel``:

        seq_len    = seq_lens[num_decodes + i]
        query_len  = query_start_loc[num_decodes+i+1] - query_start_loc[num_decodes+i]
        prefix_len = seq_len - query_len
        gather_len = query_len + min(prefix_len, window_size - 1)
    """
    device = query_start_loc.device
    # CPU metadata offload (module docstring); move the int32 result back.
    seq_lens = seq_lens.cpu()
    query_start_loc = query_start_loc.cpu()
    idx = torch.arange(num_prefills, dtype=torch.int64) + num_decodes
    seq_len = seq_lens[idx].to(torch.int64)
    qsl_start = query_start_loc[idx].to(torch.int64)
    qsl_end = query_start_loc[idx + 1].to(torch.int64)
    query_len = qsl_end - qsl_start
    prefix_len = seq_len - query_len
    gather_len = query_len + torch.minimum(
        prefix_len, torch.full_like(prefix_len, window_size - 1)
    )
    return gather_len.to(torch.int32).to(device)


def _compute_prefill_metadata_evas(
    pfx_gather_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
    window_size: int,
    **_kwargs,
) -> None:
    """Drop-in for ``sparse_swa._compute_prefill_metadata_kernel``.

    Same positional signature as the Triton kernel; writes
    ``pfx_gather_lens`` in place. The ``BLOCK_SIZE`` constexpr kwarg is
    accepted and ignored.
    """
    device = pfx_gather_lens.device
    lens = prefill_gather_lens(
        seq_lens.cpu(), query_start_loc.cpu(),
        num_prefills, num_decodes, window_size,
    )  # [num_prefills] int32 CPU
    # Diagnostic gate: this H2D write into a metadata device
    # buffer is PREFILL-EXCLUSIVE (the decode path never calls this shim), and
    # the TP4 wedge sits at the first TP collective AFTER prefill-flavoured
    # metadata prep.  Skipping it tests whether the prep writeback is what
    # leaves the copy engine stuck.  Diagnostic only (metadata would be stale).
    if os.environ.get("VLLM_FL_SKIP_PREFILL_META_WRITEBACK") == "1":
        print("[meta-wb-skip] skipped pfx_gather_lens writeback", flush=True)
    else:
        pfx_gather_lens[:num_prefills].copy_(lens.to(device))


# ──────────────────────────────────────────────────────────────────────────
# Kernel-3: _build_prefill_chunk_metadata_kernel (indexer prefill chunks)
# ──────────────────────────────────────────────────────────────────────────


def _build_prefill_chunk_metadata_evas(
    query_start_loc: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *rest,
    COMPRESS_RATIO: int = 0,
    **_kwargs,
) -> None:
    """Drop-in for ``indexer._build_prefill_chunk_metadata_kernel``.

    The upstream Triton kernel (``mla/indexer.py``, used by
    ``build_prefill_chunk_metadata``) writes OOB on LPU, zeroing the adjacent
    ``seq_lens``/``query_start_loc`` device tensors in ``CommonAttentionMetadata``
    (same LPU Triton OOB pattern as deepseek_rope/block_table). Re-expressed in
    pure PyTorch with identical semantics, per request ``b``:

        seq_start  = cu_seq_lens[b]
        seq_len_c  = cu_seq_lens[b+1] - cu_seq_lens[b]
        row_start  = row_start_csl[b]              # DCP-local base, see below
        start_pos  = uncompressed_seq_lens[b] - query_len[b]
        # query-token writes (only tokens in [qs_start, qs_stop)):
        cu_seq_len_ks[out_pos] = row_start
        cu_seq_len_ke[out_pos] = row_start + (start_pos + 1 + i) // COMPRESS_RATIO
        # compressed-span scatter (always the GLOBAL span base):
        token_to_seq[seq_start + offset] = b   for offset in [0, seq_len_c)

    The positional tail is version-dependent, so the shim binds it by arity
    rather than by name -- vLLM 0.25 inserted ``local_cu_seq_lens`` as the
    row-start base for ``cu_seq_len_k*`` and appended three DCP scalars:

        0.24: (..., cu_seq_lens, token_to_seq, ks, ke, qs_start, qs_stop)      5
        0.25: (..., cu_seq_lens, local_cu_seq_lens, token_to_seq, ks, ke,
               qs_start, qs_stop, dcp_rank, dcp_world, dcp_interleave)         9

    ``local_cu_seq_lens`` aliases ``cu_seq_lens`` when DCP is off, so the
    non-DCP numerics are identical on both cores; ``token_to_seq`` keeps using
    the global span on either path. DCP itself is unsupported here -- the
    per-rank length formula is rejected explicitly instead of silently writing
    global bounds.

    All index geometry is computed in int64 on CPU (module docstring); only the
    final int32 payloads move to the device via bounded ``.copy_``.
    ``BLOCK_SIZE`` constexpr and the ``_launch_grid`` are accepted and ignored.
    """
    if len(rest) == 5:      # vLLM <= 0.24
        token_to_seq, cu_seq_len_ks, cu_seq_len_ke, qs_start, qs_stop = rest
        row_start_csl = cu_seq_lens
    elif len(rest) == 9:    # vLLM >= 0.25
        local_cu_seq_lens, token_to_seq, cu_seq_len_ks, cu_seq_len_ke, \
            qs_start, qs_stop, dcp_rank, dcp_world, dcp_interleave = rest
        if dcp_world != 1:
            raise NotImplementedError(
                "_build_prefill_chunk_metadata_evas: DCP is not adapted on LPU "
                f"(dcp_world={dcp_world}, dcp_rank={dcp_rank}, "
                f"dcp_interleave={dcp_interleave})"
            )
        row_start_csl = local_cu_seq_lens
    else:
        raise TypeError(
            "_build_prefill_chunk_metadata_evas: upstream positional signature "
            f"drifted (got {len(rest)} trailing args, expected 5 for <=0.24 or "
            "9 for >=0.25); re-check indexer._build_prefill_chunk_metadata_kernel"
        )

    num_reqs = query_start_loc.numel() - 1
    if num_reqs <= 0:
        return

    # CPU metadata offload.
    qsl = query_start_loc.cpu().to(torch.int64)
    usl = uncompressed_seq_lens.cpu().to(torch.int64)
    csl = cu_seq_lens.cpu().to(torch.int64)
    rsl = row_start_csl.cpu().to(torch.int64)
    q_lens = qsl[1:] - qsl[:-1]     # [R] query len per request
    seq_start = csl[:-1]            # [R] compressed seq start (global)
    seq_lens_c = csl[1:] - csl[:-1]  # [R] compressed len per request
    row_start = rsl[:-1]            # [R] DCP-local row base for cu_seq_len_k*
    start_pos = usl - q_lens        # [R] uncompressed prefix len

    out_len = int(qs_stop) - int(qs_start)
    if out_len > 0:
        # Map every query token to its request: repeat the request id over its
        # query length, then index the window [qs_start, qs_stop).
        req_per_token = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int64), q_lens
        )
        abs_pos = torch.arange(int(qs_start), int(qs_stop), dtype=torch.int64)
        req = req_per_token[abs_pos]      # [out_len] request per window token
        i = abs_pos - qsl[req]            # [out_len] local token offset
        ks = row_start[req]
        ke = ks + (start_pos[req] + 1 + i) // COMPRESS_RATIO
        if os.environ.get("VLLM_FL_SKIP_PREFILL_META_WRITEBACK") == "1":
            print("[meta-wb-skip] skipped cu_seq_len_k* writeback", flush=True)
        else:
            cu_seq_len_ks[:out_len].copy_(ks.to(torch.int32))
            cu_seq_len_ke[:out_len].copy_(ke.to(torch.int32))

    total_compressed = int(csl[-1].item())
    if total_compressed > 0:
        comp_req = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int64), seq_lens_c
        )
        if os.environ.get("VLLM_FL_SKIP_PREFILL_META_WRITEBACK") == "1":
            print("[meta-wb-skip] skipped token_to_seq writeback", flush=True)
        else:
            token_to_seq[:total_compressed].copy_(comp_req.to(torch.int32))


# ──────────────────────────────────────────────────────────────────────────
# Triton-style launch shim
# ──────────────────────────────────────────────────────────────────────────


class _TritonStyleKernel:
    """Pure-PyTorch stand-in supporting the Triton ``kernel[grid](args)``
    launch syntax used by the upstream ``build()`` call sites.

    ``kernel[(grid,)]`` returns a launcher that forwards to the wrapped pure
    PyTorch function (grid is only accepted for signature compatibility).
    """

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        return _TritonStyleLauncher(self._fn, grid)

    def __call__(self, *args, **kwargs):
        # Direct call (no grid): still valid — the wrapped pure-PyTorch function
        # needs no grid/program_id. Keeps the object callable() like a Triton
        # JITFunction.
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"<TritonStyleKernel {getattr(self._fn, '__name__', self._fn)}>"


class _TritonStyleLauncher:
    __slots__ = ("_fn", "_grid")

    def __init__(self, fn, grid):
        self._fn = fn
        self._grid = grid

    def __call__(self, *args, **kwargs):
        # Forward the launch grid: a Triton ``kernel[(grid,)]`` launch bounds
        # ``program_id(0)`` by ``grid[0]``, and the wrapped pure-PyTorch kernels
        # use it to bound their token loop exactly like the upstream Triton
        # kernel. Wrappers that don't consume it absorb it via ``**_kwargs``.
        return self._fn(*args, _launch_grid=self._grid, **kwargs)


def make_triton_style_kernel(fn):
    """Wrap a pure-PyTorch function so it is callable as ``fn[(grid,)](...)``."""
    return _TritonStyleKernel(fn)


# ──────────────────────────────────────────────────────────────────────────
# get_mla_metadata (FlashMLA C-ext stand-in)
# ──────────────────────────────────────────────────────────────────────────


def get_mla_metadata(*_args, **_kwargs):
    """Pure-PyTorch stand-in for ``vllm.v1.attention.ops.flashmla.get_mla_metadata``.

    Mirrors the real ``vllm.third_party.flashmla.flash_mla_interface.get_mla_metadata``:
    returns ``(FlashMLASchedMeta(), None)`` — a fresh empty meta whose fields
    default to the dataclass values (``have_initialized=False``, ``config=None``,
    ``tile_scheduler_metadata=None``, ``num_splits=None``). The actual tile
    scheduling metadata is generated on the first ``flash_mla_with_kvcache``
    invocation, so ``build_tile_scheduler`` (which calls ``get_mla_metadata()[0]``)
    gets a valid holder and ``FlashMLAMetadataBuilder``-style consumers that
    read/adjust the four fields work unchanged.

    Args/kwargs are accepted and ignored (old FlashMLA interface compatibility).
    """
    # Imported lazily: on non-CUDA platforms the class is a ``pass`` stub, so we
    # set the four real dataclass fields explicitly to match the documented
    # FlashMLASchedMeta contract.
    from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta

    meta = FlashMLASchedMeta()
    meta.have_initialized = False
    meta.config = None
    meta.tile_scheduler_metadata = None
    meta.num_splits = None
    return meta, None


__all__ = [
    "prefill_gather_lens",
    "get_mla_metadata",
    "make_triton_style_kernel",
    "_compute_swa_indices_and_lens_evas",
    "_compute_prefill_metadata_evas",
    "_build_prefill_chunk_metadata_evas",
]


# =============================================================================
# Patch installation (module-attribute redirects)
# =============================================================================


def _install_metadata_shims() -> None:
    """Redirect sparse_swa / flashmla / indexer module attributes to the evas
    pure-PyTorch metadata implementations. Idempotent."""
    global _META_SHIMS_INSTALLED
    import vllm.v1.attention.ops.flashmla as flashmla
    from vllm.v1.attention.backends.mla import indexer, sparse_swa

    if _META_SHIMS_INSTALLED:
        return  # already installed
    _META_SHIMS_INSTALLED = True

    # Module-attribute redirect: the three Triton index kernels become launch-
    # syntax-compatible shims; get_mla_metadata becomes the evas pure-PyTorch
    # impl on BOTH modules (sparse_swa holds the already-bound `from ... import`
    # name; flashmla covers any future import order).
    sparse_swa._compute_swa_indices_and_lens_kernel = make_triton_style_kernel(
        _compute_swa_indices_and_lens_evas
    )
    sparse_swa._compute_prefill_metadata_kernel = make_triton_style_kernel(
        _compute_prefill_metadata_evas
    )
    indexer._build_prefill_chunk_metadata_kernel = make_triton_style_kernel(
        _build_prefill_chunk_metadata_evas
    )
    sparse_swa.get_mla_metadata = get_mla_metadata
    flashmla.get_mla_metadata = get_mla_metadata


def apply_lpu_dsv4_swa_metadata_patch() -> None:
    """Apply the metadata-construction shims. Idempotent, LPU-only."""
    global _PATCHED
    if _PATCHED:
        return
    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return
    _install_metadata_shims()
    _maybe_skip_swa_warmup()
    _PATCHED = True


def _maybe_skip_swa_warmup() -> None:
    """Warmup-mode control for the upstream DSv4 sparse-MLA warmup
    (``deepseek_v4_sparse_mla_attention_warmup``), via
    ``VLLM_FL_SWA_WARMUP_MODE`` (legacy ``VLLM_FL_SKIP_SWA_WARMUP=1`` maps to
    ``skip``):

    - ``upstream`` (default): no redirect — the upstream warmup runs as-is.
      Under the nonblocking_copy patch + verified copies + the flagtree-evas
      compile/launch lock the mixed-batch forward is the better default: it is
      the only warmup that exercises the prefill attention path, and with it
      skipped every attention-shape kernel was left to compile during the
      first REAL step, which is the wedge window.
    - ``skip``: neutralize the warmup (the known-good workaround).
    - ``prefill_only``: run a PURE PREFILL dummy — ``_dummy_run(n, is_profile=True,
      force_attention=True)`` with no ``create_mixed_batch`` and no
      ``uniform_decode``.  With ``num_tokens=16`` and the DSV4 test script's
      ``max_num_seqs=2`` this is 2 requests x 8 tokens: every request is a
      multi-token prefill, i.e. exactly the shape class the real first step
      uses, with no decode leg in the batch.
      ⚠️ Do NOT use ``uniform_decode=True`` here: uniform decode requires each
      request to hold exactly ``uniform_decode_query_len`` (=1 with spec-decode
      off) tokens, so ``num_reqs = min(max_num_seqs, num_tokens)`` makes the
      scheduled-token sum short of ``num_tokens`` and ``_dummy_run`` trips its
      own ``assert sum(num_scheduled_tokens_list) == num_tokens`` (surfaces as
      a bare "Worker failed with error ''" with an empty message).
    - ``mixed``: the upstream mixed-batch dummy (1 prefill + decode requests)
      without the FlashInfer autotune wrapper.

    NOTE the binding trap: ``kernel_warmup`` does ``from ...flashinfer_sparse_mla_warmup
    import deepseek_v4_sparse_mla_attention_warmup`` at module top, so the
    redirect must target the bound name in ``kernel_warmup``'s namespace.
    """
    legacy_skip = os.environ.get("VLLM_FL_SKIP_SWA_WARMUP") == "1"
    mode = os.environ.get("VLLM_FL_SWA_WARMUP_MODE")
    if mode is None:
        mode = "skip" if legacy_skip else "upstream"
    mode = mode.strip().lower()
    if mode not in ("skip", "upstream", "prefill_only", "mixed"):
        logger.warning(
            "[swa-evas] VLLM_FL_SWA_WARMUP_MODE=%r invalid "
            "(skip|upstream|prefill_only|mixed); falling back to 'upstream'",
            mode,
        )
        mode = "upstream"
    if mode == "upstream":
        return  # no redirect — upstream warmup runs unchanged

    import vllm.model_executor.warmup.kernel_warmup as _kw

    if mode == "skip":

        def _replacement_swa_warmup(worker) -> None:
            logger.warning(
                "[swa-evas] VLLM_FL_SWA_WARMUP_MODE=skip: skipping "
                "deepseek_v4_sparse_mla_attention_warmup (mixed-batch warmup)"
            )

    else:  # prefill_only

        def _replacement_swa_warmup(worker) -> None:
            runner = worker.model_runner
            # Same gate as the upstream function (this redirect replaces it
            # wholesale, and kernel_warmup() calls it for every model).
            try:
                from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
                    _has_deepseek_v4_sparse_mla_backend,
                )
            except ImportError:
                logger.warning(
                    "[swa-evas] prefill_only warmup: upstream gate helper "
                    "unavailable; skipping the warmup"
                )
                return
            if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(
                runner
            ):
                return
            max_tokens = worker.scheduler_config.max_num_batched_tokens
            num_tokens = max(0, min(16, max_tokens))
            if num_tokens <= 0:
                return
            max_num_seqs = worker.scheduler_config.max_num_seqs
            # Size ladder: VLLM_FL_WARMUP_TOKENS=2,4,8,16 runs
            # the dummy at each size in ascending order (one run then reveals
            # the pass/wedge boundary).  With max_num_seqs=2 a size N means
            # num_reqs=min(N,2) requests x N/2 tokens each, so the ladder walks
            # decode-like (1 token/req) -> prefill-like (8 tokens/req).
            _sizes_env = os.environ.get("VLLM_FL_WARMUP_TOKENS", "").strip()
            if _sizes_env:
                try:
                    sizes = [int(s) for s in _sizes_env.split(",") if s.strip()]
                except ValueError:
                    logger.warning(
                        "[swa-evas] VLLM_FL_WARMUP_TOKENS=%r not a comma list "
                        "of ints; using default", _sizes_env,
                    )
                    sizes = [num_tokens]
            else:
                sizes = [num_tokens]
            for _n in sizes:
                if _n <= 0:
                    continue
                _per_req = max(1, _n // max(1, min(_n, max_num_seqs)))
                logger.warning(
                    "[swa-evas] warmup ladder step: mode=%s N=%d "
                    "max_num_seqs=%s -> %d token(s)/request %s",
                    mode, _n, max_num_seqs, _per_req,
                    "(decode-like)" if _per_req <= 1 else "(prefill-like)",
                )
                runner._dummy_run(
                    num_tokens=_n,
                    skip_eplb=True,
                    is_profile=True,
                    force_attention=True,
                    create_mixed_batch=(mode == "mixed"),
                )
                logger.warning(
                    "[swa-evas] warmup ladder step DONE: N=%d", _n,
                )
            runner._dummy_run(
                num_tokens=num_tokens,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                create_mixed_batch=True,
            )

    _kw.deepseek_v4_sparse_mla_attention_warmup = _replacement_swa_warmup
    logger.warning(
        "[swa-evas] deepseek_v4_sparse_mla_attention_warmup redirected "
        "(VLLM_FL_SWA_WARMUP_MODE=%s)", mode,
    )

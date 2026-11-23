# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU patch: Kimi Delta Attention (KDA) pure-PyTorch fallbacks.

KimiLinear's KDA layers call three Triton ops from ``fla.ops.kda`` plus
``FusedRMSNormGated.forward_cuda``; none compile on LPU, so the module
attributes are redirected to the pure-PyTorch stand-ins below. Sibling of
``gdn.py``, not merged into it: the op families differ (decay is a per-key-dim
vector here, a per-head scalar in GDN) and only ``causal_conv1d`` is shared.
Prefill runs the sequential recurrence rather than upstream's chunked pipeline.

"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


# ---------------------------------------------------------------------------
# fused_kda_gate
# ---------------------------------------------------------------------------


def fused_kda_gate_pytorch(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Pure-PyTorch drop-in for ``fla.ops.kda.fused_kda_gate``.

    ``g`` is ``[..., H*D]``; ``A`` is ``[H]`` (``A_log``); result is
    ``[..., H, D]`` in **float32** (the Triton kernel allocates ``y`` as f32
    regardless of the input dtype).

    Kernel math (``kda_gate_fwd_kernel``)::

        b_a  = -exp(A[h])
        sp   = softplus(g + g_bias, beta, threshold)   # linear approx for large
        y    = b_a * sp

    ``F.softplus``'s own ``threshold`` has exactly that piecewise form
    (``x`` when ``beta*x > threshold``), so it is used directly.
    """
    orig_shape = g.shape[:-1]
    flat = g.reshape(-1, g.shape[-1]).to(torch.float32)
    t_tokens, hd = flat.shape
    heads = int(A.numel())
    assert heads * head_k_dim == hd, (heads, head_k_dim, hd)

    if g_bias is not None:
        flat = flat + g_bias.reshape(-1).to(torch.float32)
    sp = F.softplus(flat, beta=beta, threshold=threshold)

    a = -torch.exp(A.reshape(-1).to(torch.float32))          # [H]
    # Kernel writes y[t, h*D:(h+1)*D] = a[h] * sp[...], i.e. a is per-head and
    # multiplies that head's contiguous D-slice -- so fold the head axis before
    # broadcasting (a[H] * sp[T, H*D] would be the wrong axes).
    y = sp.reshape(t_tokens, heads, head_k_dim) * a.view(1, heads, 1)
    return y.reshape(*orig_shape, heads, head_k_dim)


# ---------------------------------------------------------------------------
# fused_recurrent_kda (delta-rule recurrence)
# ---------------------------------------------------------------------------


def fused_recurrent_kda_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch drop-in for ``fla.ops.kda.fused_recurrent_kda``.

    Shapes follow the kernel: ``q``/``k`` ``[B, T, H, K]``, ``v`` ``[B, T, HV, V]``,
    ``g`` ``[B, T, HV, K]`` (KDA per-key-dim decay), ``beta`` ``[B, T, HV]`` or
    ``[B, T, HV, V]``.  State is ``[*, HV, V, K]``.

    The Kimi decode path passes ``cu_seqlens`` + ``ssm_state_indices`` and
    ``inplace_final_state=True``, i.e. continuous batching: ``ssm_state_indices``
    is 1D ``[batch]`` with one state slot per sequence, and slot <= 0 means
    "no state" (the kernel skips the store).
    """
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    assert q.shape == (B, T, H, K), (q.shape, (B, T, H, K))
    assert g.shape[-1] == K, (g.shape, K)
    if scale is None:
        scale = K ** -0.5

    # Per-sequence [bos, eos) token ranges.
    if cu_seqlens is not None:
        bounds = [int(x) for x in cu_seqlens.tolist()]
        assert bounds[0] == 0 and bounds[-1] == T, (bounds, T)
        ranges = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    else:
        ranges = [(0, T)] * B

    o = torch.empty_like(k)
    beta_headwise = beta is not None and beta.ndim == v.ndim

    def _state_slot(seq_idx: int) -> int | None:
        # ``non_spec_state_indices_tensor`` is **1D [batch]** (one slot per
        # sequence, see GDNAttentionMetadata), so the slot does not vary with t.
        if ssm_state_indices is None:
            return seq_idx
        if ssm_state_indices.ndim != 1:
            raise ValueError(
                "fused_recurrent_kda: ssm_state_indices must be 1D [batch], got "
                f"{tuple(ssm_state_indices.shape)}"
            )
        idx = int(ssm_state_indices[seq_idx].item())
        return idx if idx > 0 else None

    def _load_state(seq_idx: int) -> torch.Tensor:
        """[HV, V, K] state for a sequence before its first token."""
        slot = _state_slot(seq_idx)
        if slot is None or initial_state is None:
            return torch.zeros(HV, V, K, dtype=torch.float32)
        return initial_state[slot].to(torch.float32)

    for seq_idx, (bos, eos) in enumerate(ranges):
        state = _load_state(seq_idx)
        for t in range(bos, eos):
            qt = q[0, t].to(torch.float32)                 # [H, K]
            kt = k[0, t].to(torch.float32)                 # [H, K]
            vt = v[0, t].to(torch.float32)                 # [HV, V]
            if use_qk_l2norm_in_kernel:
                qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            qt = qt * scale

            # KDA: decay is per-key-dim, broadcast across the V axis.
            if HV == H:
                gk = g[0, t].to(torch.float32)             # [HV, K]
            else:
                gk = g[0, t].to(torch.float32)             # [HV, K]
            state = state * torch.exp(gk).unsqueeze(1)     # [HV, V, K]

            # Delta rule: v' = v - sum_k h[i,k] * k[k]   -> [HV, V]
            vt = vt - (state * kt.unsqueeze(1)).sum(-1)
            if beta is not None:
                bt = beta[0, t].to(torch.float32)
                vt = vt * (bt.unsqueeze(-1) if beta_headwise else bt.unsqueeze(-1))
            state = state + vt.unsqueeze(-1) * kt.unsqueeze(1)   # [HV, V, K]
            o[0, t] = (state * qt.unsqueeze(1)).sum(-1).to(o.dtype)

            if inplace_final_state:
                slot = _state_slot(seq_idx)
                if slot is not None and initial_state is not None:
                    initial_state[slot].copy_(state.to(initial_state.dtype))
            elif initial_state is not None:
                # Non-inplace path writes a per-token state buffer; the KDA
                # callers use inplace, so this branch only exists for parity.
                raise NotImplementedError(
                    "fused_recurrent_kda: inplace_final_state=False not supported"
                )

    return o, initial_state


# ---------------------------------------------------------------------------
# chunk_kda_with_fused_gate (chunked prefill)
# ---------------------------------------------------------------------------


def chunk_kda_with_fused_gate_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure-PyTorch drop-in for ``fla.ops.kda.chunk_kda_with_fused_gate``.

    Upstream runs a 5-stage chunked pipeline
    (``chunk_kda_scaled_dot_kkt`` -> ``solve_tril`` -> ``recompute_w_u`` ->
    ``chunk_gated_delta_rule_fwd_h`` -> ``chunk_gla_fwd_o_gk``).  That is a
    *blocked reorganisation* of the same recurrence used by the decode path --
    it exists to expose chunk-level parallelism to Triton, not because the math
    differs.  Since the LPU path is pure PyTorch anyway, the sequential
    recurrence is used, exactly as ``gdn.py`` does for Qwen GDN
    ("naive per-token loop; fine for short prompts").

    Shapes match the Kimi call site: ``q``/``k``/``v``/``raw_g`` ``[B, T, H, D]``
    with ``B == 1`` and per-sequence ranges given by ``cu_seqlens``.
    ``initial_state`` is ``[N, H, D, D]`` (one slot per sequence) and, when
    ``output_final_state`` is set, the matching ``final_state`` is returned.
    """
    B, T, H, D = k.shape
    assert B == 1, f"only B=1 varlen is supported, got {B}"
    if scale is None:
        scale = D ** -0.5

    # Gate: raw_g is the pre-activation projection; the kernel fuses
    # softplus + -exp(A) + per-chunk cumsum, but the per-token log-decay is
    # exactly fused_kda_gate's output.
    gate_log = fused_kda_gate_pytorch(
        raw_g.reshape(T, H * D), A_log, D, g_bias=g_bias
    ).reshape(1, T, H, D)                                   # [B, T, H, D]

    if cu_seqlens is not None:
        bounds = [int(x) for x in cu_seqlens.tolist()]
        assert bounds[0] == 0 and bounds[-1] == T, (bounds, T)
        ranges = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    else:
        ranges = [(0, T)]

    o = torch.empty_like(k)
    beta_headwise = beta.ndim == v.ndim
    states: list[torch.Tensor] = []
    for seq_idx, (bos, eos) in enumerate(ranges):
        if initial_state is not None:
            state = initial_state[seq_idx].to(torch.float32).clone()
        else:
            state = torch.zeros(H, D, D, dtype=torch.float32)
        for tt in range(bos, eos):
            qt = q[0, tt].to(torch.float32)
            kt = k[0, tt].to(torch.float32)
            vt = v[0, tt].to(torch.float32)
            if use_qk_l2norm_in_kernel:
                qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            qt = qt * scale
            state = state * torch.exp(gate_log[0, tt]).unsqueeze(1)   # [H,D,D]
            vt = vt - (state * kt.unsqueeze(1)).sum(-1)               # [H, D]
            if beta is not None:
                bt = beta[0, tt].to(torch.float32)
                vt = vt * (bt.unsqueeze(-1) if beta_headwise else bt.unsqueeze(-1))
            state = state + vt.unsqueeze(-1) * kt.unsqueeze(1)
            o[0, tt] = (state * qt.unsqueeze(1)).sum(-1).to(o.dtype)
        states.append(state)

    final_state = None
    if output_final_state:
        final_state = torch.stack(states, dim=0)            # [N, H, D, D]
    return o, final_state


# ---------------------------------------------------------------------------
# rms_norm_gated (FusedRMSNormGated's forward_cuda body)
# ---------------------------------------------------------------------------


def rms_norm_gated_pytorch(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    """Pure-PyTorch drop-in for ``fla.ops.kda.rms_norm_gated``.

    ``FusedRMSNormGated`` is a ``CustomOp``, but on LPU the dispatch table has
    no evas branch and falls through to ``forward_cuda`` -- the Triton body --
    rather than the already-correct ``forward_native``.  This rebind covers
    that path so the op works regardless of which forward is selected.

    Kernel math (``layer_norm_gated_fwd_kernel`` with ``IS_RMS_NORM=True``)::

        x  = x + residual                      # if residual
        rstd = 1/sqrt(mean(x^2) + eps)         # no mean subtraction (RMS)
        y  = (x * rstd) * w + b                # if weight / bias
        y  = y * g * sigmoid(g)                # swish / silu
        y  = y * sigmoid(g)                    # sigmoid

    Returns ``y``, or ``(y, residual_out)`` when ``prenorm``.
    """
    x_shape = x.shape
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    gf = g.reshape(-1, g.shape[-1]).to(torch.float32)
    if residual is not None:
        xf = xf + residual.reshape(-1, residual.shape[-1]).to(torch.float32)
    residual_out = xf.to(
        residual.dtype if residual is not None
        else (torch.float if residual_in_fp32 else x.dtype)
    ) if prenorm else None

    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd
    if weight is not None:
        y = y * weight.reshape(-1).to(torch.float32)
    if bias is not None:
        y = y + bias.reshape(-1).to(torch.float32)

    if activation in ("swish", "silu"):
        y = y * gf * torch.sigmoid(gf)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gf)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    y = y.to(x.dtype).reshape(x_shape)
    return y if not prenorm else (y, residual_out.reshape(x_shape))


# ---------------------------------------------------------------------------
# _LpuStateRows -- safe row access on the mamba recurrent-state cache
# ---------------------------------------------------------------------------


class _LpuStateRows:
    """Row get/set stand-in for the mamba recurrent-state cache on LPU.

    ``recurrent_state`` is a **strided view into a packed mamba cache pool**
    (one slot packs ssm_state + conv_state + padding).  On LPU an advanced-index
    gather or setitem on that view materializes the *entire pool* -- in the
    tiny-random kimi-linear-kda run the pool is 12.49 GiB, and
    ``recurrent_state[zero_idx] = 0`` died with "Tried to allocate 12.49 GiB".
    That is the same failure ``gdn.py``'s ``_gdn_gather_rows`` /
    ``_gdn_scatter_rows`` exist for, and those helpers are layout-agnostic over
    any "rows of ``inner`` elems" cache, so they are reused here.

    ``KimiGatedDeltaNetAttention._forward`` touches the cache at exactly four
    sites (verified by grep; there is no other use):

        recurrent_state[zero_idx] = 0                          # prefill
        initial_state = recurrent_state[idx].contiguous()      # prefill
        recurrent_state[idx] = last_recurrent_state            # prefill
        fused_recurrent_kda(initial_state=recurrent_state, ...)  # decode

    Swapping this object in for ``recurrent_state`` at those sites means
    ``_forward`` itself needs no fork -- which is the point, since forking it
    would copy ~175 lines of upstream.

    Supported protocol (the union of what the four sites need):
      * ``int`` index        -> a **view** of that row, so ``.copy_()`` writes
                                through (the decode path's in-place state store)
      * 1-D int tensor index -> gathered rows, or a scatter of ``src``
      * ``src`` being a python number 0 -> those rows are zeroed
      * ``.dtype`` / ``.shape`` passthrough
    """

    __slots__ = ("_t",)

    def __init__(self, t: torch.Tensor) -> None:
        self._t = t

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def shape(self):
        return self._t.shape

    @property
    def _inner(self) -> int:
        """Row width in elems.  Product of the trailing dims, so it stays valid
        for an empty cache (``self._t[0]`` would raise)."""
        n = 1
        for d in self._t.shape[1:]:
            n *= int(d)
        return n

    def _view_row(self, s: int) -> torch.Tensor:
        from .gdn import _gdn_cache_flat, _gdn_row_addr

        # ``_gdn_row_addr`` is pure arithmetic on stride(0) and would happily
        # address outside the cache; a real tensor would raise IndexError.
        n = self._t.shape[0]
        if not 0 <= s < n:
            raise IndexError(f"_LpuStateRows: row {s} out of range for {n} slots")
        a = _gdn_row_addr(self._t, s)
        return _gdn_cache_flat(self._t)[a : a + self._inner].view(*self._t.shape[1:])

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._view_row(index)
        from .gdn import _gdn_gather_rows

        idx = torch.as_tensor(index, dtype=torch.long).reshape(-1)
        return _gdn_gather_rows(self._t, idx)

    def __setitem__(self, index, value) -> None:
        from .gdn import _gdn_scatter_rows

        idx = torch.as_tensor(index, dtype=torch.long).reshape(-1)
        if isinstance(value, (int, float)):
            if value != 0:
                raise ValueError(
                    f"_LpuStateRows: only 0 is supported as a scalar fill, got {value}"
                )
            src = torch.zeros(
                (1, self._inner), dtype=self._t.dtype, device=self._t.device
            )
        else:
            src = value
            # ``_gdn_scatter_rows`` broadcasts a single row across ``idx``, so a
            # silent length mismatch would write the wrong rows.  A real
            # ``t[idx] = src`` requires 1 or len(idx) rows.
            n_rows = int(src.shape[0]) if src.ndim >= 1 else 1
            if n_rows not in (1, idx.shape[0]):
                raise ValueError(
                    f"_LpuStateRows: src has {n_rows} rows for {idx.shape[0]} indices"
                )
        _gdn_scatter_rows(self._t, idx, src)


# ---------------------------------------------------------------------------
# patch install
# ---------------------------------------------------------------------------


def apply_lpu_kda_patch() -> None:
    """Redirect the KDA ops to pure-PyTorch on LPU.  Idempotent.

    Three things, all LPU-only:

    1. ``fla.ops.kda``'s four Triton entry points -> pure-PyTorch.
       ``rms_norm_gated`` is rebound too: ``FusedRMSNormGated`` is a
       ``CustomOp`` whose ``forward_cuda`` is the Triton body, and LPU has no
       evas branch in ``dispatch_forward`` so it falls through to
       ``forward_cuda`` rather than the already-correct ``forward_native``.

    2. ``causal_conv1d_fn`` / ``causal_conv1d_update`` -- the KDA layer's conv
       dependency.  Their implementation lives in ``gdn.py`` and the *source*
       module (``mamba.ops.causal_conv1d``) is rebound there; but
       ``from ..ops.causal_conv1d import causal_conv1d_fn`` freezes the binding
       at import time, so the already-imported local names are redirected here.

    3. ``KimiGatedDeltaNetAttention._forward`` wrapped so the recurrent-state
       cache is a :class:`_LpuStateRows` instead of the raw strided pool view
       (see that class for why).
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        import vllm.model_executor.layers.fla.ops.kda as _kda
    except ImportError:
        # Upstream rename would otherwise retry + stay silent forever.
        logger.warning(
            "LPU KDA patch partial: cannot import "
            "vllm.model_executor.layers.fla.ops.kda (upstream rename?); "
            "ALL KDA patches SKIPPED"
        )
        if os.environ.get("VLLM_FL_STRICT_UPSTREAM_REBIND") == "1":
            raise
        _PATCHED = True
        return

    if getattr(_kda.fused_kda_gate, "__fl_kda_pytorch__", False):
        _PATCHED = True
        return

    _kda.fused_kda_gate = fused_kda_gate_pytorch
    _kda.fused_kda_gate.__fl_kda_pytorch__ = True  # type: ignore[attr-defined]
    _kda.fused_recurrent_kda = fused_recurrent_kda_pytorch
    _kda.fused_recurrent_kda.__fl_kda_pytorch__ = True  # type: ignore[attr-defined]
    _kda.chunk_kda_with_fused_gate = chunk_kda_with_fused_gate_pytorch
    _kda.chunk_kda_with_fused_gate.__fl_kda_pytorch__ = True  # type: ignore[attr-defined]
    _kda.rms_norm_gated = rms_norm_gated_pytorch
    _kda.rms_norm_gated.__fl_kda_pytorch__ = True  # type: ignore[attr-defined]

    # (2) conv dependency: redirect the importer's frozen bindings to the
    # pure-PyTorch conv fallbacks from gdn.py (see apply_lpu_gdn_patch for the
    # source-module side).
    from . import gdn as _gdn_mod

    try:
        import vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn as _kgla
    except ImportError:
        _kgla = None
    if _kgla is not None:
        # Read the hooks, not a fixed implementation: _set_conv_impls binds
        # them to the FlagGems public names, whose pin decides kernel or twin.
        _kgla.causal_conv1d_fn = _gdn_mod._CONV_FN
        _kgla.causal_conv1d_update = _gdn_mod._CONV_UPDATE

        # (3) safe row access on the recurrent-state cache.
        _orig_forward = _kgla.KimiGatedDeltaNetAttention._forward

        def _kda_forward_lpu(self, *args, **kwargs):
            # ``kv_cache`` is bound by the model runner only after profile_run,
            # and ``_forward`` early-returns when there is no attn metadata.
            # Swap only when the caches are actually there -- otherwise this
            # would raise AttributeError during memory profiling.
            cache = getattr(self, "kv_cache", None)
            if cache is None or len(cache) != 2:
                return _orig_forward(self, *args, **kwargs)
            conv_state, recurrent_state = cache
            self.kv_cache = (conv_state, _LpuStateRows(recurrent_state))
            try:
                return _orig_forward(self, *args, **kwargs)
            finally:
                self.kv_cache = (conv_state, recurrent_state)

        _kgla.KimiGatedDeltaNetAttention._forward = _kda_forward_lpu
        _kda_forward_lpu.__fl_kda_pytorch__ = True  # type: ignore[attr-defined]

    _PATCHED = True
    if _kgla is None:
        # Step (1) (the KDA op fallbacks) is already live; only the conv
        # redirect (2) and the recurrent-state row access (3) were skipped.
        # The old unconditional info log claimed ALL of it installed -- a
        # partial patch reported as complete (a skipped _LpuStateRows later
        # costs the 12.49 GiB whole-pool materialization, see _LpuStateRows
        # docstring).
        logger.warning(
            "LPU KDA patch partial: cannot import kimi_gdn_linear_attn "
            "(upstream rename?); KDA ops installed, conv redirects + "
            "recurrent-state row access SKIPPED"
        )
        if os.environ.get("VLLM_FL_STRICT_UPSTREAM_REBIND") == "1":
            raise RuntimeError(
                "LPU KDA patch partial under VLLM_FL_STRICT_UPSTREAM_REBIND=1"
            )
    else:
        logger.info(
            "Installed LPU KDA fallbacks (fused_kda_gate, fused_recurrent_kda, "
            "chunk_kda_with_fused_gate, rms_norm_gated) + KDA conv redirects + "
            "recurrent-state row access"
        )

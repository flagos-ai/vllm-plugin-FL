# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU rewrite of the Qwen3.5 GatedDeltaNet core.

GDN does not run on LPU as shipped: its ops are Triton kernels, and
``fused_post_conv_prep`` fails at triton frontend compile (mismatched
``mask_2d``). Five functions in vLLM's gdn import namespace are replaced; the
operator bodies live in the FlagGems evas backend, reached through
``flag_gems`` public names, so the adoption pin decides per op whether the
Triton kernel or its pure twin runs. This module carries no operator copy: a
missing backend raises a named error, not a silent second implementation.

"""
from __future__ import annotations

import atexit
import os
import weakref

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_SOFTPLUS_THRESHOLD = 20.0
_L2NORM_EPS = 1e-6

# Diagnostic only (VLLM_FL_GDN_OP_BASE_DEBUG=1). An operand that is a *view* of
# a large allocation is billed by that allocation, not by its own bytes: probing
# a 16 KiB view with a 1 / 8 / 64 / 256 MiB backing costs 25 / 80 / 510 / 1994 us
# while the same tensor with its own storage costs 0.2 us. `_base` only sees
# Python views, so the descriptor parent is read out through `unzip`, which sizes
# its shards from the descriptor (see local/double_die.py:71-74).
#
# The print is a general operand probe, not an attribution for any particular
# call site: reading these numbers is what showed the GDN 700-900 us clone band
# is the broadcast multiply itself (a [1,H,V,K] * [1,H,1,K] elementwise op) and
# not a share of a large backing, so keep reporting and keep drawing the
# conclusion from the numbers rather than from the band size.
_OP_BASE_DEBUG = os.environ.get("VLLM_FL_GDN_OP_BASE_DEBUG") == "1"
_OP_BASE_SEEN: set = set()
_OP_BASE_MAX: int = 32


def _dbg_op_base(tag: str, t: torch.Tensor) -> None:
    if not _OP_BASE_DEBUG or len(_OP_BASE_SEEN) >= _OP_BASE_MAX:
        return
    base = getattr(t, "_base", None)
    held = (base if base is not None else t).numel() * t.element_size()
    try:
        sd = t.split_dim()
    except Exception:                             # noqa: BLE001 - diagnostic
        sd = "?"
    key = (tag, tuple(t.shape), held, sd)
    if key in _OP_BASE_SEEN:
        return
    _OP_BASE_SEEN.add(key)
    parent = ""
    if isinstance(sd, int) and 0 <= sd < t.dim() and t.shape[sd] > 1:
        try:
            # shaped from the descriptor, so their extent is the parent's
            shards = torch.ops.lpu.unzip_double_die_tensor(t)
            parent = (f" parent_dim{sd}={2 * shards[0].shape[sd]}"
                      f" ({2 * shards[0].numel() * t.element_size()} B)")
            del shards
        except Exception as exc:                  # noqa: BLE001 - diagnostic
            parent = f" unzip=<{type(exc).__name__}>"
    print(f"[gdn-opbase] {tag} shape={tuple(t.shape)} stride={tuple(t.stride())} "
          f"dtype={t.dtype} payload={t.numel() * t.element_size()} "
          f"storage={held} split_dim={sd} contig={t.is_contiguous()}{parent}",
          flush=True)


if _OP_BASE_DEBUG:
    # The clone that costs 810 us is reached through a plain Python
    # `tensor.contiguous()`, so wrapping the method is what covers the call
    # sites the two tagged operands below do not. Wrapped only under the debug
    # flag, and the dedup in _dbg_op_base keeps it to a handful of prints.
    _ORIG_CONTIGUOUS = torch.Tensor.contiguous

    def _dbg_contiguous(self, *args, **kwargs):
        if len(_OP_BASE_SEEN) < _OP_BASE_MAX:
            _dbg_op_base("contiguous", self)
        return _ORIG_CONTIGUOUS(self, *args, **kwargs)

    torch.Tensor.contiguous = _dbg_contiguous



def _softplus(x: torch.Tensor) -> torch.Tensor:
    # numerically stable softplus, threshold=20 (matches Triton kernel)
    return torch.where(x > _SOFTPLUS_THRESHOLD, x, torch.log1p(torch.exp(x)))


def _gdn_row_inner(t: torch.Tensor) -> int:
    """Element count of one cache row, computed from the shape alone.

    ``t[0].numel()`` reads like free metadata, but ``t[0]`` is an
    ``aten::select``: on LPU a select off a die-affinity tensor (every GDN
    cache is a view into the packed-page pool) materializes a clone kernel on
    the device.  The helpers below are called once per layer per step, each of
    them asking for this number, so that select was a pure device cost for a
    value the shape already gives: the call sites that paid it are gdn.py
    133/144/173/190, worth 1.14s of device time in the Qwen3.6-27B stack-window
    profile.
    """
    return t.shape[1:].numel()


def _gdn_cache_flat(t: torch.Tensor) -> torch.Tensor:
    """1D view of a padded-page strided cache tensor's own data region.

    GDN caches are views into a packed-page KV pool where each slot packs
    ssm_state + conv_state + padding, so ``t`` is non-contiguous with
    stride(0) > inner numel and (typically) a non-zero storage_offset. On LPU,
    any advanced-index gather or setitem on this strided source materializes
    the FULL tensor (~10 GiB) and OOMs (and torch.index_copy_ is broken on LPU
    regardless). We therefore access slots as contiguous inner-elems runs at
    ``slot*stride(0)`` within this 1D region view. ``_gdn_gather_rows`` /
    ``_gdn_scatter_rows`` below copy only the selected rows.

    A contiguous cache (mamba page size == attention page size, so no page
    padding) is returned as-is: each row is the plain ``s*inner`` run and the
    two helpers handle both layouts identically.
    """
    if t.is_contiguous():
        return t.reshape(-1)
    n = t.shape[0]
    inner = _gdn_row_inner(t)
    stride0 = t.stride(0)
    region = (n - 1) * stride0 + inner
    return t.as_strided((region,), (1,), storage_offset=t.storage_offset())


def _gdn_row_addr(t: torch.Tensor, s: int) -> int:
    """Flat offset of row ``s``. Padded-page cache: ``s*stride(0)`` (one slot
    spans a full page including conv_state/padding). Contiguous cache: ``s``
    rows of ``inner`` elems — which equals ``s*stride(0)`` since stride(0)==inner,
    so the same formula covers both."""
    inner = _gdn_row_inner(t)
    return s * t.stride(0) if not t.is_contiguous() else s * inner


def _check_row_slots(t: torch.Tensor, idx: torch.Tensor) -> list[int]:
    """Fail loud on an out-of-range cache row index, and hand back the values.

    ``_gdn_row_addr`` is pure arithmetic on ``stride(0)``, so a bad index is not
    an error -- it reads/writes outside the cache and silently corrupts whatever
    shares the packed page (conv_state, padding, a neighbouring slot).  That is
    exactly the shape of a multi-request contamination bug, so the bound is
    checked rather than trusted.

    The validated python list is returned because both callers need exactly
    these values to walk the cache: reading ``idx.cpu().tolist()`` again in the
    caller costs a second device sync for data already on the host.
    """
    n = t.shape[0]
    vals = [int(v) for v in idx.cpu().tolist()]
    bad = [v for v in vals if not 0 <= v < n]
    if bad:
        raise IndexError(
            f"cache row index out of range: {bad} not in [0, {n}) "
            f"for cache of shape {tuple(t.shape)}"
        )
    return vals


def _gdn_gather_rows(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows ``idx`` of a padded-page strided cache ``t`` without copying
    the full source. Returns ``t[idx]`` as (n, *t.shape[1:]) — identical
    semantics to the advanced-index gather (verified on LPU, see survey
    smoke_gdn_strided_cache_gather)."""
    vals = _check_row_slots(t, idx)
    inner = _gdn_row_inner(t)
    flat = _gdn_cache_flat(t)
    n = len(vals)
    out = torch.empty((n, inner), dtype=t.dtype, device=t.device)
    for i, s in enumerate(vals):
        a = _gdn_row_addr(t, s)
        out[i].copy_(flat[a : a + inner])
    return out.reshape(n, *t.shape[1:]).contiguous()


def _gdn_scatter_rows(t: torch.Tensor, idx: torch.Tensor, src: torch.Tensor) -> None:
    """Scatter rows of ``src`` into rows ``idx`` of a padded-page strided cache
    ``t``, touching only t's region (preserving the conv_state/padding regions
    sharing the packed page). Mirrors ``t[idx] = src`` (including broadcasting a
    single-row src to all indexed slots) but without LPU full-source
    materialization."""
    vals = _check_row_slots(t, idx)
    inner = _gdn_row_inner(t)
    flat = _gdn_cache_flat(t)
    src2d = src.reshape(-1, inner)
    for i, s in enumerate(vals):
        a = _gdn_row_addr(t, s)
        row = src2d[0] if src2d.shape[0] == 1 else src2d[i]
        flat[a : a + inner].copy_(row)


def _contig_meta(t):
    """A contiguous copy of a conv metadata vector, or ``t`` when it already is.

    On LPU a view over a die-affinity buffer reports ``is_contiguous() == False``
    even when its strides are the row-major ones (shape (2,) stride (1,) is one
    such case), and ``Tensor.contiguous()`` returns the same object for it.  The
    pinned kernel preflights these vectors with ``is_contiguous()``, so it
    rejects the views vLLM hands over; a real copy is the only thing that
    satisfies the check.  The vectors are a few int32 per call.
    """
    if t is None or t.is_contiguous():
        return t
    return t.clone()


# Default hooks: the local twins, replaced by ``_set_conv_impls`` when the
# patch resolves the public names.
_CONV_FN = None
_CONV_UPDATE = None
_POST_CONV_PREP = None


_CHUNK_FWD = None


def _gdn_warmup_noop(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
    """No-op warmup: the routed ops have no Triton autotuner to prime.
    Mirrors the 0.24 signature (qkv_or_qkvz, v_dim); the base warmup runs the
    real chunk_gated_delta_rule (triton/CuteDSL) on dummy tensors which crashes
    on LPU, so it is replaced with a bare flag set."""
    self._prefill_kernels_warmed_up = True


def _gdn_merge_core_out(dst: torch.Tensor, src: torch.Tensor,
                        num_actual_tokens: int, what: str) -> None:
    """Assign ``src`` into ``dst[:num_actual_tokens]``, flattening the batch.

    ``*_out`` tensors are ``[B, T, HV, V]`` with a *dummy* batch of 1 in the
    ``cu_seqlens`` form and a real batch otherwise.  ``flatten(0, 1)``
    collapses that leading pair onto the token axis -- identical to upstream's
    ``squeeze(0)`` when ``B == 1``, and correct when ``B > 1``, where a squeeze
    is a no-op and would leave a 4-D tensor to assign into the 3-D output.

    The flattened result must cover **exactly** the scheduled token range
    ``[0, num_actual_tokens)``.  Every step shape satisfies that by
    construction -- decode-only ``T == num_decodes``, prefill-only
    ``T == num_actual_tokens``, mixed ``num_decode_tokens + T_prefill``,
    spec-merged via ``index_copy_`` -- but nothing checked it, so a later change
    to chunked prefill, batch padding or spec-decode acceptance would surface
    only as a bare assignment/reshape error naming no contract.
    """
    flat = src.flatten(0, 1)
    got = int(flat.shape[0])
    if got != int(num_actual_tokens):
        raise RuntimeError(
            f"GDN core merge ({what}): flattened output covers {got} tokens "
            f"but num_actual_tokens is {num_actual_tokens} (src "
            f"{tuple(src.shape)}) -- the [B, T] layout no longer matches the "
            "scheduled token range"
        )
    dst[:num_actual_tokens] = flat


def _gdn_forward_core_decode_non_spec_lpu(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    attn_metadata,
):
    """Route through the main _forward_core path instead of the packed_decode
    fast path (identical results through the same recurrence hook).

    Rationale: replicating fused_recurrent_gated_delta_rule_packed_decode (a
    fused Triton kernel on mixed_qkv 2D tensors with per-row state lookups) is
    complex and risks numerical misalignment; the main decode branch produces
    identical results without fusing conv1d → post-conv-prep → recurrence.
    Since the dispatch to this fast path is removed in _gdn_forward_core_lpu,
    this is only a safety net for external callers.
    """
    return _gdn_forward_core_lpu(self, mixed_qkv, b, a, core_attn_out)


def _gdn_forward_core_lpu(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
):
    """LPU port of QwenGatedDeltaNetAttention._forward_core
    (vllm 0.24 qwen_gdn_linear_attn.py:1260-1576) with 5 Triton-op substitutions:
      1. causal_conv1d_fn          → the advertised hook
      2. causal_conv1d_update      → the advertised hook
      3. fused_post_conv_prep      → the advertised hook
      4. chunk_gated_delta_rule    → the advertised hook (bypasses upstream's
                                    ``chunk_gated_delta_rule`` CustomOp, whose
                                    cuda/cutedsl dispatch has no LPU impl)
      5. decode recurrence         → fused_recurrent_gated_delta_rule_fwd via
                                    ``_gdn_recurrent_update``
    The hooks resolve to whichever arm the adoption pin selects; there is no
    local operator copy in this module any more.

    This targets ``QwenGatedDeltaNetAttention`` (qwen3.5/3.6/next-moe).  Its
    ``_forward_core`` carries the ``split_non_spec`` decode+prefill branch and a
    prefill chunk path driven by ``prefill_state_indices`` /
    ``prefill_has_initial_state`` / ``prefill_query_start_loc`` (gathered/scattered
    via the LPU-safe per-row helpers instead of advanced-index on the strided
    cache).

    Deliberate omission: the ``enable_packed_recurrent_decode`` dispatch
    (``_forward_core:1286-1298``) is NOT ported — decode always uses the main
    decode branch (the same recurrence hook).
    """
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        # V1 profile run — warm up prefill kernels so that
        # autotuning completes before KV cache allocation.
        # Routed ops: no-op (no Triton autotuner).
        self._warmup_prefill_kernels(mixed_qkv, 0)
        return

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]

    # NOTE: deliberately removed the enable_packed_recurrent_decode fast-path
    # dispatch.  The main decode branch below handles all decode cases via
    # the recurrence hook.

    has_initial_state = attn_metadata.has_initial_state
    spec_query_start_loc = attn_metadata.spec_query_start_loc
    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    spec_token_indx = attn_metadata.spec_token_indx
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
    self_kv_cache = self.kv_cache
    # conv_state must be (..., dim, width-1) for the conv kernels.
    conv_state = (
        self_kv_cache[0]
        if is_conv_state_dim_first()
        else self_kv_cache[0].transpose(-1, -2)
    )
    ssm_state = self_kv_cache[1]
    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    # 1. Convolution sequence transformation
    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    if spec_sequence_masks is not None:
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            mixed_qkv_spec = mixed_qkv
            mixed_qkv_non_spec = None
        else:
            mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
    else:
        mixed_qkv_spec = None
        mixed_qkv_non_spec = mixed_qkv

    # 1.1: Process the multi-query part  (spec decode)
    if spec_sequence_masks is not None:
        assert spec_state_indices_tensor is not None
        mixed_qkv_spec = _CONV_UPDATE(
            mixed_qkv_spec,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=spec_state_indices_tensor[:, 0][
                : attn_metadata.num_spec_decodes
            ],
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=spec_query_start_loc,
            max_query_len=spec_state_indices_tensor.size(-1),
            validate_data=False,
        )

    # 1.2: Process the remaining part
    if attn_metadata.num_prefills > 0:
        assert mixed_qkv_non_spec is not None
        mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
        mixed_qkv_non_spec = _CONV_FN(
            mixed_qkv_non_spec_T,
            conv_weights,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=_contig_meta(non_spec_state_indices_tensor),
            query_start_loc=_contig_meta(non_spec_query_start_loc),
            metadata=attn_metadata,
        ).transpose(0, 1)
    elif attn_metadata.num_decodes > 0:
        assert mixed_qkv_non_spec is not None
        mixed_qkv_non_spec = _CONV_UPDATE(
            mixed_qkv_non_spec,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=_contig_meta(
                non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens]
            ),
            validate_data=True,
        )
    else:
        mixed_qkv_non_spec = None

    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

    # Split mixed non-spec-decode+prefill to process independently
    split_non_spec = (
        spec_sequence_masks is None
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decodes > 0
    )
    num_decode_tokens = attn_metadata.num_decode_tokens

    if attn_metadata.num_prefills > 0:
        assert mixed_qkv_non_spec is not None, (
            "mixed_qkv_non_spec must be provided for prefill path"
        )
        if spec_sequence_masks is not None:
            a_non_spec = a.index_select(0, non_spec_token_indx)
            b_non_spec = b.index_select(0, non_spec_token_indx)
        else:
            a_non_spec = a
            b_non_spec = b

        if split_non_spec:
            conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
            a_prefill = a_non_spec[num_decode_tokens:]
            b_prefill = b_non_spec[num_decode_tokens:]
        else:
            conv_output_prefill = mixed_qkv_non_spec
            a_prefill = a_non_spec
            b_prefill = b_non_spec

        (
            query_non_spec,
            key_non_spec,
            value_non_spec,
            g_non_spec,
            beta_non_spec,
        ) = _POST_CONV_PREP(
            conv_output=conv_output_prefill,
            a=a_prefill,
            b=b_prefill,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        query_non_spec = query_non_spec.unsqueeze(0)
        key_non_spec = key_non_spec.unsqueeze(0)
        value_non_spec = value_non_spec.unsqueeze(0)
        g_non_spec = g_non_spec.unsqueeze(0)
        beta_non_spec = beta_non_spec.unsqueeze(0)
    else:
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )
        g_non_spec = None
        beta_non_spec = None

    # 2. Recurrent attention

    # 2.1: Process the multi-query part  (spec decode)
    if spec_sequence_masks is not None:
        core_attn_out_spec, last_recurrent_state = (
            _gdn_recurrent_update(
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                    cu_seqlens=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                )
        )
    else:
        core_attn_out_spec, last_recurrent_state = None, None

    # 2.2: Process non-spec-decode part (when a decode+prefill batch is split)
    if split_non_spec:
        query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec[:num_decode_tokens]
        )
        core_attn_out_decode, _ = _gdn_recurrent_update(
            A_log=self.A_log,
            a=a[:num_decode_tokens],
            b=b[:num_decode_tokens],
            dt_bias=self.dt_bias,
            q=query_decode,
            k=key_decode,
            v=value_decode,
            initial_state=ssm_state,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
            ssm_state_indices=non_spec_state_indices_tensor,
        )
    else:
        core_attn_out_decode = None

    # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
    if attn_metadata.num_prefills > 0:
        # State indices, initial-state mask and cu_seqlens for the chunk
        # kernel are precomputed by the metadata builder.
        prefill_state_indices = attn_metadata.prefill_state_indices
        prefill_has_initial_state = attn_metadata.prefill_has_initial_state
        assert prefill_state_indices is not None
        assert prefill_has_initial_state is not None
        # LPU: advanced-index gather ssm_state[idx] on the strided packed-page
        # cache materializes the full ~10 GiB tensor and OOMs; use the per-row
        # helper instead (see _gdn_gather_rows).
        initial_state = _gdn_gather_rows(ssm_state, prefill_state_indices)
        # LPU: the boolean-mask setitem this replaces (`initial_state[~mask] = 0`)
        # is not just slow, it is WRONG for an empty mask. When every prefill
        # request already has context -- chunk 2+ of a prompt longer than
        # max_num_batched_tokens -- the mask selects nothing, CPU torch treats
        # the statement as a no-op, and the device zeroes row 0 anyway, so the
        # state carried over from the previous chunk is destroyed for that
        # request. masked_fill_ matches CPU semantics for empty, full and mixed
        # masks (verified on LPU for n=1..4, fp32/bf16).
        # Cost, separately: the setitem form is a FLAT ~22.9 ms per call
        # whatever the size (3 / 6 / 12 MiB -> 22.9 / 23.0 / 24.0 ms, and the
        # same with an empty mask), where masked_fill_ takes ~0.07 ms. Once per
        # GDN layer per prefill chunk, that is 48 x 22.9 ms ~ 1.1 s per prefill
        # forward -- 20.4% of device time in the 27B TP1 profile, and a measured
        # 1.106 s (5.25%) of the 61-token prefill carrier.
        initial_state.masked_fill_(
            (~prefill_has_initial_state).view(
                -1, *([1] * (initial_state.dim() - 1))
            ),
            0,
        )
        # The chunk entry takes the state as [N, H, K, V] -- the upstream
        # layout, and the one the kernel reads -- while the cache pool is
        # [slots, HV, V, K].  Convert on both sides of the call, exactly as the
        # decode path does around fused_recurrent_gated_delta_rule_fwd.
        chunk_out = _CHUNK_FWD(
            q=query_non_spec,
            k=key_non_spec,
            v=value_non_spec,
            g=g_non_spec,
            beta=beta_non_spec,
            initial_state=initial_state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            cu_seqlens=attn_metadata.prefill_query_start_loc,
            # The twin defaults this to q.shape[-1] ** -0.5; the kernel has no
            # default, so pass the same value explicitly and both arms agree.
            scale=query_non_spec.shape[-1] ** -0.5,
        )
        # Upstream return is (g_cumsum, o, A, final_state, w, h, v_new); only the
        # output and the final state are consumed here.
        core_attn_out_non_spec, last_recurrent_state = chunk_out[1], chunk_out[3]
        # Init cache (scatter into strided cache without LPU full-source copy)
        _gdn_scatter_rows(
            ssm_state,
            prefill_state_indices,
            last_recurrent_state.transpose(-1, -2).to(ssm_state.dtype),
        )

        if split_non_spec:
            # Stitch the peeled decode outputs in front of the prefill
            # outputs (decode-first order).
            core_attn_out_non_spec = torch.cat(
                [core_attn_out_decode, core_attn_out_non_spec], dim=1
            )
    elif attn_metadata.num_decodes > 0:
        core_attn_out_non_spec, last_recurrent_state = (
            _gdn_recurrent_update(
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                initial_state=ssm_state,
                    cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                )
        )
    else:
        core_attn_out_non_spec, last_recurrent_state = None, None

    # 3. Merge core attention output (see _gdn_merge_core_out for the contract)
    if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
        merged_out = torch.empty(
            (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
            dtype=core_attn_out_non_spec.dtype,
            device=core_attn_out_non_spec.device,
        )
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        _gdn_merge_core_out(core_attn_out, merged_out, num_actual_tokens,
                            "spec+non-spec merged")
    elif spec_sequence_masks is not None:
        _gdn_merge_core_out(core_attn_out, core_attn_out_spec,
                            num_actual_tokens, "spec only")
    else:
        _gdn_merge_core_out(core_attn_out, core_attn_out_non_spec,
                            num_actual_tokens, "non-spec")


# Resolved by ``_gdn_conv_impls()`` when the patch is applied.  The Qwen core
# below calls through these names, so production follows the adoption pin
# (kernel once it graduates, FlagGems twin before that) instead of the local
# copies.  Module-level rather than per-call because the core runs once per
# layer per step and this is a plain rebinding at patch time.
_CONV_FN = None
_CONV_UPDATE = None
_RECURRENT_FWD = None


def _gdn_recurrent_update(
    *,
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens=None,
):
    """Run the decode recurrence through ``fused_recurrent_gated_delta_rule_fwd``.

    That entry takes the decay and the delta multiplier **already computed** and
    is the only decode entry upstream gives a speculative-decoding form to
    (``num_accepted_tokens``), so the spec and non-spec decode paths run through
    one implementation instead of splitting onto two.

    Its state contract differs from vLLM's in two ways and both are handled
    here, around the call: the pool is ``[slots, HV, V, K]`` while the entry
    wants ``[slots, HV, K, V]``, and it updates those rows **in place**.  Rows
    are gathered once into a dense ``[n, HV, K, V]`` buffer (never
    ``pool[idx]`` on the strided cache -- that materializes the whole pool on
    LPU) and scattered back afterwards.

    Returns ``(out, initial_state)``, the same pair the sigmoid-gating twin
    returns, so the call sites are unchanged.  Falls back to that twin when no
    FlagGems backend exports the name.
    """
    if _RECURRENT_FWD is None:
        raise RuntimeError(
            "GDN decode recurrence unavailable: the FlagGems evas backend does "
            "not export fused_recurrent_gated_delta_rule_fwd. This tree keeps "
            "no local copy of the recurrence any more."
        )
    B, _, _, Kdim = q.shape
    n = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Same derivation the sigmoid-gating entry performs internally, kept
    # identical on purpose: the two arms must not differ in the gating math.
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    g = -torch.exp(A_log.to(torch.float32)) * _softplus(x)
    beta_t = torch.sigmoid(b.to(torch.float32))
    # The entry takes both gating tensors per (token, head) with the batch axis
    # spelled out: the kernel shards dim 2 across the dies, so a 2-D
    # [tokens, HV] input raises IndexError inside the dual-die shard.
    if g.dim() == 2:
        g = g.reshape(q.shape[0], q.shape[1], v.shape[2])
    if beta_t.dim() == 2:
        beta_t = beta_t.reshape(q.shape[0], q.shape[1], v.shape[2])

    rows = (
        torch.arange(n, device=initial_state.device)
        if ssm_state_indices is None
        else ssm_state_indices.reshape(-1)[:n]
    )
    gathered = _gdn_gather_rows(initial_state, rows)          # [n, HV, V, K]
    packed = gathered.transpose(-1, -2).contiguous()          # [n, HV, K, V]

    out, _ = _RECURRENT_FWD(
        q,
        k,
        v,
        g,
        beta_t,
        Kdim**-0.5,
        packed,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        # Always pass indices, never None: the None form selects the kernel's
        # non-continuous-batching specialisation, which does not compile in the
        # dual-die body (CompilationError at 35:12, AttributeError at 128:20).
        # The rows are already selected, so this is the identity mapping.
        ssm_state_indices=torch.arange(n, device=packed.device),
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )
    _gdn_scatter_rows(initial_state, rows, packed.transpose(-1, -2))
    return out, initial_state


def _gdn_conv_impls():
    """Public ``flag_gems`` conv names when the evas backend exports them.

    The public name resolves to the pinned upstream kernel once its adoption
    pin graduates and to the FlagGems twin before that, so production follows
    the pin status instead of a second copy of that decision.  The local pure
    implementations stay as the fallback for a tree that does not carry the
    backend yet.
    """
    try:
        import flag_gems

        fn = getattr(flag_gems, "causal_conv1d_fn", None)
        update = getattr(flag_gems, "causal_conv1d_update", None)
        prep = getattr(flag_gems, "fused_post_conv_prep_triton", None)
        chunk = getattr(flag_gems, "chunk_gated_delta_rule_fwd", None)
        recurrent = getattr(flag_gems, "fused_recurrent_gated_delta_rule_fwd", None)
        # A/B lever: VLLM_FL_GDN_RECURRENT=0 pins the decode recurrence to the
        # local twin, so the two arms can be compared on one carrier without
        # touching the pin table.
        if os.environ.get("VLLM_FL_GDN_RECURRENT", "1") == "0":
            recurrent = None
        if (
            fn is not None
            and update is not None
            and prep is not None
            and chunk is not None
        ):
            return (
                fn,
                update,
                prep,
                chunk,
                # Decode recurrence: optional, the local twin covers it.
                recurrent,
                "flag_gems public names (pin status decides each arm)",
            )
    except Exception as exc:  # noqa: BLE001 - fall back, but say why
        logger.warning("evas GDN patch: flag_gems conv unavailable (%s)", exc)
    raise RuntimeError(
        "GDN operators unavailable: the FlagGems evas backend must export "
        "causal_conv1d_fn, causal_conv1d_update, fused_post_conv_prep_triton "
        "and chunk_gated_delta_rule_fwd. This tree keeps no local copy of them."
    )


def _set_conv_impls(fn, update, prep, chunk, recurrent) -> None:
    """Bind the core's GDN hooks and the importer-level module attributes."""
    global _CONV_FN, _CONV_UPDATE, _POST_CONV_PREP, _CHUNK_FWD, _RECURRENT_FWD
    _CONV_FN, _CONV_UPDATE, _POST_CONV_PREP, _CHUNK_FWD = fn, update, prep, chunk
    _RECURRENT_FWD = recurrent


_gdn_patched = False


def apply_lpu_gdn_patch() -> None:
    """Monkey-patch vLLM GDN conv + QwenGatedDeltaNetAttention on LPU.  Idempotent.

    Two independent things live here because both replace Triton ops from the
    same mamba/linear-attention family:

    1. ``mamba.ops.causal_conv1d.{causal_conv1d_fn, causal_conv1d_update}``
       rebound to the FlagGems conv entries (pin decides kernel or twin).
       The upstream Triton kernels do
       not compile under evas-triton-opt (``tts.load`` legalization error),
       which is a compile-phase dead end, not a runtime one -- so every LPU
       importer needs a non-Triton form.  Seven modules import from that
       one site (``short_conv``, ``mamba_mixer{,2}``, ``plamo2``, and the
       qwen/kimi/olmo GDN subclasses), so the rebind is done at the **operator**
       level, not per importer.  ``from ..ops.causal_conv1d import x`` freezes
       the binding at import time, so ``kda.py`` additionally redirects the
       KDA importer's local names.

       **Contract coverage is partial, by design.**  The fallbacks implement the
       ``query_start_loc`` / ``has_initial_state`` / ``cache_indices`` contract
       that ``short_conv``, ``plamo2`` and the qwen/kimi/olmo GDN subclasses use
       for the flat addressing path.  ``mamba_mixer.py`` and ``mamba_mixer2.py``
       additionally pass the paged-block addressing group
       (``block_idx_first_scheduled_token`` / ``block_idx_last_scheduled_token``
       / ``initial_state_idx`` / ``num_computed_tokens`` /
       ``block_size_to_align``) and, on the decode entry, the spec-decode group
       (``num_accepted_tokens`` / ``query_start_loc`` / ``max_query_len``).
       Neither group is implemented and each is **rejected by name with a
       ``TypeError``, never silently ignored** -- an ignored ``initial_state_idx``
       reads the wrong conv-state row, an ignored ``num_accepted_tokens``
       convolves the wrong token count.  That rejection is enforced inside the
       FlagGems twin, which every LPU importer now reaches.

       Two consequences worth stating plainly: mamba-1/2 do not run on LPU yet
       (add the paged addressing to **both** entry points when one is needed),
       and the GDN fork's own spec-decode branch (``spec_sequence_masks`` set,
       i.e. MTP / speculative decoding) now raises rather than convolving the
       wrong token count -- so speculative decoding of qwen3.5/3.6 is not
       supported on LPU either.  Non-spec decode, prefill and the mixed
       ``split_non_spec`` step are the supported shapes.

    2. ``QwenGatedDeltaNetAttention._forward_core`` -> ``_gdn_forward_core_lpu``
       (plus the non-spec decode entry and the kernel warmup).

    GDN is split into per-model subclasses under
    ``vllm.model_executor.layers.mamba.gdn.{qwen,kimi,olmo}_gdn_linear_attn``;
    this re-keys the fork to ``QwenGatedDeltaNetAttention``, which covers qwen3.5
    / qwen3.6 / qwen3-next-moe.  The kimi/olmo subclasses keep their own
    ``_forward_core``; kimi is handled separately by ``kda.py``, olmo has no LPU
    demand yet.
    """
    global _gdn_patched
    if _gdn_patched:
        return

    # (1) operator-level conv fallback -- no class import required.
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv
    except ImportError:
        # A silent skip here would hide an upstream rename just like (2) below:
        # conv stays on the broken Triton path with no trace.
        logger.warning(
            "evas GDN patch partial: cannot import "
            "vllm.model_executor.layers.mamba.ops.causal_conv1d "
            "(upstream rename?); conv redirects SKIPPED"
        )
        if os.environ.get("VLLM_FL_STRICT_UPSTREAM_REBIND") == "1":
            raise
        _conv = None
    if _conv is not None and not getattr(
        _conv.causal_conv1d_fn, "__fl_gdn_lpu__", False
    ):
        fn, update, prep, chunk, recurrent, source = _gdn_conv_impls()
        _set_conv_impls(fn, update, prep, chunk, recurrent)
        _conv.causal_conv1d_fn = fn
        _conv.causal_conv1d_fn.__fl_gdn_lpu__ = True  # type: ignore[attr-defined]
        _conv.causal_conv1d_update = update
        _conv.causal_conv1d_update.__fl_gdn_lpu__ = True  # type: ignore[attr-defined]
        logger.info("evas GDN patch: conv redirects -> %s", source)

    # (2) Qwen GDN forward core.
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
    except ImportError:
        # Never mark this done silently again (registry version_note): this
        # is a PARTIAL state -- recording it as success would leave the three
        # forward rebinds below permanently skipped with zero trace on an
        # upstream rename.  The conv rebinds from (1) may or may not be
        # live (handler (1) has its own failure path); say which below.
        conv_state = (
            "conv redirects applied"
            if _conv is not None
            else "conv redirects SKIPPED as well"
        )
        logger.warning(
            "evas GDN patch partial: cannot import QwenGatedDeltaNetAttention "
            f"(upstream rename?); {conv_state}, forward rebinds SKIPPED"
        )
        if os.environ.get("VLLM_FL_STRICT_UPSTREAM_REBIND") == "1":
            raise
        _gdn_patched = True
        return
    QwenGatedDeltaNetAttention._forward_core = _gdn_forward_core_lpu
    QwenGatedDeltaNetAttention._forward_core_decode_non_spec = (
        _gdn_forward_core_decode_non_spec_lpu
    )
    QwenGatedDeltaNetAttention._warmup_prefill_kernels = _gdn_warmup_noop
    _gdn_patched = True

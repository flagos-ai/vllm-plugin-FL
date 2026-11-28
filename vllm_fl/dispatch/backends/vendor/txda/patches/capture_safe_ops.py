# Copyright (c) 2026 BAAI. All rights reserved.

"""Capture-safe rewrites for TXDA operators that sync during graph capture.

Graph capture on TXDA fails whenever the captured region performs a stream
synchronization. FlagGems' generic ``lt`` / ``le`` implementations do exactly
that: capturing ``x < 0`` (or ``x <= 0``) aborts with

    TXDA error: (txStreamSynchronize(stream)) = Operation not permitted
    while the stream is capturing.

``ge``, ``gt``, ``eq`` and ``ne`` are fine even though all six share the same
source shape, so this is specific to how ``lt`` / ``le`` are dispatched on this
backend rather than a general pointwise limitation.

vLLM hits it in ``get_masked_input_and_mask`` (vocab-parallel embedding), which
compares ``input_ < org_vocab_end_index`` for every forward. That makes the V1
graph path unusable for any model with a vocab-parallel embedding.

The rewrites below are exact identities, so numerical behaviour is unchanged:

    a <  b   ==  b >  a
    a <= b   ==  ~(a > b)  ==  b >= a

They are installed only on TXDA; every other backend keeps FlagGems' kernels.

Removal condition: delete this module once FlagGems' ``lt`` / ``le`` execute
without synchronizing inside a TXDA capture stream.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_installed = False


def install_capture_safe_ops() -> None:
    """Route TXDA ``lt`` / ``le`` through the capture-safe comparisons."""
    global _installed
    if _installed:
        return

    import flag_gems

    aten_lib = getattr(flag_gems, "aten_lib", None)
    if aten_lib is None:
        logger.warning(
            "FlagGems does not expose its aten Library; skipping the "
            "capture-safe comparison rewrite"
        )
        return

    def _floating(a: torch.Tensor) -> bool:
        return a.dtype.is_floating_point or a.dtype.is_complex

    # ``A < B`` for non-NaN operands equals ``B > A``, and ``A <= B`` equals
    # ``B >= A``. The reversed form is used rather than ``!(A >= B)``: the
    # negation returns True when either side is NaN (``ge``/``gt`` are False
    # for NaN), while the original comparison must be False. Swapping operands
    # keeps the native NaN behaviour and avoids float-only helpers on integer
    # tensors, where ``isnan`` does not apply.
    def lt(A, B):
        if isinstance(B, torch.Tensor):
            return torch.gt(B, A)
        if not _floating(A):
            return torch.gt(torch.full_like(A, B), A)
        return torch.logical_not(torch.ge(A, B)) & ~torch.isnan(A)

    def le(A, B):
        if isinstance(B, torch.Tensor):
            return torch.ge(B, A)
        if not _floating(A):
            return torch.ge(torch.full_like(A, B), A)
        return torch.logical_not(torch.gt(A, B)) & ~torch.isnan(A)

    # ``*_out`` overloads must write into the caller's buffer.
    def lt_out(A, B, *, out):
        out.copy_(lt(A, B))
        return out

    def le_out(A, B, *, out):
        out.copy_(le(A, B))
        return out

    # Re-register into the same IMPL library FlagGems enabled, which overrides
    # the kernels it installed for this dispatch key.
    registered = 0
    for op, fn in (
        ("aten::lt.Tensor", lt),
        ("aten::lt.Scalar", lt),
        ("aten::le.Tensor", le),
        ("aten::le.Scalar", le),
        ("aten::lt.Tensor_out", lt_out),
        ("aten::lt.Scalar_out", lt_out),
        ("aten::le.Tensor_out", le_out),
        ("aten::le.Scalar_out", le_out),
    ):
        try:
            aten_lib.impl(op, fn, "PrivateUse1")
            registered += 1
        except Exception:  # pragma: no cover - optional operator
            continue

    if registered == 0:
        # Do not return here: the remaining rewrites below are independent of
        # the comparison overrides, and skipping them would leave the capture
        # exposed to the other synchronising ops just because this one failed.
        logger.warning(
            "Could not override aten lt/le on PrivateUse1; TXDA graph capture "
            "may still synchronize on these comparisons"
        )
    else:
        logger.info(
            "Installed capture-safe TXDA comparisons (lt->gt, le->ge; %d ops)",
            registered,
        )
    _install_masked_fill(aten_lib)
    _install_pow_square(aten_lib)
    _install_gdn_split_ba()
    _install_moe_meta_ondevice()
    _install_gdn_qkv_gate()
    _install_capture_safe_where()
    _install_new_zeros_device()
    _installed = True


def _masked_fill_impl(mask, inp, value):
    """Capture-safe ``masked_fill`` built from ``where``.

    ``torch.masked_fill`` / ``masked_fill_`` fall back to the ATen CPU kernel
    on TXDA, which copies the tensor to host and synchronizes, aborting graph
    capture. vLLM's vocab-parallel embedding calls it on every forward to zero
    the out-of-shard rows.

    ``torch.where`` with a device-resident fill tensor captures cleanly and is
    an exact replacement. An earlier ``inp*(1-m) + m*value`` form was not: it
    produced NaN for a ``-inf`` fill (``0 * -inf``), kept NaN values that the
    mask should overwrite, and promoted integer inputs to float. ``where``
    keeps the input dtype and the exact original values, and is validated
    against ``torch.masked_fill`` for int64/int32/float32/bfloat16/bool across
    scalar, float and ``-inf`` fills.
    """
    mask = mask.expand(inp.shape)
    if isinstance(value, torch.Tensor):
        fill = value.to(dtype=inp.dtype, device=inp.device).expand(inp.shape)
    else:
        fill = torch.full_like(inp, value)
    return torch.where(mask, fill, inp)


def _masked_fill_inplace_impl(mask, inp, value, out=None):
    target = inp if out is None else out
    target.copy_(_masked_fill_impl(mask, inp, value))
    return target


def _install_masked_fill(aten_lib) -> None:
    def masked_fill(inp, mask, value):
        return _masked_fill_impl(mask, inp, value)

    def masked_fill_(inp, mask, value):
        return _masked_fill_inplace_impl(mask, inp, value)

    def masked_fill_out(inp, mask, value, *, out):
        return _masked_fill_inplace_impl(mask, inp, value, out=out)

    installed = 0
    for op, fn in (
        ("aten::masked_fill.Scalar", masked_fill),
        ("aten::masked_fill.Tensor", masked_fill),
        ("aten::masked_fill_.Scalar", masked_fill_),
        ("aten::masked_fill_.Tensor", masked_fill_),
        ("aten::masked_fill.Scalar_out", masked_fill_out),
        ("aten::masked_fill.Tensor_out", masked_fill_out),
    ):
        try:
            aten_lib.impl(op, fn, "PrivateUse1")
            installed += 1
        except Exception:  # pragma: no cover - operator may be absent
            continue
    if installed:
        logger.info(
            "Installed capture-safe TXDA masked_fill (where form; %d ops)",
            installed,
        )


def _install_pow_square(aten_lib) -> None:
    """Route ``x.pow(2)`` to ``x.square()``.

    FlagGems' ``pow.Tensor_Scalar`` uses ``tl_extra_shim.pow`` (an extern
    device call) which synchronizes inside a TXDA capture stream, while
    ``square`` lowers to plain elementwise multiply and captures cleanly.
    vLLM's ``ir.ops.rms_norm`` squares the input on every forward, so without
    this the first RMSNorm aborts the capture.
    """

    # Anything that is not the squared case must go to the regular
    # implementation. Calling ``torch.pow`` here would re-enter this same
    # PrivateUse1 kernel (it is registered for exactly this overload) and
    # recurse until RecursionError, so redispatch below the registration
    # instead. Only exponent 2 is rewritten; every other exponent keeps the
    # upstream semantics.
    _fallback = torch._C.DispatchKeySet(
        torch._C.DispatchKey.CompositeExplicitAutograd
    )

    def _is_square(exponent) -> bool:
        return isinstance(exponent, (int, float)) and float(exponent) == 2.0

    def pow_tensor_scalar(self, exponent):
        if _is_square(exponent):
            return torch.square(self)
        return torch.ops.aten.pow.Tensor_Scalar.redispatch(
            _fallback, self, exponent
        )

    def pow_tensor_scalar_(self, exponent):
        if _is_square(exponent):
            return self.square_()
        return torch.ops.aten.pow_.Scalar.redispatch(_fallback, self, exponent)

    def pow_tensor_tensor(self, exponent):
        return torch.ops.aten.pow.Tensor_Tensor.redispatch(
            _fallback, self, exponent
        )

    def pow_tensor_tensor_(self, exponent):
        return torch.ops.aten.pow_.Tensor.redispatch(_fallback, self, exponent)

    installed = 0
    for op, fn in (
        ("aten::pow.Tensor_Scalar", pow_tensor_scalar),
        ("aten::pow.Tensor_Tensor", pow_tensor_tensor),
        ("aten::pow_.Scalar", pow_tensor_scalar_),
        ("aten::pow_.Tensor", pow_tensor_tensor_),
    ):
        try:
            aten_lib.impl(op, fn, "PrivateUse1")
            installed += 1
        except Exception:  # pragma: no cover - operator may be absent
            continue
    if installed:
        logger.info("Installed capture-safe TXDA pow(2) -> square (%d ops)", installed)


def _install_gdn_split_ba() -> None:
    """Make GDN's ``split_ba`` return tensor slices the capture stream accepts.

    ``GatedDeltaNetAttention.split_ba`` builds the TP-local beta/alpha
    vectors with ``ba.chunk(2, -1)`` followed by a head-range slice. On the
    serving path the leading ``ba_proj`` projection output is itself already
    non-contiguous, so both the chunk and the head slice are views with
    strides like ``(8, 1)`` for a ``(16, 4)`` result.

    The GDN op then calls ``b.contiguous()`` per layer per step; on TXDA that
    lowers to ``txdnnStrideCopy``, which fails as soon as the stream is
    capturing. Materialising the slice through ``copy_`` is not enough either
    -- the source shares the same strided storage and the copy lowers to the
    same call.

    Splitting the *contiguous copy* of ``ba`` sidesteps both: ``narrow`` on a
    contiguous tensor returns a contiguous view, so the caller's
    ``.contiguous()`` becomes a no-op and no strided copy is ever issued.
    """
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
    except Exception:  # pragma: no cover - the model module ships with vLLM
        logger.debug("GDN split_ba patch skipped: module unavailable", exc_info=True)
        return

    if getattr(QwenGatedDeltaNetAttention, "_fl_txda_contig_ba", False):
        return

    original = QwenGatedDeltaNetAttention.split_ba

    # Scalar zeros are dtype/device invariant, so build each one once and reuse
    # it. Allocating a fresh ``torch.zeros(())`` per call launched FlagGems'
    # ``zeros_kernel`` per layer per step; more importantly it made the patched
    # path do device work the unpatched warmup never did, so the kernel was
    # first compiled *inside* the graph capture and its tuner sync invalidated
    # the capture stream. ``graph.py`` now installs these patches before its
    # eager warmup, which also populates this cache eagerly.
    zero_cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}

    def _scalar_zero(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (dtype, device)
        zero = zero_cache.get(key)
        if zero is None or zero.device != device:
            zero = torch.zeros((), dtype=dtype, device=device)
            zero_cache[key] = zero
        return zero

    def split_ba(self, ba):
        b, a = original(self, ba)
        # ``aten::contiguous`` and ``copy_`` both lower to ``txdnnStrideCopy``
        # for a strided PrivateUse1 source, which is illegal while capturing.
        # An elementwise op materialises the view through the regular pointwise
        # kernel instead, so use one to produce contiguous b/a.
        # ``torch.add`` with a same-dtype tensor (not a Python scalar) keeps
        # the pointwise kernel on the fast path; the scalar form re-promotes
        # the dtype through FlagGems' fused path, which synchronizes.
        return (
            b if b.is_contiguous() else b.add(_scalar_zero(b.dtype, b.device)),
            a if a.is_contiguous() else a.add(_scalar_zero(a.dtype, a.device)),
        )

    QwenGatedDeltaNetAttention.split_ba = split_ba
    QwenGatedDeltaNetAttention._fl_txda_contig_ba = True
    logger.info("Installed capture-safe TXDA GDN split_ba (contiguous input)")


def _install_moe_meta_ondevice() -> None:
    """Build the compact-grouped MoE meta entirely on device.

    FlagGems' ``_build_moe_compact_meta_from_topk`` reads ``topk_ids`` back to
    the host (``.to("cpu")`` in the default path, ``bool(...item())`` in the
    "gpu" path) and calls ``torch.argsort``. Every one of those synchronizes on
    TXDA -- ``sort``/``topk``/boolean-indexing all fail once a stream is
    capturing -- so the fused MoE aborts the capture.

    The replacement is a stable counting sort expressed with primitives that do
    capture cleanly (compare, ``cumsum``, ``arange``, ``gather``,
    ``scatter_add_``). It emits one segment per expert, empty ones included,
    which the grouped GEMM already skips (``if m_size > 0``), so the kernel grid
    is a constant ``num_experts`` and needs no host round-trip.
    """
    # The on-device meta builder lives in the module that owns the *public*
    # ``flaggems_vllm.fused_experts_impl`` symbol. Resolve it from that symbol
    # (never from FlagGems' private ``runtime.backend._tsingmicro`` path) so the
    # patch always targets the same implementation the dispatcher calls.
    from vllm_fl.ops.fused_moe.fused_moe_utils import _public_fused_moe_module

    fused_moe = _public_fused_moe_module()
    if fused_moe is None:
        logger.warning(
            "MoE meta patch skipped: public fused_experts_impl unavailable"
        )
        return

    logger.info("MoE meta patch: reached installer (file=%s)", fused_moe.__file__)
    if getattr(fused_moe, "_fl_txda_ondevice_meta", False):
        logger.info("MoE meta patch: already installed")
        return

    # ``_build`` only receives topk_ids, so the expert count is published by the
    # ``fused_experts_impl`` wrapper below (it has ``E`` from the weight shape).
    state = {"num_experts": 0}

    def _build(topk_ids, device):
        dev = topk_ids.device
        top_k = int(topk_ids.shape[1])
        n = int(topk_ids.numel())
        if n == 0:
            return None
        e_all = state["num_experts"]
        if e_all <= 0:
            # Fall back to the widest expert id ever seen; only used before the
            # plugin publishes the real count.
            e_all = int(topk_ids.max().item()) + 1 if topk_ids.numel() else 1
        flat_e = topk_ids.reshape(-1).to(torch.int64)
        valid = flat_e >= 0
        # Map invalid entries to the "+1" sentinel bucket so they never join a
        # real expert's segment.
        e = torch.where(valid, flat_e, torch.full_like(flat_e, e_all))
        ne = e_all + 1
        ar = torch.arange(ne, device=dev)
        rows = torch.arange(n, device=dev)

        onehot = e[:, None] == ar[None, :]
        cum = torch.cumsum(onehot.to(torch.int64), 0)
        rank = torch.gather(cum, 1, e.unsqueeze(1)).squeeze(1) - 1
        cnt = cum[-1][:e_all]
        off = torch.cumsum(cnt, 0) - cnt

        # Real experts occupy [0, n_valid) after grouping; sentinel entries go
        # to the tail. Clamp so the scatter stays in bounds.
        dst = torch.clamp(off[e] + rank, max=n - 1)
        perm = torch.zeros(n, device=dev, dtype=torch.int64)
        perm.scatter_add_(0, dst, rows)

        seg_starts = off.to(torch.int32)
        seg_ends = (off + cnt).to(torch.int32)
        seg_experts = torch.arange(e_all, device=dev, dtype=torch.int32)

        # ``avg_rows_per_seg`` only selects the grouped GEMM tile size, and it
        # must be a host value. Reading the true count would cost a D2H sync
        # while capturing, so use the upper bound n/e_all: the same value is
        # used at capture and at every replay, which is what matters for a
        # captured kernel.
        n_valid = n
        return {
            "n_post": n_valid,
            "n_valid": n_valid,
            "n_mm": e_all,
            "avg_rows_per_seg": n_valid / max(e_all, 1),
            "valid_ids": perm,
            # Integer division (``//``, ``div(rounding_mode=)`` and shifts)
            # all synchronize on TXDA. ``perm`` is a non-negative row index, so
            # a float divide-then-truncate is exact and captures cleanly.
            "a_rows_hidden": (perm.to(torch.float32) / float(top_k))
            .to(torch.int64)
            .contiguous(),
            "a_rows_flat": perm,
            "seg_starts": seg_starts,
            "seg_ends": seg_ends,
            "seg_experts": seg_experts,
            "meta_source": "txda_ondevice",
        }

    try:
        fused_moe._build_moe_compact_meta_from_topk = _build
        fused_moe._fl_txda_ondevice_meta = True
    except Exception:
        logger.warning("MoE meta patch install failed", exc_info=True)
        return
    # The plugin publishes the true expert count from ``global_num_experts``
    # when it dispatches, since the weight shape is not visible here.
    fused_moe._fl_txda_set_num_experts = lambda n: state.__setitem__(
        "num_experts", int(n)
    )
    logger.info("Installed capture-safe TXDA MoE meta (on-device counting sort)")


def _install_gdn_qkv_gate() -> None:
    """Keep the gated qkv split contiguous so ``reshape`` stays a view.

    With ``attn_output_gate`` on, vLLM splits the projected ``qkv`` into
    ``q_gate``/``k``/``v``, views it as ``(tokens, heads, 2*head_dim)``, chunks
    that into ``q`` and ``gate``, then reshapes both. ``q`` ends up with a
    stride like ``(2560, 128, 1)``, so the reshape copies -> ``txdnnStrideCopy``
    -> the capture aborts.

    Rather than reimplement the split, hand the *same* upstream code a
    contiguous ``qkv``: a whole-tensor elementwise add materialises the
    projection once (the only copy on this path), and every later
    view/chunk/reshape is then a free stride change.
    """
    try:
        from vllm.model_executor.models import qwen3_next
    except Exception:  # pragma: no cover - module absent on non-Qwen builds
        logger.debug("GDN qkv gate patch skipped: qwen3_next unavailable")
        return

    cls = getattr(qwen3_next, "Qwen3NextAttention", None)
    if cls is None or getattr(cls, "_fl_txda_qkv_gate", False):
        return

    original = cls._project_qkv_gate

    def _project_qkv_gate(self, qkv, positions):
        if getattr(self, "use_fused_qk_norm_rope_gate", False) or not getattr(
            self, "attn_output_gate", False
        ):
            return original(self, qkv, positions)

        # Gated path: ``qkv.split`` yields views, and reshaping the chunked
        # ``q``/``gate`` from a strided view needs a copy. Materialise the
        # q_gate slice once (contiguous source -> capture-safe ``copy_``) so the
        # subsequent view/chunk/reshape chain stays a pure stride change.
        q_size = self.q_size
        q_gate = qkv.narrow(-1, 0, q_size * 2)
        if not q_gate.is_contiguous():
            q_gate = q_gate.add(
                torch.zeros((), dtype=q_gate.dtype, device=q_gate.device)
            )
        orig_shape = q_gate.shape[:-1]
        head_dim_total = q_gate.shape[-1] // self.num_heads
        qg = q_gate.view(*orig_shape, self.num_heads, head_dim_total)
        # The two halves of each head are interleaved (``[q|gate]`` per head),
        # so flattening a ``chunk``/``narrow`` view needs a cross-stride copy.
        # Materialise each half once with an elementwise add, then the reshape
        # is a plain view.
        half = head_dim_total // 2
        q = qg.narrow(-1, 0, half)
        gate = qg.narrow(-1, half, half)
        zero = torch.zeros((), dtype=qkv.dtype, device=qkv.device)
        q = q.add(zero).reshape(*orig_shape, -1)
        gate = gate.add(zero).reshape(*orig_shape, -1)

        k = qkv.narrow(-1, q_size * 2, self.kv_size)
        v = qkv.narrow(-1, q_size * 2 + self.kv_size, self.kv_size)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    cls._project_qkv_gate = _project_qkv_gate
    cls._fl_txda_qkv_gate = True
    logger.info("Installed capture-safe TXDA GDN qkv gate (contiguous qkv)")


def _install_capture_safe_where() -> None:
    """Replace ``torch.where`` with an arithmetic select.

    ``torch.where`` is the one core primitive that fails inside a TXDA capture
    stream (add/mul/sum/mean/cumsum/gather/scatter_add/arange/comparisons all
    work). M-RoPE's ``apply_interleaved_rope`` uses it to interleave the
    time/height/width frequency layouts, so every forward aborts the capture.

    ``w * x + (1-w) * y`` is the same selection in elementwise arithmetic, which
    captures cleanly. Only scalar conditions or broadcastable weight masks are
    supported -- that is all vLLM's RoPE path and the other captured call sites
    need.
    """
    try:
        from vllm.model_executor.layers.rotary_embedding import mrope
    except Exception:  # pragma: no cover - module layout differs by version
        logger.debug("M-RoPE where patch skipped: mrope unavailable")
        return

    if getattr(mrope, "_fl_txda_interleave", False):
        return

    def _select(cond, x, y):
        w = cond.to(x.dtype)
        return x * w + y * (1 - w)

    def apply_interleaved_rope(x, mrope_section):
        channels = torch.arange(x.shape[-1], device=x.device)
        is_height = (channels % 3 == 1) & (channels < mrope_section[1] * 3)
        is_width = (channels % 3 == 2) & (channels < mrope_section[2] * 3)
        result = _select(is_height, x[1], x[0])
        return _select(is_width, x[2], result)

    mrope.apply_interleaved_rope = apply_interleaved_rope
    cls = getattr(mrope, "MRotaryEmbedding", None)
    if cls is not None and not getattr(cls, "_fl_txda_idx", False):
        orig_native = cls.forward_native

        def forward_native(self, positions, query, key=None, offsets=None):
            # ``cos_sin_cache[positions]`` goes through FlagGems' ``index``
            # kernel, whose dim-0 fast path reshapes a strided index tensor
            # (``txdnnStrideCopy``). ``index_select`` is blacklisted from
            # FlagGems and runs the native TXDA kernel instead.
            if positions.ndim > 1 and not positions.is_contiguous():
                positions = positions.add(
                    torch.zeros((), dtype=positions.dtype, device=positions.device)
                )
            return orig_native(self, positions, query, key, offsets)

        cls.forward_native = forward_native
        cls._fl_txda_idx = True
    logger.info("Installed capture-safe TXDA M-RoPE (interleave + contiguous pos)")


def _install_new_zeros_device() -> None:
    """Fix FlagGems' ``new_zeros`` scratch allocation on non-CUDA devices.

    ``flag_gems.ops.new_zeros._fill_any`` decides how to zero a buffer with

        if t.device.type == "cuda" or t.is_pinned(): _fill_device(t)
        else: scratch = torch.empty(..., device="cuda"); ...

    TXDA is ``PrivateUse1`` (device type ``"txda"``) and has no pinned memory,
    so every zero-fill takes the ``else`` branch and allocates a CUDA tensor --
    which aborts with ``Torch not compiled with CUDA enabled``. FLA's chunked
    GDN kernels call ``new_zeros`` during warmup and inference.

    The fallback exists only to move zeros from device to *pageable host*
    storage; a device buffer on TXDA can be zeroed in place by the same
    device kernel, so route the device case to ``_fill_device`` directly.
    """
    try:
        import importlib

        nz = importlib.import_module("flag_gems.ops.new_zeros")
    except Exception:  # pragma: no cover - module layout differs by version
        logger.debug("new_zeros device patch skipped: module unavailable")
        return

    if getattr(nz, "_fl_txda_device_fill", False):
        return

    original = nz._fill_any

    def _fill_any(t):
        if t.numel() == 0 or t.device.type == "meta":
            return
        if t.device.type == "cpu" and not t.is_pinned():
            # Pageable host storage: zero it directly. Upstream copies from a
            # CUDA scratch buffer, which does not exist on a PrivateUse1-only
            # torch build. FLA's ``cu_seqlens.new_zeros(1)`` hits this on the
            # host-side metadata tensors.
            t.zero_()
            return
        if t.device.type in ("cuda", "txda") or t.is_pinned():
            nz._fill_device(t)
            return
        original(t)

    nz._fill_any = _fill_any
    nz._fl_txda_device_fill = True
    logger.info("Installed capture-safe TXDA new_zeros (device fill)")

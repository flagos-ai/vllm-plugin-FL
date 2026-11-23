# Copyright (c) 2026 BAAI. All rights reserved.
"""Evas (LPU) backend operator registrations."""

from __future__ import annotations

import functools
import importlib

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available()."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def _lazy_fn(module_path: str, attr_name: str, is_available_fn):
    """Create a lazy-loaded function wrapper for model-specific dispatch ops.

    Defers the actual import to first call time via ``importlib``, avoiding
    a hard import of model-specific modules during ``register_builtins()``.
    The ``module_path`` follows the same dot-separated string-path convention
    as ``EvasBackend.attention_backend()``.

    Args:
        module_path: Fully qualified module path (e.g.
            ``"vllm_fl.dispatch.backends.vendor.evas.patches.ops_fallback.dsv4_swa_metadata"``).
        attr_name: Function name within the module.
        is_available_fn: Callable for ``OpImpl.is_available()`` check.

    Returns:
        A callable wrapper that loads and delegates to the real function on
        first invocation. The wrapper carries ``_is_available`` so
        ``OpImpl.is_available()`` works without triggering the import.
    """
    _fn = None

    def wrapper(*args, **kwargs):
        nonlocal _fn
        if _fn is None:
            mod = importlib.import_module(module_path)
            _fn = getattr(mod, attr_name)
        return _fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """Register all evas (VENDOR) operator implementations."""
    from .evas import EvasBackend

    backend = EvasBackend()
    is_avail = backend.is_available

    # fused_q_kv_rmsnorm routes to the evas pure-PyTorch override exported in
    # flag_gems (flag_gems/runtime/backend/_evas/fused/deepseek_v4_attention_fused_q_kv_rmsnorm.py).

    # Model-specific dispatch ops: lazy-loaded via string paths (same convention
    # as EvasBackend.attention_backend()) so register_builtins() stays model-free.

    impls = [
        # ===== Activation =====
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gelu_and_mul, is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # silu_and_mul_with_clamp has no vendor.evas impl: its flag_os impl is
        # LPU-safe since upstream added it, and the vendor copy predated the
        # (obj, x) dispatch convention. See dispatch/config/evas.yaml.
        # ===== Normalization =====
        # rms_norm needs no vendor:evas impl: the flag_os impl resolves through
        # flag_gems.modules.normalization.gems_rms_forward to the top-level
        # _evas/ops/rms_norm override.
        # ===== Rotary Embedding =====
        # rotary_embedding needs no vendor:evas impl: the flag_os impl resolves
        # through flag_gems.modules.rotary_embedding.gems_rope_forward to the
        # top-level _evas/fused/rotary_embedding override.
        # ===== DSV4 attention fused rmsnorm =====
        OpImpl(
            op_name="fused_q_kv_rmsnorm",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "fused_q_kv_rmsnorm", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # ===== DSV4 sparse-attention Indexer (C4A decode) =====
        OpImpl(
            op_name="compute_global_topk_indices_and_lens",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "compute_global_topk_indices_and_lens", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # ===== DSV4 sparse-attention prefill chunk (compressed-topk + SWA) =====
        OpImpl(
            op_name="combine_topk_swa_indices",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "combine_topk_swa_indices", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # ===== DSV4 bf16 sparse-decode MLA (flash_mla_with_kvcache) =====
        # The flaggems DEFAULT impl for this op imports the sm90 Triton kernel
        # from ``flag_gems.fused.flash_mla_with_kvcache`` -- that submodule is
        # NOT the evas override (which lives in the top-level ``flag_gems``
        # namespace after SpecOpRegistrar). The evas VENDOR impl points at
        # the top-level override so CachedOp("flash_mla_with_kvcache") resolves to
        # the pure-PyTorch evas implementation on LPU. Requires the matching
        # ``op_backends.flash_mla_with_kvcache: [vendor:evas]`` entry in
        # dispatch/config/evas.yaml (default order prefers flagos/DEFAULT).
        OpImpl(
            op_name="flash_mla_with_kvcache",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "flash_mla_with_kvcache", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # ===== DSV4 generic FP8 attention (quant_config is not None) =====
        # The flaggems DEFAULT impls for these ops call ``flag_gems.<op>``, which
        # on LPU must resolve to the ``_evas/fused`` pure-PyTorch overrides (uint8
        # E4M3 bytes + fp32 E8M0 block scales, no float8 dtype).  The flaggems
        # impls are gated off by the LPU platform blacklist so they would fall
        # through to the reference ``pass`` stubs, hence we register the top-level
        # overrides as vendor:evas so CachedOp resolves to them on LPU (same
        # pattern as flash_mla_with_kvcache).
        # Exception: ``deepseek_v4_fp8_einsum`` is intentionally NOT registered
        # here.  Its flaggems DEFAULT impl calls ``flag_gems.fp8_einsum``
        # (hopper-style, provided by ``_evas/ops/fp8_einsum.py``), so
        # ``_has_flaggems_op("fp8_einsum")`` is True and no vendor:evas
        # fallback is needed.
        # Requires the matching ``op_backends.<op>: [vendor:evas]`` entries in
        # dispatch/config/evas.yaml.
        OpImpl(
            op_name="fused_inv_rope_fp8_quant",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "fused_inv_rope_fp8_quant", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn(
                "flag_gems",
                "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
                is_avail,
            ),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="dequantize_and_gather_k_cache",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "dequantize_and_gather_k_cache", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # ===== DSV4 sparse-indexer path =====
        # fused_indexer_q_rope_quant: the quant variant (FP8 indexer, selected
        # when the checkpoint declares expert_dtype=fp8).  The evas flag_gems
        # override quantizes Q to FP8 E4M3 and folds the scale into weights.
        # Not whitelist-gated: VENDOR impls bypass the VLLM_FL_FLAGOS_WHITELIST
        # flaggems gate, so without this the op falls through to the reference
        # stub (pass -> None) and the indexer forward crashes unpacking the
        # result.
        OpImpl(
            op_name="fused_indexer_q_rope_quant",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "fused_indexer_q_rope_quant", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # top_k_per_row_prefill / top_k_per_row_decode / pack_seq_triton /
        # unpack_seq_triton: flaggems provides Triton kernels, but on LPU the
        # pointwise-mask / radix-reduction / 3D-grid scatter patterns are a
        # documented compile dead-end -- pure-PyTorch overrides keep the bf16
        # indexer path runnable on LPU (CPU-verified equivalents).
        OpImpl(
            op_name="top_k_per_row_prefill",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "top_k_per_row_prefill", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="top_k_per_row_decode",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "top_k_per_row_decode", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="pack_seq_triton",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "pack_seq_triton", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="unpack_seq_triton",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_lazy_fn("flag_gems", "unpack_seq_triton", is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
        # ===== Attention Backend =====
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.evas",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="evas",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)

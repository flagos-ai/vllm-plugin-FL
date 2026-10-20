# Copyright (c) 2026 BAAI. All rights reserved.

"""Enable vLLM native INT8 (compressed-tensors W8A8) on Sunrise/PTPU."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ENABLED = False
_OOT_KERNEL_CLS = None
_MOE_DECISION_LOGGED = False
_QUANT_PATCHED = False
_MOE_ACT_QUANT_PATCHED = False
_MM_PATCHED = False
_FG_MM_FN = None
_MM_SHAPE_LOGGED = False


def _log_w8a8_moe_decision(selection) -> None:
    """Record, once, which experts class the oracle actually handed back.

    W8A8 MoE reaching the wrong experts class does not raise here; it shows up
    much later as a dtype or contract mismatch deep inside the expert GEMMs, so
    the resolved class is worth stating plainly in the server log.
    """
    global _MOE_DECISION_LOGGED
    if _MOE_DECISION_LOGGED:
        return
    _MOE_DECISION_LOGGED = True

    experts_cls = selection[1] if isinstance(selection, tuple) else None
    experts_name = getattr(experts_cls, "__name__", repr(experts_cls))
    if experts_name == "TritonW8A8Experts":
        logger.info(
            "native-int8: W8A8 MoE resolved to experts=TritonW8A8Experts -> "
            "vLLM functional fused_experts."
        )
    else:
        logger.warning(
            "native-int8: W8A8 MoE resolved to experts=%s, not the expected "
            "TritonW8A8Experts. On PTPU only the functional fused_experts path "
            "is supported for compressed-tensors W8A8.",
            experts_name,
        )


def _build_oot_int8_kernel_cls():
    """Create (once) the OOT-enabled subclass of vLLM's Triton INT8 kernel."""
    global _OOT_KERNEL_CLS
    if _OOT_KERNEL_CLS is not None:
        return _OOT_KERNEL_CLS

    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )

    class OOTTritonInt8ScaledMMLinearKernel(TritonInt8ScaledMMLinearKernel):
        """``TritonInt8ScaledMMLinearKernel`` allowed on the OOT platform.

        Compute is unchanged (pure-Triton ``triton_scaled_mm`` + the patched
        activation quant); only platform gating is relaxed.
        """

        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None

        @classmethod
        def can_implement(cls, c):
            return True, None

    _OOT_KERNEL_CLS = OOTTritonInt8ScaledMMLinearKernel
    return _OOT_KERNEL_CLS


def _register_oot_int8_kernel() -> bool:
    """Put the sunrise Triton INT8 ScaledMM kernel first in the OOT candidates.

    ``vllm_fl.quantization.quant_linear.add_oot_quant_kernel`` clones the CUDA
    candidate list into ``PlatformEnum.OOT``, but only when that key is absent.
    Whether it or this function runs first depends on how the engine was
    launched, so do not rely on either order: register when missing, and always
    move our kernel to the front. Calling this again after the clone therefore
    keeps the CUDA kernels as fallbacks while still preferring ours.
    """
    try:
        from vllm.platforms import PlatformEnum
        from vllm.model_executor.kernels.linear import (
            _POSSIBLE_INT8_KERNELS,
            register_linear_kernel,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: vLLM kernel registry unavailable (%s)", e)
        return False

    kernel_cls = _build_oot_int8_kernel_cls()

    candidates = _POSSIBLE_INT8_KERNELS.get(PlatformEnum.OOT)
    if candidates is None:
        register_linear_kernel(kernel_cls, PlatformEnum.OOT, kernel_type="int8")
        candidates = _POSSIBLE_INT8_KERNELS[PlatformEnum.OOT]
    elif kernel_cls not in candidates:
        candidates.insert(0, kernel_cls)
    elif candidates[0] is not kernel_cls:
        candidates.remove(kernel_cls)
        candidates.insert(0, kernel_cls)
    else:
        return True

    logger.info(
        "native-int8: INT8 linear kernel candidates for PlatformEnum.OOT are "
        "now %s.",
        [k.__name__ for k in candidates],
    )
    return True


def _patch_scaled_int8_quant() -> bool:
    """Route ``vllm._custom_ops.scaled_int8_quant`` to sunrise Triton impl."""
    global _QUANT_PATCHED
    if _QUANT_PATCHED:
        return True
    try:
        import vllm._custom_ops as _vllm_ops
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: vllm._custom_ops unavailable (%s)", e)
        return False

    from ..impl.int8.scaled_int8_quant import scaled_int8_quant as _sunrise_quant

    _orig = getattr(_vllm_ops, "scaled_int8_quant", None)

    def _fl_scaled_int8_quant(input, scale=None, azp=None, symmetric=True):
        return _sunrise_quant(input, scale=scale, azp=azp, symmetric=symmetric)

    _fl_scaled_int8_quant._fl_original = _orig  # type: ignore[attr-defined]
    _vllm_ops.scaled_int8_quant = _fl_scaled_int8_quant
    _QUANT_PATCHED = True
    logger.info(
        "native-int8: patched vllm._custom_ops.scaled_int8_quant -> "
        "sunrise impl/int8 scaled_int8_quant (Triton per-token)."
    )
    return True


def _resolve_fg_scaled_mm():
    """Return FlagGems' autotuned INT8 ``scaled_mm`` (cached)."""
    global _FG_MM_FN
    if _FG_MM_FN is not None:
        return _FG_MM_FN
    from flag_gems.ops.scaled_mm import scaled_mm as _fg
    _FG_MM_FN = _fg
    return _fg


def _patch_triton_scaled_mm() -> bool:
    """Route vLLM's ``triton_scaled_mm`` to FlagGems' autotuned ``scaled_mm``."""
    global _MM_PATCHED
    if _MM_PATCHED:
        return True
    try:
        import vllm.model_executor.kernels.linear.scaled_mm.triton as _mm_mod
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: vLLM triton scaled_mm module unavailable (%s)", e)
        return False

    _orig = getattr(_mm_mod, "triton_scaled_mm", None)
    if _orig is None:
        logger.debug("native-int8: triton_scaled_mm symbol not found; skip.")
        return False
    if getattr(_orig, "_fl_native_int8_mm", False):
        _MM_PATCHED = True
        return True

    def _fl_triton_scaled_mm(
        input, weight, scale_a, scale_b, out_dtype, bias=None, *args, **kwargs
    ):
        global _MM_SHAPE_LOGGED
        fg = _resolve_fg_scaled_mm()

        # Vision encoder QKV (and any batched linear) may pass [..., K].
        # FlagGems / vLLM triton_scaled_mm only accept 2D [M, K]; CUDA
        # cutlass_scaled_mm flattens the same way before the GEMM.
        input_shape = input.shape
        if input.ndim != 2:
            input = input.reshape(-1, input_shape[-1])

        # FlagGems computes input[M,K] @ mat2[K,N]; orient weight to [K,N].
        mat2 = weight
        transposed = False
        if mat2.shape[0] != input.shape[1]:
            mat2 = mat2.t()
            transposed = True

        # FlagGems right-scale accepts scalar / [N] / [1,N] (not [N,1]).
        sb = scale_b
        if sb is not None and sb.ndim == 2 and sb.shape[-1] == 1 and sb.numel() > 1:
            sb = sb.reshape(-1)

        if not _MM_SHAPE_LOGGED:
            _MM_SHAPE_LOGGED = True
            logger.info(
                "native-int8: triton_scaled_mm->FlagGems scaled_mm active. "
                "input=%s (2d=%s) weight=%s (transposed=%s) scale_a=%s "
                "scale_b=%s out_dtype=%s bias=%s",
                tuple(input_shape), tuple(input.shape),
                tuple(weight.shape), transposed,
                tuple(scale_a.shape), tuple(scale_b.shape) if scale_b is not None
                else None, out_dtype, None if bias is None else tuple(bias.shape),
            )

        try:
            out = fg(input, mat2, scale_a, sb, bias=bias, out_dtype=out_dtype)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "native-int8: FlagGems scaled_mm failed (%s); falling back to "
                "vLLM triton_scaled_mm.", e,
            )
            out = _orig(input, weight, scale_a, scale_b, out_dtype, bias,
                        *args, **kwargs)

        if len(input_shape) != 2:
            out = out.view(*input_shape[:-1], out.shape[-1])
        return out

    _fl_triton_scaled_mm._fl_native_int8_mm = True  # type: ignore[attr-defined]
    _fl_triton_scaled_mm._fl_original = _orig  # type: ignore[attr-defined]

    # Patch the defining module, then rebind any module that already did
    # ``from ...scaled_mm.triton import triton_scaled_mm``.
    import sys

    _mm_mod.triton_scaled_mm = _fl_triton_scaled_mm
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "triton_scaled_mm", None) is _orig:
            _mod.triton_scaled_mm = _fl_triton_scaled_mm

    _MM_PATCHED = True
    logger.info(
        "native-int8: patched triton_scaled_mm -> FlagGems scaled_mm "
        "(autotuned INT8 GEMM) on sunrise."
    )
    return True


def _patch_int8_moe_oracle() -> bool:
    """Publish the FL int8 MoE selector to importers that already bound a stale one.

    ``install_fl_w8a8_moe_selector`` only rebinds the oracle module and the
    compressed-tensors scheme module. Any module that did a
    ``from ...oracle.int8 import select_int8_moe_backend`` beforehand still holds
    the stock selector, whose support gate is CUDA/ROCm-only and would reject
    PTPU. Re-runnable on purpose: ``register_oot_ops`` installs the FL selector
    after the import-time patches, so sunrise has to be the last writer, and
    ``apply_sunrise_patches`` calls this again once that has happened.

    The wrapper is otherwise transparent; it exists to log which experts class
    the oracle settles on.
    """
    try:
        import sys

        import vllm.model_executor.layers.fused_moe.oracle.int8 as _int8_oracle
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: int8 MoE oracle unavailable (%s)", e)
        return False

    _orig = _int8_oracle.select_int8_moe_backend
    if getattr(_orig, "_fl_native_int8_moe", False):
        return True

    def _select_int8_moe_backend_oot(config, *args, **kwargs):
        selection = _orig(config, *args, **kwargs)
        _log_w8a8_moe_decision(selection)
        return selection

    _select_int8_moe_backend_oot._fl_native_int8_moe = True  # type: ignore[attr-defined]
    _select_int8_moe_backend_oot._fl_original = _orig  # type: ignore[attr-defined]

    _int8_oracle.select_int8_moe_backend = _select_int8_moe_backend_oot
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "select_int8_moe_backend", None) is _orig:
            _mod.select_int8_moe_backend = _select_int8_moe_backend_oot

    logger.info(
        "native-int8: wrapped select_int8_moe_backend on OOT; the active "
        "selector is %s.",
        getattr(_orig, "__qualname__", _orig),
    )
    return True


def _patch_moe_per_token_quant_int8() -> bool:
    """Replace vLLM's CUDA-only MoE ``per_token_quant_int8`` with sunrise impl.

    Stock kernel uses ``tl.extra.cuda.libdevice.round`` → TANG reports
    ``kernel function contain unknown call`` / ``TANG_ERROR_INVALID_IMAGE``.
    Required once MoE runs true W8A8 (see ``_ensure_dynamic_w8a8_quant_config``).
    """
    global _MOE_ACT_QUANT_PATCHED
    if _MOE_ACT_QUANT_PATCHED:
        return True
    try:
        import sys

        import vllm.model_executor.layers.quantization.utils.int8_utils as _iu
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: int8_utils unavailable (%s)", e)
        return False

    from ..impl.int8.scaled_int8_quant import scaled_int8_quant as _sunrise_quant

    _orig = getattr(_iu, "per_token_quant_int8", None)
    if _orig is None:
        return False
    if getattr(_orig, "_fl_native_int8_moe_act", False):
        _MOE_ACT_QUANT_PATCHED = True
        return True

    def _fl_per_token_quant_int8(x):
        q, s, _azp = _sunrise_quant(x, scale=None, azp=None, symmetric=True)
        return q, s

    _fl_per_token_quant_int8._fl_native_int8_moe_act = True  # type: ignore[attr-defined]
    _fl_per_token_quant_int8._fl_original = _orig  # type: ignore[attr-defined]
    _iu.per_token_quant_int8 = _fl_per_token_quant_int8

    # Rebind modules that already did ``from ...int8_utils import per_token_quant_int8``
    # (notably ``vllm.model_executor.layers.fused_moe.utils``).
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "per_token_quant_int8", None) is _orig:
            _mod.per_token_quant_int8 = _fl_per_token_quant_int8

    _MOE_ACT_QUANT_PATCHED = True
    logger.info(
        "native-int8: patched per_token_quant_int8 -> sunrise "
        "scaled_int8_quant (MoE W8A8 activation quant on sunrise)."
    )
    return True


def _ensure_dynamic_w8a8_quant_config() -> bool:
    """Make sure dynamic per-token MoE builds a W8A8 (not W8A16) quant config.

    ``CompressedTensorsW8A8Int8MoEMethod`` leaves ``w13_input_scale`` /
    ``w2_input_scale`` at ``None`` for dynamic token quant, and upstream reads a
    missing activation scale as W8A16, so ``use_int8_w8a8`` would stay False and
    the experts would run int8xbf16. ``install_fl_w8a8_moe_selector`` fixes that
    for every FL platform; sunrise only has to confirm it actually ran, since
    ``register_oot_ops`` swallows the failure.
    """
    from importlib import import_module

    try:
        scheme_module = import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: W8A8 MoE scheme module unavailable (%s)", e)
        return False

    builder = getattr(scheme_module, "make_int8_moe_quant_config", None)
    if not getattr(builder, "_vllm_fl_dynamic_w8a8_config", False):
        from vllm_fl.quantization.w8a8.moe import install_fl_w8a8_moe_selector

        install_fl_w8a8_moe_selector()
        builder = scheme_module.make_int8_moe_quant_config
        logger.info(
            "native-int8: installed the FL dynamic W8A8 MoE quant-config "
            "builder because register_oot_ops had not done so yet."
        )

    # The FL installer only rebinds the compressed-tensors scheme module. Other
    # importers hold the stock builder by value (notably the online-INT8
    # quantizer), so hand them the fixed one as well.
    import sys

    import vllm.model_executor.layers.fused_moe.oracle.int8 as _oracle

    stock = _oracle.make_int8_moe_quant_config
    if stock is not builder:
        for module in list(sys.modules.values()):
            if module is None:
                continue
            if getattr(module, "make_int8_moe_quant_config", None) is stock:
                module.make_int8_moe_quant_config = builder
    return True


def enable_native_int8() -> None:
    """Enable the vLLM-native compressed-tensors INT8 path on sunrise/ptpu.

    Idempotent; safe to call at import time even for non-INT8 models.
    """
    global _ENABLED
    if _ENABLED:
        return
    ok_kernel = _register_oot_int8_kernel()
    ok_quant = _patch_scaled_int8_quant()
    ok_mm = _patch_triton_scaled_mm()
    ok_moe_act = _patch_moe_per_token_quant_int8()
    ok_moe = _patch_int8_moe_oracle()
    _ENABLED = ok_kernel and ok_quant and ok_mm and ok_moe_act and ok_moe


def install_late_patches() -> None:
    """Re-assert the INT8 routing that ``register_oot_ops`` overwrites.

    ``apply_sunrise_patches`` runs inside ``register_oot_ops``, after it has
    installed the generic FL W8A8 selector, and possibly after
    ``add_oot_quant_kernel`` has cloned the CUDA linear kernels into the OOT
    slot. Re-running the order-sensitive patches here makes sunrise the last
    writer regardless of how the engine reached this point.
    """
    _ensure_dynamic_w8a8_quant_config()
    _register_oot_int8_kernel()
    _patch_int8_moe_oracle()


__all__ = ["enable_native_int8", "install_late_patches"]

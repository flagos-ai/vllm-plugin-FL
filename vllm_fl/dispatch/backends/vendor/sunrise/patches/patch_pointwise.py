# Copyright (c) 2026 BAAI. All rights reserved.

"""Restore FlagGems PointwiseDynamicFunction fast path on PTPU."""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_patch() -> bool:
    """Idempotently restore ``PointwiseDynamicFunction._call_real_impl``.

    Returns ``True`` on the first successful restore, ``False`` if already
    applied or if FlagGems / pointwise_dynamic are not importable (non-PTPU
    or pre-5.4 FlagGems, in which case the patch is a silent no-op).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return False

    try:
        from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction
    except Exception as exc:  # pragma: no cover - non-PTPU env
        _log.debug(
            "Skipping pointwise_dynamic restore (FlagGems not importable): %s",
            exc,
        )
        return False

    if getattr(PointwiseDynamicFunction, "_sunrise_complex_patch_restored", False):
        return False

    # Only restore if the sunrise patch is actually present. If we are on a
    # pre-5.4 FlagGems (no patch installed) this attribute is False and we
    # leave the original method alone.
    if not getattr(PointwiseDynamicFunction, "_sunrise_complex_patched", False):
        _log.debug(
            "PointwiseDynamicFunction._call_real_impl is not sunrise-patched "
            "(likely pre-5.4 FlagGems); no restore needed."
        )
        _PATCH_APPLIED = True
        return False

    def _call_real_impl_fast_path(
        self, *args, _skip_tensor_check=False, **kwargs
    ):
        ndim, args, kwargs = self.prepare_args(
            *args, _skip_tensor_check=_skip_tensor_check, **kwargs
        )
        overload = self.instantiate(ndim)
        out = overload(*args, **kwargs)
        return self._unwrap(out)

    _call_real_impl_fast_path.__module__ = __name__
    _call_real_impl_fast_path.__qualname__ = (
        "PointwiseDynamicFunction._call_real_impl"
    )

    PointwiseDynamicFunction._call_real_impl = _call_real_impl_fast_path
    PointwiseDynamicFunction._sunrise_complex_patch_restored = True
    _PATCH_APPLIED = True

    _log.info(
        "Sunrise vendor: restored PointwiseDynamicFunction._call_real_impl "
        "fast-path (skip the fp64-fallback check that adds ~4.3 µs / "
        "pointwise op on Qwen3.5 bf16 hot path; saves ~2-4 ms / step on "
        "Qwen3.5-35B-A3B). See sunrise/patches/patch_pointwise.py docstring."
    )
    return True

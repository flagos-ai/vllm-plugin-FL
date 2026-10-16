"""FL MetaX boot shim - safe UnquantizedFusedMoEMethod.is_monolithic.

Under VLLM_PLUGINS=metax the FL MoE factory may not replace the base method.
Accessing is_monolithic then raises AttributeError inside the property
(via nn.Module.__getattr__ remapping moe_kernel/experts_cls misses).

Kill switch: FL_METAX_MOE_IS_MONOLITHIC_PATCH=0
"""
from __future__ import annotations

import importlib.abc as _abc
import os as _os
import sys as _sys

_TARGETS = (
    "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
    "vllm.model_executor.layers.fused_moe.fused_moe_method_base",
)
_MARK = "_fl_metax_moe_is_monolithic_patched"


def _enabled():
    return _os.environ.get("FL_METAX_MOE_IS_MONOLITHIC_PATCH", "") != "0"


def _is_metax():
    # Prefer env (VLLM_PLUGINS=metax / GEMS_VENDOR) 鈥?current_platform may
    # report as cuda without is_metax() during early import.
    env = (_os.environ.get("VLLM_PLUGINS", "") + "," + _os.environ.get("GEMS_VENDOR", "")).lower()
    if "metax" in env or "maca" in env:
        return True
    try:
        from vllm.platforms import current_platform as plat

        if hasattr(plat, "is_metax") and plat.is_metax():
            return True
    except Exception:
        pass
    try:
        import torch

        return "maca" in str(getattr(torch.version, "hip", "") or "").lower() or hasattr(
            torch, "maca"
        )
    except Exception:
        return False


def _safe_is_monolithic(self) -> bool:
    # IMPORTANT: returning True with moe_kernel=None routes into
    # apply_monolithic() which asserts moe_kernel is not None (r79 FAIL).
    # Default False 鈫?modular path when kernel/experts are absent.
    mk = getattr(self, "moe_kernel", None)
    if mk is not None:
        v = getattr(mk, "is_monolithic", False)
        return bool(v() if callable(v) else v)
    ec = getattr(self, "experts_cls", None)
    if ec is None:
        return False
    try:
        v = getattr(ec, "is_monolithic", False)
        return bool(v() if callable(v) else v)
    except Exception:
        return False


def _patch_select(mod):
    """If experts_cls resolution fails, fall back to TritonExperts."""
    orig = getattr(mod, "select_unquantized_moe_backend", None)
    if orig is None or getattr(mod, "_fl_metax_select_backend_patched", False):
        return
    try:
        from vllm.model_executor.layers.fused_moe.experts.triton_experts import (
            TritonExperts,
        )
    except Exception:
        TritonExperts = None  # type: ignore

    def _select(*args, **kwargs):
        backend, cls = orig(*args, **kwargs)
        if cls is None and TritonExperts is not None:
            try:
                from vllm_metax.utils.fused_moe import get_triton_experts_cls

                cls2 = get_triton_experts_cls()
                if cls2 is not None:
                    return (backend, cls2)
            except Exception:
                pass
            return (backend, TritonExperts)
        return (backend, cls)

    mod.select_unquantized_moe_backend = _select
    setattr(mod, "_fl_metax_select_backend_patched", True)


def _patch(mod):
    if getattr(mod, _MARK, False):
        return False, "already patched"
    # Always rebind safe property when kill-switch allows. Safe getattr is
    # correct on all platforms; env/metax check only gates select() fallback.
    patched = []
    for name in ("UnquantizedFusedMoEMethod", "FusedMoEMethodBase"):
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            setattr(cls, "is_monolithic", property(_safe_is_monolithic))
            patched.append(name)
        except Exception as exc:
            patched.append("%s:err:%s" % (name, exc))

    if _is_metax():
        try:
            _patch_select(mod)
            patched.append("select")
        except Exception as exc:
            patched.append("select:err:%s" % (exc,))

    setattr(mod, _MARK, True)
    return True, "patched " + ",".join(patched) if patched else "no class"


class _Loader(_abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        cm = getattr(self._inner, "create_module", None)
        if cm is not None:
            return cm(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch(module)
        finally:
            _detach()


class _Finder(_abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS or not _enabled():
            return None
        spec = None
        for finder in list(_sys.meta_path):
            if finder is self:
                continue
            if finder.__class__.__name__ == "_Finder":
                continue
            fn = getattr(finder, "find_spec", None)
            if fn is None:
                continue
            try:
                spec = fn(fullname, path, target)
            except Exception:
                continue
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


_FINDER = _Finder()


def _detach():
    try:
        _sys.meta_path.remove(_FINDER)
    except ValueError:
        pass


def _attach():
    if not _enabled():
        return
    if _FINDER not in _sys.meta_path:
        _sys.meta_path.insert(0, _FINDER)
    for name in _TARGETS:
        mod = _sys.modules.get(name)
        if mod is not None:
            try:
                _patch(mod)
            except Exception:
                pass


_attach()

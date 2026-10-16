"""FL MetaX boot shim - skip triton topk/topp (SSA dominate on MetaX TTGIR).

E7_eager_r18 still failed after AST rewrite of topk_topp_triton.py:
  loc(_topk_topp_triton.py:208:23) operand #0 does not dominate this use
Do not keep rewriting that kernel.  Route apply_top_k_top_p to the PyTorch
impl when current_platform.is_metax().  Upstream files are NEVER edited.

Kill switch: FL_METAX_SAMPLER_PATCH=0
"""
from __future__ import annotations

import importlib.abc as _abc
import os as _os
import sys as _sys

_TARGET = "vllm.v1.sample.ops.topk_topp_sampler"
_MARK = "_fl_metax_sampler_patched"


def _enabled():
    return _os.environ.get("FL_METAX_SAMPLER_PATCH", "") != "0"


def _is_metax(plat):
    fn = getattr(plat, "is_metax", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    for _s in (type(plat).__module__ or "", type(plat).__name__ or "",
               str(getattr(plat, "device_name", "") or ""),
               _os.environ.get("GEMS_VENDOR", ""),
               _os.environ.get("VLLM_PLUGINS", "")):
        if "metax" in _s.lower() or "maca" in _s.lower():
            return True
    return False


def _patch(mod):
    if getattr(mod, _MARK, False):
        return False, "already patched"
    orig = getattr(mod, "apply_top_k_top_p", None)
    pytorch = getattr(mod, "apply_top_k_top_p_pytorch", None)
    if orig is None or pytorch is None:
        return False, "missing apply_top_k_top_p"
    plat = None
    try:
        from vllm.platforms import current_platform as plat
    except Exception as exc:
        return False, "no current_platform: %s" % (exc,)

    def apply_top_k_top_p(logits, k, p):
        if _is_metax(plat):
            return pytorch(logits, k, p)
        return orig(logits, k, p)

    mod.apply_top_k_top_p = apply_top_k_top_p
    if _is_metax(plat):
        if hasattr(mod, "apply_top_k_top_p_triton"):
            mod.apply_top_k_top_p_triton = pytorch
    setattr(mod, _MARK, True)
    return True, "metax -> apply_top_k_top_p_pytorch"


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
        if fullname != _TARGET or not _enabled():
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
                spec = None
            if spec is not None:
                break
        if spec is None or getattr(spec, "loader", None) is None:
            return None
        spec.loader = _Loader(spec.loader)
        _detach()
        return spec


def _detach():
    try:
        for f in list(_sys.meta_path):
            if isinstance(f, _Finder):
                _sys.meta_path.remove(f)
    except Exception:
        pass


def install():
    if not _enabled():
        return "disabled"
    mod = _sys.modules.get(_TARGET)
    if mod is not None:
        ok, detail = _patch(mod)
        return detail if ok else ("skip: " + detail)
    for f in _sys.meta_path:
        if isinstance(f, _Finder):
            return "armed (already)"
    _sys.meta_path.insert(0, _Finder())
    return "armed"


def status():
    mod = _sys.modules.get(_TARGET)
    return {
        "enabled": _enabled(),
        "armed": any(isinstance(f, _Finder) for f in _sys.meta_path),
        "patched": bool(getattr(mod, _MARK, False)) if mod is not None else False,
    }


INSTALL_RESULT = install()

# SPDX-License-Identifier: Apache-2.0
"""FlagScale-Agent MetaX boot shim: restore SparseMLAAttentionImpl (vllm 0.28).

vllm 0.28 dropped ``SparseMLAAttentionImpl`` from ``vllm.v1.attention.backend``.
vllm_metax 0.20.0 still imports the OLD name from the OLD module and uses it as
a base class, so a thin subclass of ``MLAAttentionImpl`` that still supports
``Generic[T]`` subscription is enough to keep the old import path working.

Injection is double-tracked, because hooking a SUBMODULE alone proved
unreliable (the module can be pulled in indirectly while ``vllm`` itself is
still initialising):

  track 1  top-level ``vllm`` is wrapped; once ``vllm/__init__.py`` has run we
           ensure the symbol exists (importing the backend module if needed).
  track 2  ``vllm.v1.attention.backend`` is wrapped too.

Only injects when the symbol is genuinely absent.  Never touches vllm or
vllm_metax on disk.  Installed from the FL plugin entry point
(vllm_fl.register); the hooks only patch modules that vllm imports lazily.

Kill switch: FL_METAX_SPARSE_MLA_COMPAT=0
"""

import importlib.util
import os
import sys

_ROOT = "vllm"
_TARGET = "vllm.v1.attention.backend"
_SYMBOL = "SparseMLAAttentionImpl"

_STATE = {"calls": 0, "patched": False, "detail": "init", "seen": []}


def _enabled():
    v = os.environ.get("FL_METAX_SPARSE_MLA_COMPAT", "1")
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def _applies():
    from ._fl_metax_gate import metax_active

    return metax_active()


def _make_class(mod):
    base = getattr(mod, "MLAAttentionImpl", None)
    if base is None:
        return None, "no MLAAttentionImpl in %s" % _TARGET
    tvar = getattr(mod, "T", None)
    if tvar is None:
        from typing import TypeVar

        tvar = TypeVar("T", bound=getattr(mod, "AttentionMetadata", object))
    from typing import Generic

    import types as _types
    cls = _types.new_class(_SYMBOL, (base, Generic[tvar]), {}, lambda ns: ns.update({"__module__": _TARGET}))
    return cls, None


def patch_module(mod):
    if getattr(mod, "__name__", "") != _TARGET:
        return False
    if not _enabled() or not _applies():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    if hasattr(mod, _SYMBOL):
        _STATE["detail"] = "already present"
        return True
    cls, err = _make_class(mod)
    if cls is None:
        _STATE["detail"] = err
        return False
    setattr(mod, _SYMBOL, cls)
    _STATE["patched"] = True
    _STATE["detail"] = "injected %s.%s" % (_TARGET, _SYMBOL)
    return True


def _ensure_patched():
    mod = sys.modules.get(_TARGET)
    if mod is None:
        try:
            mod = importlib.import_module(_TARGET)
        except Exception as exc:
            _STATE["detail"] = "warm import failed: %r" % (exc,)
            return False
    return patch_module(mod)


class _Loader:
    def __init__(self, inner, fullname):
        self._inner = inner
        self._fullname = fullname

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        if self._fullname == _ROOT:
            _ensure_patched()
        else:
            patch_module(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (_ROOT, _TARGET):
            return None
        if not _enabled() or not _applies():
            return None
        _STATE["calls"] += 1
        _STATE["seen"].append(fullname)
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:
                continue
            if spec is None or spec.loader is None:
                continue
            if not hasattr(spec.loader, "exec_module"):
                continue
            spec.loader = _Loader(spec.loader, fullname)
            if fullname == _TARGET:
                _detach()
            return spec
        _STATE["detail"] = "no spec for %s (path=%r)" % (fullname, path)
        return None


def _detach():
    for f in list(sys.meta_path):
        if isinstance(f, _Finder):
            try:
                sys.meta_path.remove(f)
            except ValueError:
                pass


def install():
    if not _enabled() or not _applies():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    if _TARGET in sys.modules:
        _ensure_patched()
        _STATE["detail"] = "eager(%s)" % _STATE["detail"]
        return True
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    _STATE["detail"] = "armed for %s + %s" % (_ROOT, _TARGET)
    return True


def status():
    return dict(_STATE)


install()

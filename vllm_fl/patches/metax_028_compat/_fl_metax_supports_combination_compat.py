# SPDX-License-Identifier: Apache-2.0
"""FlagScale-Agent MetaX boot shim: absorb supports_combination arg drift.

vllm 0.28 added the ``use_mm_prefix`` parameter to
``AttentionBackend.supports_combination``; vllm_metax 0.20.0 still declares the
old 8-parameter form, so the 0.28 call site raises
``TypeError: MacaFlashAttentionBackend.supports_combination() takes 9
positional arguments but 10 were given``.

A meta_path finder lets any ``vllm_metax.v1.attention.backends*`` module finish
its normal import, then wraps the ``supports_combination`` classmethods declared
there so parameters unknown to the metax implementation are dropped BY NAME
before forwarding the call.  Drift in either direction is absorbed the same way.
Nothing on disk is modified.

Kill switch: FL_METAX_SUPPORTS_COMBINATION_COMPAT=0
"""

import inspect
import os
import sys

_PREFIX = "vllm_metax.v1.attention.backends"
_STATE = {"calls": 0, "patched": [], "detail": "init"}


def _enabled():
    v = os.environ.get("FL_METAX_SUPPORTS_COMBINATION_COMPAT", "1")
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def _applies():
    from ._fl_metax_gate import metax_active

    return metax_active()


def _param_names(func):
    try:
        sig = inspect.signature(func)
    except Exception:
        return []
    return [p for p in sig.parameters if p not in ("cls", "self")]


def _raw(fn):
    return fn.__func__ if isinstance(fn, classmethod) else fn


def _patch_class(cls):
    fn = cls.__dict__.get("supports_combination")
    if fn is None:
        return False
    func = _raw(fn)
    own = _param_names(func)
    best = []
    for klass in cls.__mro__:
        cand = klass.__dict__.get("supports_combination")
        if cand is None:
            continue
        names = _param_names(_raw(cand))
        if len(names) > len(best):
            best = names
    missing = [n for n in best if n not in own]
    if not missing:
        return False

    def _wrapper(c, *args, **kwargs):
        if len(args) >= len(best):
            kept = [a for n, a in zip(best, args) if n not in missing]
            extra = args[len(best):]
        else:
            kept = list(args)
            extra = []
        kwargs = {k: v for k, v in kwargs.items() if k in own}
        return func(c, *kept, *extra, **kwargs)

    _wrapper.__name__ = "supports_combination"
    setattr(cls, "supports_combination", classmethod(_wrapper))
    return True


def patch_module(mod):
    if not _enabled() or not _applies():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    changed = []
    for name, obj in list(vars(mod).items()):
        if not isinstance(obj, type):
            continue
        if "supports_combination" not in obj.__dict__:
            continue
        try:
            if _patch_class(obj):
                changed.append(name)
        except Exception as exc:
            _STATE["detail"] = "patch failed for %s: %r" % (name, exc)
    if changed:
        _STATE["patched"].extend(changed)
        _STATE["detail"] = "patched %s" % ",".join(changed)
    return bool(changed)


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        patch_module(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        if not _enabled() or not _applies():
            return None
        _STATE["calls"] += 1
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
            spec.loader = _Loader(spec.loader)
            return spec
        return None


def install():
    if not _enabled() or not _applies():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    _STATE["detail"] = "armed for %s.*" % _PREFIX
    return True


def status():
    return dict(_STATE)


install()

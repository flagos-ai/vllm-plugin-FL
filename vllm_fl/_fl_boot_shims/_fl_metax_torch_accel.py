# SPDX-License-Identifier: Apache-2.0
"""MetaX boot shim: torch.accelerator APIs missing on torch 2.8.

vLLM 0.28 gpu_worker.py uses torch.accelerator.empty_cache / MemorySnapshot
which needs memory_stats. MetaX torch 2.8.0+metax3.7.0.7 has the module
but not those attributes. Proxy to torch.cuda.* (MACA).

Kill switch: FL_METAX_TORCH_ACCEL_PATCH=0.
"""
import importlib.machinery
import importlib.util
import os
import sys

_TARGET = "torch.accelerator"
_HOOK = None
_STATE = {"armed": False, "patched": [], "detail": "init"}

# dest attr on torch.accelerator -> source attr on torch.cuda
_ALIASES = (
    ("empty_cache", "empty_cache"),
    ("memory_stats", "memory_stats"),
    ("get_memory_info", "mem_get_info"),
    ("memory_reserved", "memory_reserved"),
    ("memory_allocated", "memory_allocated"),
    ("max_memory_allocated", "max_memory_allocated"),
    ("max_memory_reserved", "max_memory_reserved"),
    ("reset_peak_memory_stats", "reset_peak_memory_stats"),
    ("reset_max_memory_allocated", "reset_max_memory_allocated"),
    ("mem_get_info", "mem_get_info"),
)


def _noop(*_a, **_k):
    return None


def _enabled():
    v = os.environ.get("FL_METAX_TORCH_ACCEL_PATCH", "1").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _metax_active():
    vp = os.environ.get("VLLM_PLUGINS")
    if vp is not None and vp.strip() and "metax" not in vp:
        return False
    try:
        return importlib.util.find_spec("vllm_metax") is not None
    except Exception:
        return False


def _cuda_proxy(src):
    def _fn(*a, **k):
        import torch

        fn = getattr(torch.cuda, src, None)
        if not callable(fn):
            raise AttributeError("torch.cuda.%s" % src)
        return fn(*a, **k)

    _fn.__name__ = src
    return _fn


def _patch(mod):
    added = []
    for dst, src in _ALIASES:
        if not hasattr(mod, dst):
            setattr(mod, dst, _cuda_proxy(src))
            added.append(dst)
    if not hasattr(mod, "empty_host_cache"):
        setattr(mod, "empty_host_cache", _noop)
        added.append("empty_host_cache")
    if added:
        _STATE["patched"] = list(_STATE.get("patched") or []) + added
        _STATE["detail"] = "added " + ",".join(added)
    else:
        _STATE["detail"] = "all attrs present"
    return bool(added)


def _detach():
    global _HOOK
    if _HOOK is None:
        return
    try:
        sys.meta_path.remove(_HOOK)
    except ValueError:
        pass
    _HOOK = None


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module)
        _detach()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        if not _enabled() or not _metax_active():
            return None
        real = None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                real = find(fullname, path, target)
            except Exception:
                real = None
            if real is not None:
                break
        if real is None:
            try:
                real = importlib.machinery.PathFinder.find_spec(fullname, path)
            except Exception:
                return None
        if real is None or real.loader is None:
            return None
        real.loader = _Loader(real.loader)
        return real


def install():
    global _HOOK
    if not _enabled() or not _metax_active():
        _STATE["detail"] = "skipped"
        return False
    acc = None
    t = sys.modules.get("torch")
    if t is not None:
        acc = getattr(t, "accelerator", None)
    if acc is None:
        acc = sys.modules.get(_TARGET)
    if acc is not None:
        _patch(acc)
        return True
    if _HOOK is not None:
        return False
    _HOOK = _Finder()
    sys.meta_path.insert(0, _HOOK)
    _STATE["armed"] = True
    _STATE["detail"] = "finder armed"
    return True


def status():
    return dict(_STATE)


install()

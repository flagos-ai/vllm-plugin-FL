# SPDX-License-Identifier: Apache-2.0
"""MetaX boot shim: vllm.v1.worker.gpu._load_ptr needs a constexpr unwrap.

Triton 3.0 (MetaX bundle) wraps a jit-helper dtype argument into tl.constexpr,
so `tl.pointer_type(elem_dtype)` raises
TypeError('element_ty is a constexpr.'); upstream triton 3.6 tolerated it.
Unwrap with `.value` before building the pointer type.

Probe evidence on triton 3.0.0 (probe V1/V2/V3):
  inline tl.pointer_type(tl.int32)          -> OK
  helper(elem_dtype: tl.constexpr) + .value -> OK
  helper(elem_dtype) no annot + .value      -> OK
  helper + tl.pointer_type(elem_dtype)      -> ERR (both annot forms)

Kill switch: FL_METAX_LOAD_PTR_PATCH=0.
"""
import importlib.machinery
import importlib.util
import os
import sys

_TARGETS = (
    "vllm.v1.worker.gpu.buffer_utils",
    "vllm.v1.worker.gpu.block_table",
)
_HOOK = None
_STATE = {"armed": False, "patched": [], "detail": "init",
          "targets": list(_TARGETS)}


def _enabled():
    v = os.environ.get("FL_METAX_LOAD_PTR_PATCH", "1").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _metax_active():
    vp = os.environ.get("VLLM_PLUGINS")
    if vp is not None and vp.strip() and "metax" not in vp:
        return False
    try:
        return importlib.util.find_spec("vllm_metax") is not None
    except Exception:
        return False


def _make_impl():
    import triton
    import triton.language as tl

    @triton.jit
    def _load_ptr(ptr_to_ptr, elem_dtype: tl.constexpr):
        ptr = tl.load(ptr_to_ptr)
        ptr = tl.cast(ptr, tl.pointer_type(elem_dtype.value))
        return tl.multiple_of(ptr, 16)

    return _load_ptr


def _patch(mod):
    name = getattr(mod, "__name__", "?")
    if getattr(mod, "_load_ptr", None) is None:
        return False
    try:
        impl = _make_impl()
    except Exception as e:
        _STATE["detail"] = "impl failed: %r" % (e,)
        return False
    mod._load_ptr = impl
    _STATE["patched"] = list(_STATE.get("patched") or []) + [name]
    _STATE["detail"] = "patched " + name
    return True


def _detach():
    global _HOOK
    if _HOOK is None:
        return
    try:
        sys.meta_path.remove(_HOOK)
    except ValueError:
        pass
    _HOOK = None
    _STATE["armed"] = False


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
        done = set(_STATE.get("patched") or [])
        if all(t in done for t in _TARGETS):
            _detach()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
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
    hit = 0
    for name in _TARGETS:
        mod = sys.modules.get(name)
        if mod is not None and _patch(mod):
            hit += 1
    if hit == len(_TARGETS):
        _STATE["detail"] = "patched already-imported"
        return True
    if _HOOK is not None:
        return False
    _HOOK = _Finder()
    sys.meta_path.insert(0, _HOOK)
    _STATE["armed"] = True
    if _STATE.get("detail") == "init":
        _STATE["detail"] = "finder armed"
    return True


def status():
    return dict(_STATE)


install()


# === FL r12d module-level tl for jit globals ===
# triton 3.0 resolves annotation names from the jit function's __globals__;
# _make_impl imported tl as a LOCAL, so the subprocess rebuild of _load_ptr
# raised NameError('tl is not defined'). Expose it at module level.
try:
    import triton as _fl_triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception as _e:
    print("R12D_TL_IMPORT_FAIL %r" % (_e,))
else:
    print("R12D_TL_GLOBAL_OK")

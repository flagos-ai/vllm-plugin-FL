"""FlagScale-Agent MetaX boot shim: register triton.experimental stubs early.

The MetaX Triton 3.0.0 runtime ships no triton.experimental / gluon subpackage,
and its triton.language.core has no `_aggregate`, while vllm 0.28 imports both
unconditionally (vllm/triton_utils/__init__.py lines 14-17) when HAS_TRITON.
Registering stub modules in sys.modules (experimental) plus a minimal
`_aggregate` placeholder on triton.language.core keeps that import working,
without touching upstream vllm.

Only the IMPORT path is covered: on MACA the consumers of `aggregate` are the
AMD/ROCm gluon kernels (vllm/models/inkling/amd/ops/gluon/*), which never run here.

Loaded from a site-packages .pth at interpreter startup.
Stdlib only: must not import vllm, torch or triton.
"""
import os
import sys
import types

_STUB_MODULE = "_fl_triton_experimental_stub"


def _env_enabled():
    forced = os.environ.get("FL_TRITON_EXPERIMENTAL_STUB", "").strip().lower()
    if forced in ("0", "false", "off", "no"):
        return False
    if forced in ("1", "true", "on", "yes", "metax"):
        return True
    if os.environ.get("GEMS_VENDOR", "").strip().lower() == "metax":
        return True
    if "metax" in os.environ.get("VLLM_PLUGINS", "").strip().lower():
        return True
    return False


def _triton_pkg_dir():
    import importlib.util
    try:
        spec = importlib.util.find_spec("triton")
    except Exception:
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def _vendor_artifacts():
    pkg = _triton_pkg_dir()
    if not pkg:
        return False
    sp = os.path.dirname(pkg)
    for name in ("mcoplib", "maca", "vllm_metax"):
        if os.path.isdir(os.path.join(sp, name)):
            return True
    return False


def _needs_stub():
    pkg = _triton_pkg_dir()
    if not pkg:
        return False
    return not os.path.isdir(os.path.join(pkg, "experimental"))


def _placeholder(name):
    mod = types.ModuleType(name)
    mod.__doc__ = name + " placeholder installed by " + _STUB_MODULE + "."
    mod.__fl_triton_stub__ = True

    def _missing(attr, _name=name):
        raise AttributeError(
            _name + "." + attr + " is unavailable: this Triton runtime has no "
            "experimental/gluon subpackage and " + _STUB_MODULE
            + " installed a placeholder.  gluon kernels cannot run on this backend."
        )

    mod.__getattr__ = _missing
    return mod


# --------------------------------------------------------------------------
# triton.language.core._aggregate minimal placeholder
# --------------------------------------------------------------------------
# vllm/triton_utils/__init__.py:17 does `from triton.language.core import
# _aggregate as aggregate` under `if TYPE_CHECKING or HAS_TRITON`.  Triton 3.0.0
# on MACA does not define it.  The real implementation (triton 3.5.1) builds a
# wrapper value class on `base_value` and `ir`, neither of which exists in 3.0.0,
# so copying it is not viable.  Provide the smallest object that satisfies the
# documented usage shape: a class decorator usable as `@aggregate`, on a class
# that is instantiated normally and whose methods are referenced as `Cls[...]`
# / `Cls.method`.
_AGG_PATCHED = False
_AGG_HOOK = None


def _core_py_path():
    pkg = _triton_pkg_dir()
    if not pkg:
        return None
    p = os.path.join(pkg, "language", "core.py")
    return p if os.path.isfile(p) else None


def _needs_aggregate():
    """True when triton.language.core does not define _aggregate."""
    mod = sys.modules.get("triton.language.core")
    if mod is not None and hasattr(mod, "_aggregate"):
        return False
    p = _core_py_path()
    if not p:
        return False
    try:
        with open(p, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("def _aggregate"):
                    return False
    except Exception:
        return False
    return True


def _make_aggregate():
    def _aggregate(cls):
        """Minimal stand-in for triton.language.core._aggregate (import-only)."""
        if not isinstance(cls, type):
            return cls
        for attr, value in (
            ("__triton_builtin__", True),
            ("__triton_aggregate__", True),
        ):
            if not hasattr(cls, attr):
                try:
                    setattr(cls, attr, value)
                except Exception:
                    pass
        if not hasattr(cls, "_get_instance"):
            def _get_instance(this_cls):
                return object.__new__(this_cls)
            try:
                cls._get_instance = classmethod(_get_instance)
            except Exception:
                pass
        if not hasattr(cls, "__class_getitem__"):
            def __class_getitem__(this_cls, item):
                return this_cls
            try:
                cls.__class_getitem__ = classmethod(__class_getitem__)
            except Exception:
                pass
        return cls

    _aggregate.__name__ = "_aggregate"
    return _aggregate


def _patch_core(mod):
    global _AGG_PATCHED
    if mod is None:
        return False
    if hasattr(mod, "_aggregate"):
        _AGG_PATCHED = True
        return False
    try:
        setattr(mod, "_aggregate", _make_aggregate())
    except Exception:
        return False
    _AGG_PATCHED = True
    return True


def _detach_finder():
    for f in list(sys.meta_path):
        if isinstance(f, _AggregateFinder):
            try:
                sys.meta_path.remove(f)
            except ValueError:
                pass


class _AggregateLoader:
    """Delegating loader: runs the real exec_module, then patches the module."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch_core(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _AggregateFinder:
    """meta_path hook: wrap the loader of triton.language.core."""

    _TARGET = "triton.language.core"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._TARGET:
            return None
        spec = None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None:
            return None
        loader = getattr(spec, "loader", None)
        if loader is None or not hasattr(loader, "exec_module"):
            return None
        spec.loader = _AggregateLoader(loader)
        _detach_finder()
        return spec


def _install_aggregate():
    global _AGG_HOOK
    if not _needs_aggregate():
        return False
    mod = sys.modules.get("triton.language.core")
    if mod is not None:
        return _patch_core(mod)
    if _AGG_HOOK is not None:
        return False
    _AGG_HOOK = _AggregateFinder()
    sys.meta_path.insert(0, _AGG_HOOK)
    return True


def install():
    forced = os.environ.get("FL_TRITON_EXPERIMENTAL_STUB", "").strip().lower()
    if forced in ("0", "false", "off", "no"):
        return False
    if not (_env_enabled() or _vendor_artifacts()):
        return False

    did = False

    if "triton.experimental" not in sys.modules and _needs_stub():
        exp = _placeholder("triton.experimental")
        gluon = _placeholder("triton.experimental.gluon")
        lang = _placeholder("triton.experimental.gluon.language")
        exp.gluon = gluon
        gluon.language = lang
        sys.modules["triton.experimental"] = exp
        sys.modules["triton.experimental.gluon"] = gluon
        sys.modules["triton.experimental.gluon.language"] = lang
        triton = sys.modules.get("triton")
        if triton is not None:
            try:
                triton.experimental = exp
            except Exception:
                pass
        did = True

    if _install_aggregate():
        did = True

    return did


def status():
    return {
        "installed": _INSTALLED,
        "env_enabled": _env_enabled(),
        "vendor_artifacts": _vendor_artifacts(),
        "path_enabled": "triton.experimental" in sys.modules,
        "aggregate_patched": _AGG_PATCHED,
        "aggregate_hook": _AGG_HOOK is not None,
    }


_INSTALLED = install()

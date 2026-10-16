"""FlagScale-Agent MetaX boot shim: restore AttentionBackendEnum.TREE_ATTN.

vllm_metax 0.20.0 (site-packages) assumes the vllm 0.20 attention backend enum,
which has TREE_ATTN. vllm 0.28 dropped that member, so
vllm_metax.platform.register_attention_backends() raises AttributeError while
building the engine config. This shim re-adds the missing member in-process
before vllm_metax touches it; gated on the metax backend. It never modifies
vllm or vllm_metax on disk.

Loaded from a site-packages .pth at interpreter start (same mechanism as
_fl_triton_experimental_stub), so it must not import vllm_fl or vllm.
"""
import importlib.util
import os
import sys

_TARGET = "vllm.v1.attention.backends.registry"
_ENUM = "AttentionBackendEnum"
_MEMBER = "TREE_ATTN"
_VALUE = "vllm_metax.v1.attention.backends.tree_attn.MacaTreeAttentionBackend"

_STATE = {"patched": False, "detail": "not installed"}


def _applies():
    plugins = os.environ.get("VLLM_PLUGINS")
    if plugins:
        names = {p.strip() for p in plugins.split(",") if p.strip()}
        if "metax" not in names:
            return False
    try:
        return importlib.util.find_spec("vllm_metax") is not None
    except Exception:
        return False


def _add_member(enum_cls, name, value):
    if name in enum_cls.__members__:
        return enum_cls.__members__[name]
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member._sort_order_ = len(enum_cls._member_names_)
    member.__objclass__ = enum_cls
    enum_cls._member_map_[name] = member
    enum_cls._value2member_map_[value] = member
    try:
        type.__setattr__(enum_cls, name, member)
    except Exception:
        pass
    return member


def patch_module(mod):
    if getattr(mod, "__name__", "") != _TARGET:
        return False
    if not _applies():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    enum_cls = getattr(mod, _ENUM, None)
    if enum_cls is None:
        _STATE["detail"] = "no %s in module" % _ENUM
        return False
    try:
        _add_member(enum_cls, _MEMBER, _VALUE)
    except Exception as exc:
        _STATE["detail"] = "inject failed: %r" % (exc,)
        return False
    _STATE["patched"] = True
    _STATE["detail"] = "injected %s.%s" % (_ENUM, _MEMBER)
    return True


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        patch_module(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
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
            _detach()
            return spec
        return None


def _detach():
    for f in list(sys.meta_path):
        if isinstance(f, _Finder):
            try:
                sys.meta_path.remove(f)
            except ValueError:
                pass


def install():
    if not _applies():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    mod = sys.modules.get(_TARGET)
    if mod is not None:
        return patch_module(mod)
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    _STATE["detail"] = "finder armed for %s" % _TARGET
    return True


def status():
    return dict(_STATE)


install()

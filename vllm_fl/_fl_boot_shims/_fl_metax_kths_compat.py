"""FL MetaX boot shim - triton 3.0 NameError across constexpr-if.

Root cause (measured 2026-09-21, mx_plugin_028, triton 3.0.0, E7_eager_r14):
  vllm_metax/.../triton_unified_attention.py  kernel_unified_attention
    first  if USE_PER_TOKEN_HEAD_SCALES:  k/v_token_head_scales = tl.load(...)
    later  if USE_PER_TOKEN_HEAD_SCALES:  uses those names
  Triton 3.0 does not leak names assigned inside one constexpr-if into another,
  even when the test is the same.  Upstream files are NEVER edited.

Fix: copy the defining assignments into every later `if USE_PER_TOKEN_HEAD_SCALES`
that uses the names but does not assign them, so each constexpr-if is self-contained.

Kill switch: FL_METAX_KTHS_PATCH=0
"""
from __future__ import annotations

import ast as _ast
import copy as _copy
import importlib.abc as _abc
import os as _os
import sys as _sys

_TARGET = "vllm_metax.v1.attention.ops.triton_unified_attention"
_MARK = "_fl_metax_kths_patched"
_NAMES = ("k_token_head_scales", "v_token_head_scales")
_GATE = ("k_token_head_scales", "USE_PER_TOKEN_HEAD_SCALES")
_FLAG = "USE_PER_TOKEN_HEAD_SCALES"


def _enabled():
    return _os.environ.get("FL_METAX_KTHS_PATCH", "") != "0"


def _is_flag_if(node):
    t = node.test
    return isinstance(t, _ast.Name) and t.id == _FLAG


def _assign_names(stmt):
    out = set()
    if isinstance(stmt, _ast.Assign):
        for t in stmt.targets:
            if isinstance(t, _ast.Name):
                out.add(t.id)
    return out


class _Hoist(_ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        ifs = [n for n in _ast.walk(node) if isinstance(n, _ast.If) and _is_flag_if(n)]
        def_block = None
        for iff in ifs:
            names = set()
            for stmt in iff.body:
                names |= _assign_names(stmt)
            if "k_token_head_scales" in names:
                def_block = iff
                break
        if def_block is None:
            return node
        defs = [stmt for stmt in def_block.body if _assign_names(stmt)]
        if not defs:
            return node
        for iff in ifs:
            if iff is def_block:
                continue
            body_set = set()
            for stmt in iff.body:
                body_set |= _assign_names(stmt)
            extra = []
            for stmt in defs:
                names = _assign_names(stmt)
                if names - body_set:
                    extra.append(_copy.deepcopy(stmt))
                    body_set |= names
            if extra:
                iff.body[:0] = extra
        return node


def rewrite_src(src: str) -> str:
    tree = _ast.parse(src)
    tree = _Hoist().visit(tree)
    _ast.fix_missing_locations(tree)
    return _ast.unparse(tree)


def _gated(text: str) -> bool:
    return all(g in text for g in _GATE)


class _Loader(_abc.Loader):
    def __init__(self, inner, origin):
        self._inner = inner
        self._origin = origin

    def create_module(self, spec):
        cm = getattr(self._inner, "create_module", None)
        if cm is not None:
            return cm(spec)
        return None

    def exec_module(self, module):
        path = self._origin or getattr(getattr(self._inner, "path", None), "as_posix", lambda: None)()
        if not path:
            path = getattr(module, "__file__", None)
        src = None
        if path:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                src = None
        if src and _gated(src) and _enabled():
            try:
                rew = rewrite_src(src)
                sidecar = _os.path.join(
                    _os.path.dirname(_os.path.abspath(__file__)),
                    "_kths_triton_unified_attention.py",
                )
                # Triton JIT reads inspect.getsource from co_filename; compiling
                # against the original path would still JIT the unpatched file.
                with open(sidecar, "w", encoding="utf-8") as fh:
                    fh.write(rew)
                code = compile(rew, sidecar, "exec")
                module.__file__ = sidecar
                module.__dict__["__file__"] = sidecar
                exec(code, module.__dict__)
                setattr(module, _MARK, True)
                _detach()
                return
            except Exception:
                pass
        self._inner.exec_module(module)
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
        origin = getattr(spec, "origin", None)
        spec.loader = _Loader(spec.loader, origin)
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
    if _sys.modules.get(_TARGET) is not None:
        return "skip: already imported"
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

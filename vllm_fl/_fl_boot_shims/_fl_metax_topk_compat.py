"""FL MetaX boot shim - triton 3.0 SSA dominance in topk/topp kernel.

Root cause (measured 2026-09-21, mx_plugin_028, E7_eager_r17):
  vllm/v1/sample/ops/topk_topp_triton.py
    L376  final_pivot = k_pivot if num_finite_total > k else -float("inf")
    L378  if TOPP_ENABLED and num_finite_total > k:
  MetaX TTGIR: operand #0 does not dominate this use / PassManager::run failed.
  Mixing a constexpr (TOPP_ENABLED) with a runtime compare in one `if`, and a
  Python ternary, produces SSA uses that do not dominate.  Upstream is NEVER edited.

Fix: AST-rewrite then write a sidecar (triton.jit reads inspect.getsource from
co_filename).  Split `if A and B` into nested ifs; expand `x = a if c else b`
into a real if/else.

Kill switch: FL_METAX_TOPK_PATCH=0
"""
from __future__ import annotations

import ast as _ast
import importlib.abc as _abc
import os as _os
import sys as _sys

_TARGET = "vllm.v1.sample.ops.topk_topp_triton"
_MARK = "_fl_metax_topk_patched"
_GATE = ("TOPP_ENABLED", "final_pivot")
_SIDECAR = "_topk_topp_triton.py"


def _enabled():
    return _os.environ.get("FL_METAX_TOPK_PATCH", "") != "0"


class _Rewrite(_ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if not (isinstance(t, _ast.BoolOp) and isinstance(t.op, _ast.And) and len(t.values) >= 2):
            return node
        acc = _ast.If(test=t.values[-1], body=list(node.body), orelse=list(node.orelse))
        for val in reversed(t.values[:-1]):
            acc = _ast.If(test=val, body=[acc], orelse=[])
        return acc

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, _ast.IfExp) and len(node.targets) == 1:
            tgt = node.targets[0]
            return _ast.If(
                test=node.value.test,
                body=[_ast.Assign(targets=[tgt], value=node.value.body)],
                orelse=[_ast.Assign(targets=[tgt], value=node.value.orelse)],
            )
        return node


def rewrite_src(src: str) -> str:
    tree = _ast.parse(src)
    tree = _Rewrite().visit(tree)
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
        path = self._origin or getattr(module, "__file__", None)
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
                    _SIDECAR,
                )
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

"""FL MetaX boot shim - triton 3.0.0 CodeGenerator AnnAssign compatibility.

Root cause (measured 2026-09-21, container mx_plugin_028, triton 3.0.0):
  <site-packages>/triton/compiler/code_generator.py
    L479  def visit_AnnAssign(self, node)
    L493      # default: call visit_Assign
    L494      return self.visit_Assign(node)
  visit_Assign (L496) iterates `node.targets`, which ast.AnnAssign does not
  have -> AttributeError("'AnnAssign' object has no attribute 'targets'")
  -> triton CompilationError pointing at the annotated assignment.

  First in-tree trigger: vllm 0.28
    vllm/v1/attention/ops/triton_attention_helpers.py:94   left: tl.int32 = 0

Fix (upstream triton is NOT edited): degrade the annotated assignment
`x: T = v` to a plain assignment `x = v` (annotation is dropped) before the
broken dispatch happens; a bare annotation `x: T` binds nothing and is a no-op.

Only triton.compiler.code_generator is touched, and only after a CONTENT gate
finds the broken stub in that file's source, so a future triton shipping a real
visit_AnnAssign is left alone.

Kill switch: FL_METAX_ANNASSIGN_PATCH=0
"""
import ast as _ast
import importlib.abc as _abc
import os as _os
import sys as _sys

_TARGET = "triton.compiler.code_generator"
_MARK = "_fl_metax_annassign_patched"
_STUB_MARK = "default: call visit_Assign"


def _enabled():
    return _os.environ.get("FL_METAX_ANNASSIGN_PATCH", "") != "0"


def _is_broken(text):
    return ("def visit_AnnAssign" in text) and (_STUB_MARK in text)


def _as_assign(node):
    """`x: T = v` -> ast.Assign(x = v); `x: T` -> None."""
    if getattr(node, "value", None) is None:
        return None
    new = _ast.Assign(targets=[node.target], value=node.value)
    try:
        _ast.copy_location(new, node)
    except Exception:
        pass
    return new


def _patch(mod):
    if getattr(mod, _MARK, False):
        return False, "already patched"
    gen = getattr(mod, "CodeGenerator", None)
    if gen is None:
        return False, "no CodeGenerator"
    path = getattr(mod, "__file__", None)
    verified = False
    if path:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            text = None
        if text is not None:
            if not _is_broken(text):
                setattr(mod, _MARK, True)
                return False, "not-needed (no broken stub)"
            verified = True
    orig_assign = gen.visit_Assign

    def visit_Assign(self, node):
        if node.__class__.__name__ == "AnnAssign":
            node = _as_assign(node)
            if node is None:
                return False
        return orig_assign(self, node)

    def visit_AnnAssign(self, node):
        conv = _as_assign(node)
        if conv is None:
            return False
        return visit_Assign(self, conv)

    gen.visit_Assign = visit_Assign
    gen.visit_AnnAssign = visit_AnnAssign
    setattr(mod, _MARK, True)
    return True, "patched visit_AnnAssign/visit_Assign (stub verified=%s)" % verified


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
            if finder is self or finder is _sys.meta_path[0] and finder is self:
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
        "armed": any(isinstance(f, _Finder) for f in _sys.meta_path),
        "patched": bool(getattr(mod, _MARK, False)) if mod is not None else False,
    }


INSTALL_RESULT = install()

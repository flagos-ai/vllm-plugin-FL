"""FL MetaX boot shim - triton 3.0.0 CodeGenerator chained-BoolOp compatibility.

Root cause (measured 2026-09-21, container mx_plugin_028, triton 3.0.0):
  /opt/conda/lib/python3.12/site-packages/triton/compiler/code_generator.py
    L1148 def visit_BoolOp(self, node: ast.BoolOp)
    L1149     if len(node.values) != 2:
    L1150         raise self._unsupported(
    L1151             node, "chained boolean operators (A or B or C) are not
                        supported; use parentheses to split the chain.")

  CPython parses `A or B or C` as a SINGLE ast.BoolOp holding THREE values,
  so any chained boolean operator in triton kernel source is rejected.

  Trigger (vllm 0.28 sdist, line 44 of the kernel):
    vllm/v1/attention/ops/triton_attention_helpers.py:185
      if USE_MM_PREFIX or USE_PER_SEQ_CAUSAL or (not USE_CAUSAL):

Fix (upstream triton is NEVER edited): left-fold a multi-value BoolOp into
nested binary BoolOps -- `A or B or C` -> `(A or B) or C`, exactly the form
the error message asks for and the checker accepts.

The sibling AnnAssign shim is re-applied to the same module as well: only one
meta_path finder can win the load of triton.compiler.code_generator, so if
this finder wins, that one must not lose its fix.

Kill switch: FL_METAX_BOOLCHAIN_PATCH=0
"""
import ast as _ast
import importlib.abc as _abc
import os as _os
import sys as _sys

_TARGET = "triton.compiler.code_generator"
_MARK = "_fl_metax_boolchain_patched"
_STUB_MARK = "chained boolean operators"


def _enabled():
    return _os.environ.get("FL_METAX_BOOLCHAIN_PATCH", "") != "0"


def _as_binary(node):
    """Multi-value BoolOp -> left-leaning nested binary BoolOp."""
    vals = list(node.values)
    if len(vals) == 2:
        return node
    acc = vals[0]
    for v in vals[1:]:
        new = _ast.BoolOp(op=node.op, values=[acc, v])
        try:
            _ast.copy_location(new, node)
        except Exception:
            pass
        acc = new
    return acc


def _wrap(orig):
    def visit_BoolOp(self, node):
        if len(node.values) < 2:
            return self.visit(node.values[0])
        return orig(self, _as_binary(node))
    return visit_BoolOp


def _reapply_sibling():
    try:
        import _fl_metax_annassign_compat as sib
        sib.install()
        return sib.status().get("patched")
    except Exception as exc:
        return "n/a (%s)" % (exc,)


def _patch(mod):
    if getattr(mod, _MARK, False):
        return False, "already patched"
    path = getattr(mod, "__file__", None)
    if path:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            text = None
        if text is not None:
            if not (("def visit_BoolOp" in text) and (_STUB_MARK in text)):
                setattr(mod, _MARK, True)
                return False, "not-needed (no chained-BoolOp stub)"
    touched = []
    for name, obj in list(vars(mod).items()):
        if not isinstance(obj, type):
            continue
        orig = obj.__dict__.get("visit_BoolOp")
        if orig is None:
            continue
        obj.visit_BoolOp = _wrap(orig)
        touched.append(name)
    setattr(mod, _MARK, True)
    if not touched:
        return False, "no visit_BoolOp found"
    return True, "patched visit_BoolOp on %s" % (touched,)


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
            _reapply_sibling()
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
            # Sibling boot shims also install a class named _Finder for this
            # same module. Calling them here recurses until RecursionError
            # and both patches are lost. Skip by class name; _reapply_sibling
            # re-runs the AnnAssign patch after we load the module.
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
        _reapply_sibling()
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

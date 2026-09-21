# SPDX-License-Identifier: Apache-2.0
"""FlagScale-Agent MetaX boot shim (v2).

Restores two pre-0.28 vllm APIs that vllm_metax 0.20.0 still relies on:

1) ALIAS ``vllm.model_executor.layers.fused_moe.lora_experts_mixin`` ->
   ``vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin``.
   vllm 0.28 MOVED that module; vllm_metax imports the OLD dotted path and uses
   ``LoRAExpertsMixin`` as a base class, so both names must be the SAME object.

2) PATCH ``vllm.model_executor.layers.fused_moe.utils`` right after it is
   executed: vllm 0.28 DELETED ``disable_inplace()``, while vllm_metax still does
   ``from ...fused_moe.utils import _resize_cache, disable_inplace,
   moe_kernel_quantize_input`` and asserts
   ``assert not inplace or not disable_inplace()`` (fused_moe.py:1929).
   We inject ``lambda: False`` (= inplace allowed, matches 0.28 semantics) and
   ONLY when the attribute is genuinely missing.

This module never imports vllm / vllm_metax / vllm_fl, never edits an upstream
vllm file and never touches site-packages/vllm_metax. Every hook self-detaches
after its first successful hit. Nothing is patched when the real symbol is
already present.

Gate: if VLLM_PLUGINS is set and does not contain "metax" -> skip;
      if vllm_metax is not importable -> skip.
Kill switches: FL_METAX_LORA_COMPAT=0 (alias), FL_METAX_UTILS_PATCH=0 (patch).
"""

import importlib
import importlib.machinery
import importlib.util
import os
import sys

_OLD = "vllm.model_executor.layers.fused_moe.lora_experts_mixin"
_NEW = "vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin"
_PARENT = "vllm.model_executor.layers.fused_moe"
_BASENAME = "lora_experts_mixin.py"
_UTILS = "vllm.model_executor.layers.fused_moe.utils"

# vllm 0.28 deleted disable_inplace(); returning False keeps the 0.28 semantics
# (inplace allowed) so ``assert not inplace or not disable_inplace()`` holds.
_PLACEHOLDERS = {"disable_inplace": (lambda: False)}

_STATE = {
    "armed": False,
    "done": False,
    "detail": "init",
    "patched_utils": None,
    "patched_utils_when": None,
}


def _flag(name, default="1"):
    v = os.environ.get(name, default)
    if v is None:
        return True
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def _env_enabled():
    return _flag("FL_METAX_LORA_COMPAT")


def _patch_enabled():
    return _flag("FL_METAX_UTILS_PATCH")


def _metax_active():
    vp = os.environ.get("VLLM_PLUGINS")
    if vp is not None and vp.strip() and "metax" not in vp:
        return False
    try:
        return importlib.util.find_spec("vllm_metax") is not None
    except Exception:
        return False


def _applies():
    if not _metax_active():
        _STATE["detail"] = "skipped: metax backend not active"
        return False
    return True


def _old_file_exists():
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        return False
    root = os.path.dirname(spec.origin)
    return os.path.isfile(
        os.path.join(root, "model_executor", "layers", "fused_moe", _BASENAME)
    )


def _patch_module(mod):
    """Inject placeholders that are genuinely absent; return the names added."""
    added = []
    for name, fn in _PLACEHOLDERS.items():
        if not hasattr(mod, name):
            setattr(mod, name, fn)
            added.append(name)
    return added


_HOOKS = []


def _detach(hook):
    try:
        sys.meta_path.remove(hook)
    except ValueError:
        pass
    if hook in _HOOKS:
        _HOOKS.remove(hook)


class _AliasLoader:
    def create_module(self, spec):
        return importlib.import_module(_NEW)

    def exec_module(self, module):
        _STATE["done"] = True
        _STATE["detail"] = "aliased %s -> %s" % (_OLD, _NEW)
        for h in list(_HOOKS):
            if getattr(h, "_kind", None) == "alias":
                _detach(h)


class _AliasFinder:
    _kind = "alias"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _OLD:
            return None
        if not _env_enabled() or not _applies():
            return None
        try:
            if _old_file_exists():
                _STATE["detail"] = "skipped: old path exists on disk"
                return None
        except Exception:
            pass
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader())


class _PatchLoader:
    def __init__(self, inner, fullname):
        self._inner = inner
        self._fullname = fullname

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        added = _patch_module(module)
        _STATE["patched_utils"] = added
        _STATE["patched_utils_when"] = "import-hook"
        for h in list(_HOOKS):
            if getattr(h, "_kind", None) == "patch":
                _detach(h)


class _PatchFinder:
    """Wrap the REAL loader of the utils module (PathFinder: no meta_path
    recursion) and setattr placeholders right after exec_module."""

    _kind = "patch"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _UTILS:
            return None
        if not _patch_enabled() or not _applies():
            return None
        try:
            real = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception:
            return None
        if real is None or real.loader is None:
            return None
        try:
            return importlib.util.spec_from_loader(
                fullname, _PatchLoader(real.loader, fullname), origin=real.origin
            )
        except Exception:
            return None


def install():
    if not _applies():
        return False
    armed = []

    if _env_enabled():
        if not any(getattr(f, "_kind", None) == "alias" for f in sys.meta_path):
            h = _AliasFinder()
            sys.meta_path.insert(0, h)
            _HOOKS.append(h)
            armed.append("lora_experts_mixin")

    if _patch_enabled():
        if not any(getattr(f, "_kind", None) == "patch" for f in sys.meta_path):
            h = _PatchFinder()
            sys.meta_path.insert(0, h)
            _HOOKS.append(h)
            armed.append("utils.disable_inplace")
        mod = sys.modules.get(_UTILS)
        if mod is not None:
            _STATE["patched_utils"] = _patch_module(mod)
            _STATE["patched_utils_when"] = "eager(sys.modules)"

    _STATE["armed"] = bool(armed)
    if armed:
        _STATE["detail"] = "finder armed for %s" % ",".join(armed)
    elif _STATE["detail"] == "init":
        _STATE["detail"] = "nothing armed (kill switch?)"
    return bool(armed)


def status():
    return dict(_STATE)


install()

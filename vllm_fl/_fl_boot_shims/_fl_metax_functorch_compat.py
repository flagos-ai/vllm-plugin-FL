"""FL MetaX boot shim — fill functorch config keys vLLM 0.28 expects.

E9_graph (enforce_eager=False) died in vllm/compilation/backends.py:
  torch._functorch.config.patch(autograd_cache_normalize_inputs=True)
  AttributeError: ... does not exist
MetaX torch 2.8.0+metax3.7.0.7 only has remote_autograd_cache_default;
vLLM 0.28 was written against newer torch. Upstream files are NEVER edited.

Kill switch: FL_METAX_FUNCTORCH_PATCH=0
"""
from __future__ import annotations

import os as _os

_KEYS = {
    "autograd_cache_normalize_inputs": True,
}


def _enabled() -> bool:
    return _os.environ.get("FL_METAX_FUNCTORCH_PATCH", "") != "0"


def install() -> dict:
    if not _enabled():
        return {"installed": False, "reason": "kill"}
    try:
        import torch._functorch.config as cfg
        from torch.utils._config_module import _ConfigEntry, _UNSET_SENTINEL
    except Exception as exc:
        return {"installed": False, "reason": "import: %s" % (exc,)}

    added = []
    skipped = []
    for name, default in _KEYS.items():
        ent = cfg._config.get(name)
        if ent is not None and not getattr(ent, "hide", False):
            skipped.append(name)
            continue
        e = object.__new__(_ConfigEntry)
        e.default = default
        e.value_type = type(default)
        e.user_override = _UNSET_SENTINEL
        e.justknob = None
        e.env_value_force = _UNSET_SENTINEL
        e.env_value_default = _UNSET_SENTINEL
        e.hide = False
        e.alias = None
        cfg._config[name] = e
        added.append(name)
    return {"installed": True, "added": added, "skipped": skipped}


_STATUS = install()


def status() -> dict:
    return dict(_STATUS)

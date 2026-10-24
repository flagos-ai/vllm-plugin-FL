# Copyright (c) 2025 BAAI. All rights reserved.
"""Alias vLLM <= 0.24 FLA module paths onto the >= 0.28 layout.

vLLM 0.28 moved the Flash-Linear-Attention kernels from
``vllm.model_executor.layers.fla`` to
``vllm.third_party.flash_linear_attention``.  The public symbols are
unchanged, only the package prefix moved.

The plugin (and several vendor backends) still spell the old prefix in module
level imports, string constants and runtime ``importlib`` calls, because those
spellings are correct for every vLLM release the plugin supports up to 0.24.
Rewriting each site would only re-encode the same knowledge in a dozen places
and would also force edits to vendor packages owned by other hardware teams.

Instead this module registers the *whole* old prefix in ``sys.modules`` as an
alias of the new package, so every existing spelling keeps working verbatim.
It is deliberately a no-op whenever the alias is unnecessary or unsafe:

* vLLM <= 0.24 still ships the real ``...layers.fla`` package -> do nothing.
* the new ``vllm.third_party.flash_linear_attention`` is absent -> do nothing.

The hook is idempotent because vLLM loads platform plugins more than once per
process (model inspection, worker spawn, engine core).
"""

import importlib
import importlib.util
import logging
import pkgutil
import sys

logger = logging.getLogger(__name__)

_OLD_PREFIX = "vllm.model_executor.layers.fla"
_NEW_PREFIX = "vllm.third_party.flash_linear_attention"

# Only the subpackages the plugin actually reaches through the old prefix.
_ALIASED_SUBPREFIXES = (_NEW_PREFIX + ".ops",)

def _module_exists(name):
    """``find_spec`` that never raises on broken parent packages."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def install():
    """Install the FLA module alias.  Returns ``True`` when it was applied."""
    if _module_exists(_OLD_PREFIX):
        # vLLM <= 0.24: the genuine package is importable, leave it alone.
        return False
    if not _module_exists(_NEW_PREFIX):
        return False

    new_pkg = importlib.import_module(_NEW_PREFIX)
    modules = {_NEW_PREFIX: new_pkg}
    for subprefix in _ALIASED_SUBPREFIXES:
        if not _module_exists(subprefix):
            continue
        sub_pkg = importlib.import_module(subprefix)
        modules[subprefix] = sub_pkg
        for info in pkgutil.iter_modules(sub_pkg.__path__):
            name = "%s.%s" % (subprefix, info.name)
            try:
                modules[name] = importlib.import_module(name)
            except Exception as exc:  # pragma: no cover - backend dependent
                logger.debug("[vllm_fl] FLA alias skipped %s: %s", name, exc)

    added = 0
    for new_name, module in modules.items():
        old_name = _OLD_PREFIX + new_name[len(_NEW_PREFIX):]
        if old_name not in sys.modules:
            sys.modules[old_name] = module
            added += 1

    # ``import vllm.model_executor.layers.fla.ops`` resolves attributes through
    # the parent packages rather than through sys.modules, so the ``fla``
    # attribute must also be bound on the real parent package.
    layers = importlib.import_module("vllm.model_executor.layers")
    if not hasattr(layers, "fla"):
        setattr(layers, "fla", new_pkg)

    logger.info(
        "[vllm_fl] aliased %s -> %s (%d modules)",
        _OLD_PREFIX,
        _NEW_PREFIX,
        added,
    )
    return True


apply_fla_module_relocation = install

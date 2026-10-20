# Copyright 2026 FlagOS Contributors
"""Keep release MetaX attention importable without the optional top-k TLE API."""

import hashlib
import importlib.machinery
import importlib.util
import sys

import vllm_fl.envs as fl_envs

_OPS_MODULE = "flaggems_vllm.runtime.backend._metax.ops"
# FlagGems-vllm v0.2.0, b5c2c2249319277ccda4d41aa0f65164bd9b2a19.
_OPS_SOURCE_SHA256 = "6cff2a2ccf74ff3986d16d8f4b268933a7a04774d88276949b15ec953d79f1fc"
_TOPK_IMPORT = f"from {_OPS_MODULE}.persistent_topk import persistent_topk\n"


class _OpsLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_source(fullname)
        if (
            source is None
            or hashlib.sha256(source.encode()).hexdigest() != _OPS_SOURCE_SHA256
        ):
            raise ImportError(
                "MetaX optional TLE compatibility requires FlagGems-vllm v0.2.0 "
                "commit b5c2c2249319277ccda4d41aa0f65164bd9b2a19"
            )
        # Omit only the unavailable optional export. All remaining imports and
        # implementations execute unchanged; no installed package file is edited.
        source = source.replace(_TOPK_IMPORT, "", 1)
        source += '\n__all__.remove("persistent_topk")\n'
        return compile(source, self.path, "exec")


class _OpsFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname != _OPS_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(
            spec.loader, importlib.machinery.SourceFileLoader
        ):
            spec.loader = _OpsLoader(fullname, spec.origin)
        return spec


def prepare_flaggems_vllm_metax():
    """Called only for MetaX, before loading FlagGems-vllm's vendor exports."""
    if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        return
    try:
        if importlib.util.find_spec("triton.experimental.tle.language") is not None:
            return
    except ModuleNotFoundError as error:
        if error.name not in (
            "triton.experimental",
            "triton.experimental.tle",
            "triton.experimental.tle.language",
        ):
            raise
    if not any(isinstance(finder, _OpsFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _OpsFinder())

# SPDX-License-Identifier: Apache-2.0
"""Give each Hygon CI worker its own FlagGems autotuning database."""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

_cache = os.environ.get("FLAGGEMS_CACHE_DIR")
if _cache:
    # Python imports sitecustomize before pytest, Torch, or spawned workers.
    # Children inherit the parent path and append their own distinct PID.
    _isolated_cache = os.path.join(_cache, str(os.getpid()))
    os.environ["FLAGGEMS_CACHE_DIR"] = _isolated_cache

# Keep any startup hook supplied by the runner image. The temporary helper
# precedes it on PYTHONPATH solely to isolate the cache before vendor imports.
_directory = Path(__file__).resolve().parent
_paths = [entry for entry in sys.path if Path(entry or ".").resolve() != _directory]
_original = importlib.machinery.PathFinder.find_spec("sitecustomize", _paths)
if _original is not None and _original.loader is not None:
    _module = importlib.util.module_from_spec(_original)
    sys.modules["sitecustomize"] = _module
    _original.loader.exec_module(_module)
if _cache:
    os.environ["FLAGGEMS_CACHE_DIR"] = _isolated_cache

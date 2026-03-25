"""vllm_fl version information (PyTorch-style).

This module is intentionally lightweight and safe to import.

At build time (pip install .), setuptools-scm writes `vllm_fl/_version.py`
with pre-computed `git_version` and `git_date`.  At runtime we simply import
that static file, exactly as vLLM does, so no subprocess is needed in the
happy path.  Subprocess git calls are only used as a last-resort fallback
(e.g. pip install -e . or running from source without a build step).
"""

from __future__ import annotations

import os
import subprocess
from importlib import metadata


def _pkg_version() -> str:
    # Project name in pyproject.toml is "vllm-plugin-fl".
    try:
        return metadata.version("vllm-plugin-fl")
    except Exception:
        pass
    try:
        from . import _version
        return _version.__version__
    except Exception:
        return "0.0.0+unknown"


def _git_head_from_repo() -> str | None:
    # Only works from a git checkout with `git` available.
    try:
        root = os.path.dirname(os.path.dirname(__file__))
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.2,
        ).strip() or None
    except Exception:
        return None


def _git_commit_date_from_repo() -> str | None:
    # Returns YYYY-MM-DD of the HEAD committer date.
    try:
        root = os.path.dirname(os.path.dirname(__file__))
        out = subprocess.check_output(
            ["git", "-C", root, "show", "-s", "--format=%cI", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.2,
        ).strip()
        return out[:10] if out else None
    except Exception:
        return None


def _load_scm() -> tuple[str | None, str | None]:
    """Read (commit_id, commit_date) from the build-time generated _version.py.

    setuptools-scm writes `vllm_fl/_version.py` at `pip install .` time with
    pre-computed `git_version` (commit sha) and `git_date` (YYYY-MM-DD)
    derived from the version string.  Importing that static file avoids any
    runtime git subprocess in the common installed case.
    """
    try:
        from . import _version as _v
    except Exception:
        return None, None
    cid = getattr(_v, "git_version", None)
    cdate = getattr(_v, "git_date", None)
    cid = cid if isinstance(cid, str) and cid not in ("", "Unknown") else None
    cdate = cdate if isinstance(cdate, str) and cdate not in ("", "Unknown") else None
    return cid, cdate


__version__ = _pkg_version()

_scm_id, _scm_date = _load_scm()

# Public: git commit id of the installed package/build (best-effort).
git_version: str = _scm_id or _git_head_from_repo() or "Unknown"

# Public: git metadata aligned with torch.version.git_info style.
git_info: dict[str, str] = {
    "id": git_version,
    "date": _scm_date or _git_commit_date_from_repo() or "Unknown",
}


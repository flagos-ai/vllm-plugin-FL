#!/usr/bin/env python3
"""Install ``vllm_fl/_fl_boot_shims`` as early-import ``.pth`` hooks.

Why
---
Several shims under ``vllm_fl/_fl_boot_shims`` must execute *before* the module
they patch is first imported: ``triton.jit`` reads its kernel source from disk
at first import, and ``torch``/``vllm`` read their config at import time.  The
earliest portable hook in CPython start-up is ``site``: a ``.pth`` file placed
in ``site-packages`` whose lines start with ``import`` is executed by
``site.addpackage`` before any user code runs.

This script reproduces, in a parameterised way, the mechanism used on the MetaX
boxes (which hand-writes one ``.pth`` per shim into
``site-packages`` with a hard-coded machine path)::

    zz_fl_<shim-stem>.pth      e.g. zz_fl_metax_annassign_compat.pth

Each generated file holds a single executable line of the form::

    import os,sys; sys.path.insert(0, "<SHIMS_DIR>"); \
os.environ.get("FL_BOOT_SHIMS_PATCH","1")!="0" and __import__("<shim-stem>")

Usage
-----
    python scripts/install_fl_boot_shims.py             # install / refresh
    python scripts/install_fl_boot_shims.py --check      # CI drift gate
    python scripts/install_fl_boot_shims.py --dry-run    # show, do not write
    python scripts/install_fl_boot_shims.py --uninstall  # remove (see below)

Switches
--------
* ``FL_BOOT_SHIMS_PATCH=0``  -- **master switch**, disables every shim at once.
  It is baked into each generated ``.pth``; any value other than ``0`` (or the
  variable being unset) leaves the shims enabled.
* Per-shim switches are honoured by the shims themselves, e.g.
  ``FL_METAX_ANNASSIGN_PATCH=0``, ``FL_METAX_TOPK_PATCH=0``,
  ``FL_METAX_SAMPLER_PATCH=0``, ``FL_METAX_TORCH_ACCEL_PATCH=0``.

Safety
------
* Idempotent: a ``.pth`` is rewritten only when its bytes actually differ.
* ``--uninstall`` removes **only** files this script manages -- those carrying
  the marker line below.  Hand-written or vendor ``.pth`` files are left alone.
* No credential and no site-specific absolute path is hard-coded here; the
  shims directory defaults to the checkout this script lives in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
from pathlib import Path

MARKER = "# managed-by: scripts/install_fl_boot_shims.py"
SHIM_PREFIX = "_fl_"
DEFAULT_PTH_PREFIX = "zz_fl_"
DEFAULT_MASTER_SWITCH = "FL_BOOT_SHIMS_PATCH"


def repo_root() -> Path:
    """Return the repository root (parent of this script's directory)."""
    return Path(__file__).resolve().parent.parent


def find_shims(shims_dir: Path) -> list[str]:
    """Return the module stems that need a ``.pth`` hook, sorted.

    Only ``_fl_*.py`` files are early-injected; the remaining helpers in the
    directory (``_kths_triton_unified_attention.py``, ``_topk_topp_triton.py``)
    are imported by the shims themselves and must not get a hook.
    """
    if not shims_dir.is_dir():
        raise SystemExit(f"[fl-shims] shims directory not found: {shims_dir}")
    stems = [
        p.stem for p in sorted(shims_dir.glob(f"{SHIM_PREFIX}*.py")) if p.is_file()
    ]
    if not stems:
        raise SystemExit(
            f"[fl-shims] no '{SHIM_PREFIX}*.py' shim found under {shims_dir}"
        )
    return stems


def pth_filename(stem: str, pth_prefix: str) -> str:
    """``_fl_metax_topk_compat`` -> ``zz_fl_metax_topk_compat.pth``."""
    short = stem[len(SHIM_PREFIX) :] if stem.startswith(SHIM_PREFIX) else stem
    return f"{pth_prefix}{short}.pth"


def pth_body(shims_dir: Path, stem: str, master_switch: str) -> str:
    """Build the full ``.pth`` text for one shim."""
    guard = (
        f"os.environ.get({json.dumps(master_switch)},'1')!='0' "
        f"and __import__({json.dumps(stem)})"
    )
    line = f"import os,sys; sys.path.insert(0, {json.dumps(str(shims_dir))}); {guard}"
    return f"{MARKER}\n{line}\n"


def detect_site_packages() -> Path:
    """Best-effort ``site-packages`` detection for the running interpreter."""
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        return Path(purelib)
    import site as _site

    for candidate in _site.getsitepackages():
        return Path(candidate)
    return Path(_site.getusersitepackages())


def is_managed(path: Path) -> bool:
    """True when ``path`` is a ``.pth`` this script wrote."""
    try:
        return MARKER in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def plan_writes(
    shims_dir: Path, site_packages: Path, pth_prefix: str, master_switch: str
) -> list[tuple[Path, str]]:
    """Return ``[(pth_path, content), ...]`` for every shim."""
    return [
        (
            site_packages / pth_filename(stem, pth_prefix),
            pth_body(shims_dir, stem, master_switch),
        )
        for stem in find_shims(shims_dir)
    ]


def install(writes, dry_run: bool) -> int:
    changed = 0
    for path, content in writes:
        current = None
        if path.exists():
            try:
                current = path.read_text(encoding="utf-8")
            except OSError:
                current = None
        if current == content:
            print(f"[fl-shims] unchanged {path.name}")
            continue
        changed += 1
        if dry_run:
            print(f"[fl-shims] would write {path.name}")
            continue
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)  # atomic; never leaves a half-written .pth
        print(f"[fl-shims] wrote {path.name}")
    return changed


def uninstall(writes, dry_run: bool) -> int:
    removed = 0
    for path, _ in writes:
        if not path.exists():
            continue
        if not is_managed(path):
            print(f"[fl-shims] SKIP {path.name} (not managed by this script)")
            continue
        removed += 1
        if dry_run:
            print(f"[fl-shims] would remove {path.name}")
        else:
            path.unlink()
            print(f"[fl-shims] removed {path.name}")
    return removed


def check(writes) -> int:
    drift = []
    for path, content in writes:
        if not path.exists():
            drift.append(f"missing: {path.name}")
        elif path.read_text(encoding="utf-8") != content:
            drift.append(f"stale:   {path.name}")
    if drift:
        for item in drift:
            print(f"[fl-shims] {item}")
        print(f"[fl-shims] {len(drift)} hook(s) out of date -> run without --check")
        return 1
    print(f"[fl-shims] all {len(writes)} hook(s) up to date")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install vllm_fl boot shims as early-import .pth hooks."
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=None,
        help="target site-packages (default: auto-detect for this interpreter)",
    )
    parser.add_argument(
        "--shims-dir",
        type=Path,
        default=None,
        help="directory holding the shims (default: <repo>/vllm_fl/_fl_boot_shims)",
    )
    parser.add_argument("--pth-prefix", default=DEFAULT_PTH_PREFIX)
    parser.add_argument("--master-switch", default=DEFAULT_MASTER_SWITCH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args(argv)

    shims_dir = (
        args.shims_dir if args.shims_dir else repo_root() / "vllm_fl" / "_fl_boot_shims"
    ).resolve()
    site_packages = (
        args.site_packages if args.site_packages else detect_site_packages()
    ).resolve()

    if (
        not args.dry_run
        and not args.check
        and not args.uninstall
        and not os.access(site_packages, os.W_OK)
    ):
        print(f"[fl-shims] ERROR: {site_packages} is not writable", file=sys.stderr)
        return 2

    writes = plan_writes(shims_dir, site_packages, args.pth_prefix, args.master_switch)
    print(f"[fl-shims] shims-dir    : {shims_dir}")
    print(f"[fl-shims] site-packages: {site_packages}")
    print(f"[fl-shims] hooks        : {len(writes)}")
    print(f"[fl-shims] master switch: {args.master_switch}=0 disables all shims")

    if args.check:
        return check(writes)
    if args.uninstall:
        uninstall(writes, args.dry_run)
        return 0
    install(writes, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

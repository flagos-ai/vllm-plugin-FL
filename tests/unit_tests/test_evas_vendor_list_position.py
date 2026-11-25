# SPDX-License-Identifier: Apache-2.0

"""Merge hygiene: evas is appended last to the vendor lists upstream owns.

evas is a newly adapted vendor, so every vendor list that also exists upstream
has to carry it *last*.  Being mid-list is not a runtime bug -- these are
membership tests and dict lookups, and order changes nothing -- but it turns
every later upstream edit of that line into a conflict that has to be resolved
by hand, which is how an upstream branch ends up with evas missing.

The walk below checks every shared container that takes an evas entry, wherever
it lives in the package, and the guard test fails if fewer than five are found,
so a move of the package cannot make it pass vacuously.  Last is not the same as
present: presence is pinned by ``test_lpu_platform_divergences.py``.
"""

import ast
from pathlib import Path

import vllm_fl

VENDORS = {
    "cuda",
    "xpu",
    "npu",
    "musa",
    "gcu",
    "hpu",
    "mlu",
    "sdaa",
    "maca",
    "metax",
    "ascend",
    "camb",
    "thead",
    "iluvatar",
    "kunlunxin",
    "ppu",
    "corex",
    "mthreads",
    "moore",
    "muxi",
    "biren",
    "enflame",
    "rocm",
    "amd",
    "gen",
    "spark",
    "tsingmicro",
    "sunrise",
    "hygon",
    "lpu",
    "evas",
}
OURS = {"evas", "lpu"}


def _literal_strings(node):
    """The string elements of a list/tuple/set, or the keys of a dict."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        elements = node.elts
    elif isinstance(node, ast.Dict):
        elements = node.keys
    else:
        return None
    return [
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]


def _mid_list_sites():
    """Every vendor literal in the package whose last vendor name is not ours."""
    package = Path(vllm_fl.__file__).resolve().parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            values = _literal_strings(node)
            if values is None:
                continue
            names = [value for value in values if value in VENDORS]
            if len(names) < 2 or not any(name in OURS for name in names):
                continue
            if names[-1] not in OURS:
                rel = path.relative_to(package.parent)
                offenders.append(f"{rel}:{node.lineno} {names}")
    return offenders


def test_evas_is_the_last_vendor_in_every_vendor_list():
    offenders = _mid_list_sites()
    assert not offenders, (
        "evas/lpu must be the LAST vendor in these lists -- a newly adapted "
        "vendor appends, so later upstream edits to the line still merge "
        "cleanly:\n  " + "\n  ".join(offenders)
    )


def test_the_walk_actually_sees_the_shared_containers():
    """A guard on the guard: the walk must find the containers we know about.

    If the package moves or the literal shapes change, the check above would
    pass by finding nothing at all.
    """
    sites = []
    package = Path(vllm_fl.__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            values = _literal_strings(node)
            if values is None:
                continue
            names = [value for value in values if value in VENDORS]
            if len(names) >= 2 and any(name in OURS for name in names):
                sites.append(f"{path.name}:{node.lineno}")
    assert len(sites) >= 5, f"the walk found only {len(sites)} vendor lists: {sites}"
    assert any(name.startswith("platform.py") for name in sites)
    assert any(name.startswith("utils.py") for name in sites)

#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.

"""Verify effective vendor dependencies and fixed-commit vLLM source bytes."""

import argparse
import ast
import datetime
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import subprocess
from collections import deque
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

UPSTREAM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
SDK = {
    "torch": "2.11.0+cpu",
    "torch-gcu": "2.11.0+3.8.20260713",
    "flag-gems": "5.4.0",
    "flagtree": "0.7.0+enflame3.6",
    "triton-gcu": "3.6.0+1.0.20260722",
    "openai": "3.28.0",
    "pandas": "3.0.6",
}


def verify(source: Path):
    observed = datetime.datetime.now(datetime.timezone.utc).isoformat()
    version = Version(metadata.version("vllm"))
    assert version.base_version == "0.32.0" and "empty" in (version.local or "").split(
        "."
    )
    for name, expected in SDK.items():
        assert metadata.version(name) == expected, (
            name,
            metadata.version(name),
            expected,
        )

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    assert git("rev-parse", "HEAD").decode().strip() == UPSTREAM_COMMIT
    base = Path(importlib.util.find_spec("vllm").origin).parent
    assert (
        base.resolve()
        == Path(metadata.distribution("vllm").locate_file("vllm")).resolve()
    ), base
    files = []
    for row in git("ls-tree", "-r", "-z", UPSTREAM_COMMIT, "--", "vllm").split(b"\0"):
        if not row:
            continue
        info, raw_path = row.split(b"\t", 1)
        mode = info.split()[0]
        name = raw_path.decode()
        if not name.endswith(".py") or mode not in (b"100644", b"100755"):
            continue
        expected = git("show", UPSTREAM_COMMIT + ":" + name)
        actual = base / Path(name).relative_to("vllm")
        before = actual.stat()
        data = actual.read_bytes()
        after = actual.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ), name
        assert data == expected, name
        files.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    # FlagTree deliberately owns the triton namespace without a triton dist-info.
    # Keep this metadata exception explicit; never install upstream Triton over it.
    provider = metadata.distribution("flagtree")
    namespace = Path(importlib.util.find_spec("triton").origin).resolve()
    assert namespace == Path(provider.locate_file("triton/__init__.py")).resolve()
    namespace_version = None
    for node in ast.parse(namespace.read_bytes()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            namespace_version = ast.literal_eval(node.value)
    assert namespace_version == "3.6.0", namespace_version

    common = git("show", UPSTREAM_COMMIT + ":requirements/common.txt").decode()
    roots = ["vllm"] + [
        line.split("#", 1)[0].strip()
        for line in common.splitlines()
        if line.split("#", 1)[0].strip()
    ]
    queue = deque((Requirement(raw), frozenset()) for raw in roots)
    visited, checked, provided, unresolved = set(), [], [], []
    while queue:
        req, extras = queue.popleft()
        if req.marker and not any(
            req.marker.evaluate({"extra": e}) for e in (extras or {""})
        ):
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            installed = None
        satisfied = installed is not None and (
            not req.specifier or req.specifier.contains(installed, prereleases=True)
        )
        checked.append(
            {
                "requirement": str(req),
                "installed": installed,
                "metadata_satisfied": satisfied,
            }
        )
        if not satisfied:
            if canonicalize_name(req.name) == "triton" and (
                not req.specifier or req.specifier.contains(namespace_version)
            ):
                provided.append(
                    {
                        "requirement": str(req),
                        "metadata_satisfied": False,
                        "provider": "flagtree",
                        "namespace_version": namespace_version,
                        "path": str(namespace),
                    }
                )
            else:
                unresolved.append(str(req))
            continue
        key = canonicalize_name(req.name), frozenset(req.extras)
        if key not in visited:
            visited.add(key)
            queue.extend(
                (Requirement(child), frozenset(req.extras))
                for child in metadata.requires(req.name) or []
            )
    report = {
        "observed_at": observed,
        "upstream_commit": UPSTREAM_COMMIT,
        "runtime_version": str(version),
        "scope": "all tracked regular vllm Python files versus fixed Git blobs; effective common dependency closure",
        "files": files,
        "requirements": checked,
        "vendor_namespace_providers": provided,
        "unresolved": unresolved,
    }
    assert not unresolved, unresolved
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.source)
    raw = (json.dumps(report, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    summary = {
        "passed": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "observed_at": report["observed_at"],
        "verified_python_files": len(report["files"]),
        "checked_requirements": len(report["requirements"]),
        "vendor_namespace_providers": report["vendor_namespace_providers"],
        "unresolved": [],
        "output": str(args.output),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    encoded = json.dumps(summary)
    assert len(encoded.encode()) <= 8192
    print(encoded)


if __name__ == "__main__":
    main()

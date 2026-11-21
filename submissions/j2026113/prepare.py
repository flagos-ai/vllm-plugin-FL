#!/usr/bin/env python3
"""Reconstruct a reviewed source layout from this competition submission."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = "13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0"

def read_json(path):
    return json.loads(path.read_bytes())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(root, expected):
    for rel, value in expected.items():
        if digest(root / rel) != value:
            raise ValueError(f"SHA256 mismatch: {root / rel}")

def apply(source, dest):
    if os.name == "nt" and not str(source).startswith(chr(92)*2+"?"+chr(92)):
        source = Path(chr(92)*2+"?"+chr(92)+str(source.resolve()))
    for path in source.rglob("*"):
        if path.is_file():
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

def read_baseline():
    raw = subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "archive", BASE], cwd=REPO
    )
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        result = {}
        for member in archive.getmembers():
            if member.isfile():
                result[member.name] = (archive.extractfile(member).read(), member.mode)
            elif member.issym():
                target = (Path(member.name).parent / member.linkname).as_posix()
                result[member.name] = (archive.extractfile(target).read(), 0o644)
        return result

def restore(platform, out, gems):
    source = HERE / platform
    baseline = read_baseline()
    if platform == "metax":
        manifest = read_json(source / "base.json")
        if gems is None:
            raise ValueError("Supply --gems as the parent of the installed flag_gems package")
        gems = gems.resolve()
        if not (gems / "flag_gems" / "__init__.py").is_file():
            raise ValueError("--gems must be the parent of the installed flag_gems package")
        verify(gems, manifest["gems_unmodified"])
    out.mkdir(parents=True, exist_ok=False)
    if platform == "metax":
        plugin = out / "vllm-plugin-FL"
        for rel, (content, mode) in baseline.items():
            path = plugin / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode)
        apply(source / "plugin", plugin)
        verify(plugin, manifest["plugin_expected"])
        for rel in manifest["gems_unmodified"]:
            path = out / "FlagGems" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gems / rel, path)
        apply(source / "FlagGems", out / "FlagGems")
        verify(out / "FlagGems", manifest["gems_expected"])
        apply(source / "runtime", out / "runtime")
        for name in ("reproduce_c500.sh", "readme.md", "evidence/model_file_hashes.json"):
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, path)
        files = sorted(p for p in out.rglob("*") if p.is_file())
        (out / "SOURCE_SHA256SUMS").write_bytes("".join(
            f"{digest(p)}  {p.relative_to(out).as_posix()}\n" for p in files
        ).encode())
    else:
        apply(source / "runtime", out)
        manifest = read_json(out / "manifest.json")
        for rel, expected in read_json(source / "runtime_plugin_base.json").items():
            path = out / "plugin" / rel
            if not path.is_file():
                content, mode = baseline[rel]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(mode)
            if digest(path) != expected:
                raise ValueError(f"SHA256 mismatch: {path}")
        verify(out, manifest["payload_SHA256"])
    print(f"{platform}: verified all reconstructed source hashes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=("metax", "iluvatar"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--gems", type=Path)
    args = parser.parse_args()
    restore(args.platform, args.output.resolve(), args.gems)

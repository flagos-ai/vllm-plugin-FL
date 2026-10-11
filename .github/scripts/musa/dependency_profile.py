#!/usr/bin/env python3
"""Verify an explicit ordinary-wheel profile; never import accelerator modules.

Default baseline retains inherited FlagGems/FlagTree. Latest installs exactly
three supplied, whole-SHA-bound wheels inside the current fresh private venv.
This static verification is not accelerator, model, or Gate acceptance.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import importlib.machinery
import importlib.metadata as md
import io
import json
import math
import os
import platform
import re
import signal
import stat
import sys
import sysconfig
import zipfile
from pathlib import Path, PurePosixPath

NAMES = {"flag-gems", "flagtree", "pyyaml"}
MODULES = {"flag-gems": "flag_gems", "flagtree": "triton", "pyyaml": "yaml"}
WHEEL_CAP = 512 * 1024 * 1024
JSON_CAP = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)

    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), "Nonfinite JSON number")
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def identity(item):
    return (item.st_dev, item.st_ino, item.st_mode, item.st_size,
            item.st_mtime_ns, item.st_ctime_ns)


def read(path, cap=JSON_CAP, expected=None):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path,
            "Noncanonical input: " + str(path))
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= cap,
            "Regular input/cap: " + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require(identity(before) == identity(opened), "Changed before read")
        chunks, count = [], 0
        while True:
            chunk = os.read(fd, min(65536, cap + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            require(count <= cap, "Input grew beyond cap")
            chunks.append(chunk)
        require(identity(opened) == identity(os.fstat(fd)), "Changed during read")
    finally:
        os.close(fd)
    require(identity(before) == identity(path.lstat()), "Changed after read")
    raw = b"".join(chunks)
    require(len(raw) == before.st_size, "Incomplete input EOF")
    fp = {"path": str(path), "bytes": len(raw),
          "sha256": hashlib.sha256(raw).hexdigest()}
    if expected is not None:
        require(type(expected["bytes"]) is int and expected["bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]), "Input pin")
        require((fp["bytes"], fp["sha256"]) ==
                (expected["bytes"], expected["sha256"]), "Whole input pin: " + str(path))
    return raw, fp


def save(path, raw):
    require(path.parent.resolve(strict=True) == path.parent, "Canonical output parent")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(raw)
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def save_json(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    require(len(raw) <= JSON_CAP, "Profile report exceeds 8 MiB")
    return save(path, raw)


def wheel_review(raw, fp, expected):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [x.filename for x in infos]
        require(0 < len(names) < 20000 and len(names) == len(set(names)),
                "Wheel member uniqueness/count")
        regular, directories, total = set(), set(), 0
        for entry in infos:
            member = entry.filename
            parts = PurePosixPath(member).parts
            mode = stat.S_IFMT(entry.external_attr >> 16)
            require(parts and not member.startswith("/") and "\\" not in member
                    and ".." not in parts and member == PurePosixPath(member).as_posix()
                    + ("/" if entry.is_dir() else ""), "Canonical wheel member")
            require(not entry.flag_bits & 1, "Encrypted wheel member")
            if entry.is_dir():
                require(mode in (0, stat.S_IFDIR) and entry.file_size == 0
                        and archive.read(member) == b"", "Safe empty ZIP directory")
                directories.add(member[:-1])
            else:
                require(mode in (0, stat.S_IFREG) and entry.file_size <= WHEEL_CAP,
                        "Wheel regular-file type/size")
                total += entry.file_size
                regular.add(member)
        require(total <= 1024 * 1024 * 1024 and not regular & directories,
                "Expanded wheel cap/file-directory conflict")
        paths = {PurePosixPath(x) for x in regular}
        require(all(not (set(PurePosixPath(x).parents) - {PurePosixPath(".")}) & paths
                    for x in names), "Wheel file used as directory")
        metadata_paths = [x for x in regular if len(PurePosixPath(x).parts) == 2
                          and x.endswith(".dist-info/METADATA")]
        require(len(metadata_paths) == 1, "One top-level wheel METADATA")
        dist_info = metadata_paths[0].split("/")[0]
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_paths[0]))
        require(len(metadata.get_all("Name", [])) == 1
                and len(metadata.get_all("Version", [])) == 1, "Unique name/version")
        name, version = metadata["Name"], metadata["Version"]
        require(normalized(name) == normalized(expected["name"])
                and version == expected["version"], "Wheel distribution/version")
        roots = {
            "flag-gems": {"flag_gems", "flaggems_tests", "flaggems_benchmark", "flaggems_setup.py"},
            "flagtree": {"flagtree", "triton"},
            "pyyaml": {"yaml", "_yaml"},
        }[normalized(name)] | {dist_info}
        require(all(PurePosixPath(x).parts[0] in roots for x in regular),
                "Dependency wheel cannot overwrite other packages or SDK")
        record = dist_info + "/RECORD"
        record_raw = archive.read(record)
        require(len(record_raw) <= JSON_CAP, "Wheel RECORD cap")
        rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8"))))
        require(all(len(x) == 3 for x in rows) and len(rows) == len(regular)
                and len({x[0] for x in rows}) == len(rows)
                and {x[0] for x in rows} == regular, "Complete regular-file wheel RECORD")
        payload = []
        for member, encoded, size in rows:
            body = archive.read(member)
            if member == record:
                require(encoded == size == "", "Wheel RECORD self entry")
                continue
            digest = hashlib.sha256(body).digest()
            require(encoded == "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")
                    and size == str(len(body)), "Wheel RECORD hash/size: " + member)
            payload.append({"member": member, "bytes": len(body), "sha256": digest.hex()})
        wheel = email.parser.BytesParser().parsebytes(archive.read(dist_info + "/WHEEL"))
        tags = wheel.get_all("Tag", [])
        require(tags, "Wheel tag missing")
        if normalized(name) == "flagtree":
            require(tags == ["cp310-cp310-linux_x86_64"] and "+mthreads" in version,
                    "Require an explicit MUSA FlagTree wheel, not a CUDA artifact")
        elif normalized(name) == "flag-gems":
            require(tags == ["py3-none-any"], "Ordinary FlagGems Python wheel")
        else:
            require(version == "6.0.1" and all(x.startswith("cp310-cp310-") for x in tags),
                    "Explicit cp310 PyYAML 6.0.1 wheel")
        return dict(fp, name=name, version=version, dist_info=dist_info, payload=payload,
                    wheel_tags=tags, wheel_RECORD_verified=True)


def static_origin(name):
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    require(spec is not None and spec.origin is not None, "Module source not found: " + name)
    path = Path(spec.origin).resolve(strict=True)
    return read(path)[1]


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and type(manifest.get("schema_version")) is int
            and manifest["schema_version"] == 1 and manifest.get("profile") == "latest",
            "Latest manifest schema/profile")
    wheels = manifest.get("wheels")
    require(isinstance(wheels, list) and len(wheels) == 3
            and {normalized(x["name"]) for x in wheels} == NAMES,
            "Latest profile exactly FlagGems/FlagTree/PyYAML")
    for row in wheels:
        require(isinstance(row["version"], str) and row["version"]
                and re.fullmatch(r"/data/[/A-Za-z0-9_.+-]+\.whl", row["path"])
                and ".." not in PurePosixPath(row["path"]).parts
                and type(row["bytes"]) is int and 0 < row["bytes"] <= WHEEL_CAP
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Explicit wheel fields")
    require(manifest.get("vendor_versions") ==
            {"torch": "2.9.0", "torch_musa": "2.9.0", "musa_sdk": "4.3.5"},
            "Preserve explicit SDK/Torch baseline")
    files = manifest.get("vendor_files")
    roles = {"torch_source", "torch_musa_source", "torch_metadata", "torch_musa_metadata", "sdk_version"}
    require(isinstance(files, list) and len(files) == 5
            and {x["role"] for x in files} == roles, "Five exact vendor source roles")
    for row in files:
        require(isinstance(row["path"], str) and row["path"].startswith("/usr/local/")
                and type(row["bytes"]) is int and 0 < row["bytes"] <= JSON_CAP
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Explicit vendor source fields")
    require(isinstance(manifest.get("source_lineage"), dict) and manifest["source_lineage"],
            "Record the reviewed artifact source lineage")


def vendor_snapshot(site, manifest):
    modules, distributions = {}, {}
    for name in ("torch", "torch_musa"):
        origin = static_origin(name)
        require(not Path(origin["path"]).is_relative_to(site), "Inherited vendor source required")
        modules[name] = origin
        dist = md.distribution(name)
        metadata = Path(dist._path).resolve(strict=True) / "METADATA"
        require(not metadata.is_relative_to(site), "Inherited vendor distribution required")
        distributions[name] = {"version": dist.version, "metadata": read(metadata)[1],
                               "record": read(metadata.parent / "RECORD")[1]}
    snapshot = {"modules": modules, "distributions": distributions}
    if manifest is not None:
        require(manifest["vendor_versions"] ==
                {"torch": "2.9.0", "torch_musa": "2.9.0", "musa_sdk": "4.3.5"},
                "Latest profile keeps the explicit SDK/Torch baseline")
        require(all(x["version"] == "2.9.0" for x in distributions.values()),
                "Latest profile cannot upgrade Torch or TorchMUSA")
        actual = {"torch_source": modules["torch"], "torch_musa_source": modules["torch_musa"],
                  "torch_metadata": distributions["torch"]["metadata"],
                  "torch_musa_metadata": distributions["torch_musa"]["metadata"]}
        expected = {x["role"]: x for x in manifest["vendor_files"]}
        require(len(expected) == len(manifest["vendor_files"]) == 5
                and set(expected) == set(actual) | {"sdk_version"}, "Five vendor source profiles")
        sdk = expected["sdk_version"]
        require(sdk["path"] == "/usr/local/musa-4.3.5/version.json", "Explicit SDK 4.3.5 path")
        actual["sdk_version"] = read(Path(sdk["path"]), expected=sdk)[1]
        for role, fp in actual.items():
            require(fp == {key: expected[role][key] for key in ("path", "bytes", "sha256")},
                    "Vendor profile mismatch: " + role)
        snapshot["sdk"] = actual["sdk_version"]
    return snapshot


def installed_wheel(row, site, venv):
    for member in row["payload"]:
        read(site / member["member"], WHEEL_CAP, member)
    metadata = site / row["dist_info"]
    raw, record_fp = read(metadata / "RECORD")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    require(all(len(x) == 3 for x in rows) and len({x[0] for x in rows}) == len(rows),
            "Installed RECORD unique/three columns")
    records = {}
    for member, encoded, size in rows:
        require(not member.startswith("/") and "\\" not in member, "Relative installed RECORD")
        path = (site / member).resolve(strict=True)
        require(path.is_relative_to(venv), "Installed RECORD outside private venv")
        if path == metadata / "RECORD":
            require(encoded == size == "", "Installed RECORD self entry")
            records[member] = [encoded, size]
            continue
        body, _ = read(path, WHEEL_CAP)
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
        require(encoded == "sha256=" + digest and size == str(len(body)),
                "Installed RECORD hash/size: " + member)
        records[member] = [encoded, size]
    require(row["dist_info"] + "/RECORD" in records, "Installed RECORD self missing")
    for member in row["payload"]:
        digest = base64.urlsafe_b64encode(bytes.fromhex(member["sha256"])).decode().rstrip("=")
        require(records.get(member["member"]) == ["sha256=" + digest, str(member["bytes"])],
                "Installed wheel member absent from RECORD")
    direct_raw, direct_fp = read(metadata / "direct_url.json")
    direct = strict_json(direct_raw)
    require("dir_info" not in direct and direct.get("url") == Path(row["path"]).as_uri()
            and direct.get("archive_info", {}).get("hashes") == {"sha256": row["sha256"]},
            "Ordinary whole-wheel direct_url required")
    if "hash" in direct["archive_info"]:
        require(direct["archive_info"]["hash"] == "sha256=" + row["sha256"], "Direct URL hash")
    dist = md.distribution(row["name"])
    require(Path(dist._path).resolve(strict=True) == metadata and dist.version == row["version"],
            "Actual private distribution identity")
    origin = static_origin(MODULES[normalized(row["name"])])
    member = Path(origin["path"]).relative_to(site).as_posix()
    require(any(x["member"] == member and x["sha256"] == origin["sha256"]
                for x in row["payload"]), "Private module source must be actual wheel payload")
    return {"name": row["name"], "version": row["version"], "wheel":
            {key: row[key] for key in ("path", "bytes", "sha256")}, "origin": origin,
            "record": record_fp, "direct_url": direct_fp, "payload_verified": True,
            "record_verified": True, "record_file_count": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "installed"))
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("baseline", "latest"), required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--manifest-sha256", default="")
    args = parser.parse_args()
    root = args.task_dir
    require(root.is_absolute() and root.resolve(strict=True) == root, "Fresh canonical task root")
    site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    venv = root / "venv"
    require(Path(sys.prefix).resolve(strict=True) == venv and site.is_relative_to(venv)
            and sys.flags.isolated and not sys.flags.no_site, "Normal private -I -B interpreter")
    if args.phase == "installed":
        raw, _ = read(root / "dependency-profile-staged.json")
        staged = strict_json(raw)
        require(staged["profile"] == args.profile, "Profile changed during install")
        manifest = staged["manifest"]
        if manifest is not None:
            require(args.manifest == staged["manifest_fingerprint"]["path"]
                    and args.manifest_sha256 == staged["manifest_fingerprint"]["sha256"],
                    "Manifest arguments changed during install")
            read(Path(args.manifest), expected=staged["manifest_fingerprint"])
        else:
            require(not args.manifest and not args.manifest_sha256,
                    "Baseline manifest arguments changed during install")
        require(vendor_snapshot(site, manifest) == staged["vendor_before"],
                "Protected vendor source/metadata changed")
        checks = []
        for row in staged["wheels"]:
            read(Path(row["path"]), WHEEL_CAP, row)
            checks.append(installed_wheel(row, site, venv))
        save_json(root / "dependency-profile-installed.json",
                  {"profile": args.profile, "installed": checks,
                   "vendor_before_after_equal": True, "scope": "Static ordinary-wheel/source verification",
                   "GPU_ABI_passed": False, "model_passed": False, "Gate_passed": False})
        return 0
    manifest, manifest_fp, wheels = None, None, []
    if args.profile == "latest":
        require(sys.platform == "linux" and sys.version_info[:2] == (3, 10)
                and platform.machine() == "x86_64",
                "Pinned latest MUSA artifacts require CPython3.10/Linux x86_64")
        require(re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256), "Explicit latest manifest SHA")
        raw, manifest_fp = read(Path(args.manifest), 1024 * 1024)
        require(manifest_fp["sha256"] == args.manifest_sha256, "Whole manifest SHA")
        manifest = strict_json(raw)
        validate_manifest(manifest)
        for expected in manifest["wheels"]:
            source = Path(expected["path"])
            require(str(source).startswith("/data/") and source.suffix == ".whl", "Read-only /data wheel")
            body, fp = read(source, WHEEL_CAP, expected)
            row = wheel_review(body, fp, expected)
            copied = save(root / "wheels" / source.name, body)
            row.update(copied)
            wheels.append(row)
    else:
        require(not args.manifest and not args.manifest_sha256,
                "Baseline uses inherited packages; latest manifest must not be silently ignored")
    save_json(root / "dependency-profile-staged.json",
              {"schema_version": 1, "profile": args.profile, "manifest": manifest,
               "manifest_fingerprint": manifest_fp, "wheels": wheels,
               "vendor_before": vendor_snapshot(site, manifest),
               "target_modules_imported": [], "source_lineage_rederived": False,
               "GPU_ABI_passed": False, "Gate_passed": False})
    return 0


if __name__ == "__main__":
    def deadline(signum, frame):
        raise TimeoutError("Dependency profile verification exceeded 300 seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        raise SystemExit(main())
    finally:
        signal.alarm(0)

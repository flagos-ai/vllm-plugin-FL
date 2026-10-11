#!/usr/bin/env python3
"""Verify/install six pinned ordinary wheels without importing accelerator modules."""

import argparse
import base64
import csv
import email.parser
import hashlib
import importlib.machinery
import importlib.metadata as md
import io
import json
import re
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path, PurePosixPath

NAMES = {
    "vllm",
    "vllm-plugin-fl",
    "musa-gate-observer",
    "flag-gems",
    "flagtree",
    "pyyaml",
}
MODULES = {
    "vllm": "vllm",
    "vllm-plugin-fl": "vllm_fl",
    "musa-gate-observer": "musa_gate_observer",
    "flag-gems": "flag_gems",
    "flagtree": "triton",
    "pyyaml": "yaml",
}
VENDOR_ROLES = {
    "torch_source",
    "torch_musa_source",
    "torch_metadata",
    "torch_musa_metadata",
    "sdk_version",
}
FIXED = {
    "vllm": (
        "0.28.0+empty",
        9106158,
        "2306c2ea119182a5690eb05776cdffa4ba097a92ba13114a6cb49722d0b53fec",
    ),
    "flag-gems": (
        "5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix2",
        12057216,
        "0bdf03d1730885a502c05995ba0d16f89b0f0a6fe8584d24de76dc6e58c69e71",
    ),
    "flagtree": (
        "0.7.0+mthreads3.6.git69148faf.profilefix1",
        180126552,
        "57fd36f4c9be804bb9ab0af2103316746be4c45bc14fe370b543ee0d9c3d1d7e",
    ),
    "pyyaml": (
        "6.0.1",
        705525,
        "ba336e390cd8e4d1739f42dfe9bb83a3cc2e80f567d8805e11b46f4a943f5515",
    ),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def fingerprint(path, cap=512 * 1024 * 1024):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "Regular input required")
    before = path.stat()
    require(before.st_size <= cap, "Input exceeds cap")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            require(size <= cap, "Input grows beyond cap")
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
        == (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev),
        "Input changed while read",
    )
    require(size == before.st_size, "Input size changed")
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def load_manifest(path):
    manifest = json.loads(Path(path).read_bytes())
    require(
        manifest.get("schema_version") == 1
        and manifest.get("ordinary_wheels_only") is True,
        "Ordinary image manifest schema",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", manifest.get("plugin_commit", "")),
        "Committed plugin source required",
    )
    require(
        manifest.get("observer_commit") == manifest["plugin_commit"],
        "Observer and plugin source commit differ",
    )
    for key in ("plugin_build_receipt", "observer_build_receipt"):
        receipt = manifest.get(key, {})
        require(
            isinstance(receipt.get("filename"), str)
            and Path(receipt["filename"]).name == receipt["filename"]
            and type(receipt.get("bytes")) is int
            and receipt["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", "")),
            "Actual source-build receipt binding required",
        )
    rows = manifest.get("wheels", [])
    require(
        len(rows) == 6 and {normalized(x["name"]) for x in rows} == NAMES,
        "Exactly six stack wheels",
    )
    require(len({x["filename"] for x in rows}) == 6, "Duplicate wheel filename")
    for row in rows:
        name = normalized(row["name"])
        require(
            re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", row["filename"]),
            "Safe ordinary wheel filename",
        )
        require(
            type(row["bytes"]) is int
            and 0 < row["bytes"] <= 512 * 1024 * 1024
            and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
            "Explicit whole wheel pin",
        )
        if name in FIXED:
            require(
                (row["version"], row["bytes"], row["sha256"]) == FIXED[name],
                "Validated dependency wheel changed",
            )
        else:
            require(
                row.get("source_commit") == manifest["plugin_commit"],
                "Fresh plugin/observer wheel source commit",
            )
    require(
        manifest.get("vendor_versions")
        == {"torch": "2.9.0", "torch_musa": "2.9.0", "musa_sdk": "4.3.5"},
        "SDK/Torch vendor baseline changed",
    )
    require(
        len(manifest.get("vendor_files", [])) == 5
        and {x["role"] for x in manifest["vendor_files"]} == VENDOR_ROLES,
        "Exactly five vendor source roles",
    )
    require(
        isinstance(manifest.get("source_lineage"), dict) and manifest["source_lineage"],
        "Actual source/build lineage required",
    )
    return manifest


def vendor_snapshot(manifest):
    rows = []
    for expected in manifest["vendor_files"]:
        require(expected["path"].startswith("/usr/local/"), "Protected vendor path")
        actual = fingerprint(expected["path"], 16 * 1024 * 1024)
        require(
            (actual["bytes"], actual["sha256"])
            == (expected["bytes"], expected["sha256"]),
            "Vendor fingerprint mismatch: " + expected["role"],
        )
        rows.append(dict(actual, role=expected["role"]))
    for name in ("torch", "torch_musa"):
        require(md.version(name) == "2.9.0", "Vendor version mismatch: " + name)
    return rows


def wheel_review(path, expected):
    actual = fingerprint(path)
    require(
        (actual["bytes"], actual["sha256"]) == (expected["bytes"], expected["sha256"]),
        "Whole wheel pin mismatch",
    )
    with zipfile.ZipFile(path) as wheel:
        infos = wheel.infolist()
        names = [x.filename for x in infos]
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        regular = set()
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            require(
                parts
                and not info.filename.startswith("/")
                and ".." not in parts
                and "\\" not in info.filename,
                "Unsafe wheel member",
            )
            require(
                not info.flag_bits & 1
                and (info.external_attr >> 16) & 0o170000 != 0o120000,
                "Encrypted/symlink wheel member",
            )
            require(
                not any(
                    x.endswith((".pth", ".egg-link")) or x.startswith("__editable__")
                    for x in parts
                ),
                "Wheel contains editable/source path hook",
            )
            if not info.is_dir():
                regular.add(info.filename)
        metadata_names = [
            x
            for x in regular
            if len(PurePosixPath(x).parts) == 2 and x.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_names) == 1, "One top-level METADATA required")
        metadata = email.parser.BytesParser().parsebytes(wheel.read(metadata_names[0]))
        require(
            normalized(metadata["Name"]) == normalized(expected["name"])
            and metadata["Version"] == expected["version"],
            "Wheel metadata differs from manifest",
        )
        dist_info = metadata_names[0].rsplit("/", 1)[0]
        require(
            not any(
                ".data" in part
                for name in regular
                for part in PurePosixPath(name).parts
            ),
            "Stack wheel cannot relocate data/scripts outside site-packages",
        )
        rows = list(
            csv.reader(io.StringIO(wheel.read(dist_info + "/RECORD").decode("utf-8")))
        )
        require(
            all(len(x) == 3 for x in rows)
            and len(rows) == len(regular)
            and {x[0] for x in rows} == regular,
            "Complete unique wheel RECORD required",
        )
        payload = []
        for member, encoded, size in rows:
            if member == dist_info + "/RECORD":
                require((encoded, size) == ("", ""), "RECORD self row must be empty")
                continue
            body = wheel.read(member)
            digest = hashlib.sha256(body).digest()
            require(
                encoded
                == "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")
                and size == str(len(body)),
                "Wheel RECORD hash mismatch: " + member,
            )
            payload.append(
                {"member": member, "bytes": len(body), "sha256": digest.hex()}
            )
        require(wheel.testzip() is None, "Wheel CRC failure")
    return dict(expected, wheel=actual, dist_info=dist_info, payload=payload)


def reject_editable():
    checked = []
    for root in sys.path:
        candidate = Path(root)
        if not candidate.is_dir():
            continue
        require(
            not list(candidate.glob("*.egg-link"))
            and not list(candidate.glob("__editable__*")),
            "Active interpreter contains editable installation",
        )
    for distribution in md.distributions():
        text = distribution.read_text("direct_url.json")
        if text:
            direct = json.loads(text)
            require(
                direct.get("dir_info", {}).get("editable") is not True,
                "Editable distribution: "
                + distribution.metadata.get("Name", "unknown"),
            )
            checked.append(distribution.metadata.get("Name", "unknown"))
    return sorted(checked)


def installed_review(row, site, venv):
    name = normalized(row["name"])
    distribution = md.distribution(row["name"])
    require(
        distribution.version == row["version"]
        and Path(distribution.locate_file("")).resolve(strict=True) == site,
        "Stack distribution must be installed in image venv",
    )
    direct = json.loads(distribution.read_text("direct_url.json"))
    require(
        "dir_info" not in direct
        and direct.get("url") == Path(row["wheel"]["path"]).as_uri()
        and direct.get("archive_info", {}).get("hashes") == {"sha256": row["sha256"]},
        "Ordinary whole-wheel direct_url required",
    )
    record_text = distribution.read_text("RECORD")
    records = list(csv.reader(io.StringIO(record_text)))
    require(
        all(len(x) == 3 for x in records)
        and len({x[0] for x in records}) == len(records),
        "Installed RECORD uniqueness",
    )
    record = {x[0]: x[1:] for x in records}
    for entry, encoded, size in records:
        path = (site / entry).resolve(strict=True)
        require(path.is_relative_to(venv), "Installed RECORD escapes image venv")
        if not encoded:
            require(entry.endswith(("/RECORD", ".pyc")), "Unhashed installed member")
            continue
        actual = fingerprint(path)
        expected = "sha256=" + base64.urlsafe_b64encode(
            bytes.fromhex(actual["sha256"])
        ).decode().rstrip("=")
        require(
            [encoded, size] == [expected, str(actual["bytes"])],
            "Installed RECORD hash mismatch",
        )
    for member in row["payload"]:
        path = (site / member["member"]).resolve(strict=True)
        require(path.is_relative_to(site), "Wheel member escaped site")
        fp = fingerprint(path)
        require(
            (fp["bytes"], fp["sha256"]) == (member["bytes"], member["sha256"]),
            "Installed payload differs from wheel",
        )
        expected = "sha256=" + base64.urlsafe_b64encode(
            bytes.fromhex(member["sha256"])
        ).decode().rstrip("=")
        require(
            record.get(member["member"]) == [expected, str(member["bytes"])],
            "Wheel payload absent from installed RECORD",
        )
    origin = importlib.machinery.PathFinder.find_spec(MODULES[name], sys.path)
    require(origin is not None and origin.origin, "Stack module absent")
    path = Path(origin.origin).resolve(strict=True)
    require(
        path.is_relative_to(site), "Module resolves to inherited/source installation"
    )
    member = path.relative_to(site).as_posix()
    require(
        any(x["member"] == member for x in row["payload"]),
        "Module origin not wheel payload",
    )
    return {
        "name": row["name"],
        "version": distribution.version,
        "editable": False,
        "origin": fingerprint(path),
        "direct_url": direct,
        "payload_files_verified": len(row["payload"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("wheel", "install", "installed"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    manifest_fp = fingerprint(args.manifest)
    if args.manifest_sha256:
        require(
            re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256)
            and manifest_fp["sha256"] == args.manifest_sha256,
            "Whole stack manifest SHA mismatch",
        )
    if args.expected_commit:
        require(
            args.expected_commit == manifest["plugin_commit"],
            "OCI build commit differs from actual wheel sources",
        )
    wheels = Path(args.wheelhouse).resolve(strict=True)
    require(
        {x.name for x in wheels.iterdir()}
        == {x["filename"] for x in manifest["wheels"]},
        "Wheelhouse contains unexpected files",
    )
    rows = [wheel_review(wheels / x["filename"], x) for x in manifest["wheels"]]
    evidence = {
        "scope": "Ordinary-wheel image static verification; no accelerator import/execution",
        "manifest": manifest_fp,
        "wheels": [{k: x[k] for k in ("name", "version", "wheel")} for x in rows],
        "target_modules_imported": [],
        "GPU_executed": False,
        "model_Gate_passed": False,
    }
    if args.phase != "wheel":
        venv = Path(sys.prefix).resolve(strict=True)
        site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
        require(
            venv == Path("/opt/musa/venv") and site.is_relative_to(venv),
            "Use the dedicated image venv",
        )
        before = vendor_snapshot(manifest)
        reject_editable()
        if args.phase == "install":
            command = [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-index",
                "--no-deps",
                "--ignore-installed",
                "--no-cache-dir",
                "--no-compile",
                *[x["wheel"]["path"] for x in rows],
            ]
            result = subprocess.run(command, check=False, timeout=600)
            require(result.returncode == 0, "Ordinary wheel installation failed")
        evidence["installed"] = [installed_review(x, site, venv) for x in rows]
        evidence["active_direct_urls_checked"] = reject_editable()
        evidence["vendor_before"] = before
        evidence["vendor_after"] = vendor_snapshot(manifest)
        require(
            evidence["vendor_before"] == evidence["vendor_after"],
            "Vendor changed during image installation",
        )
        evidence["vendor_before_after_equal"] = True
        require(
            not any(
                normalized(x.metadata.get("Name", "")) == "flagcx"
                for x in md.distributions()
            ),
            "Unexpected FlagCX distribution",
        )
    Path(args.output).write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "ordinary_wheels": len(rows),
                "output": args.output,
                "success": True,
            }
        )
    )


if __name__ == "__main__":
    main()

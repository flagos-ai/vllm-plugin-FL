#!/usr/bin/env python3
"""Preserve finite CI diagnostics independently from wheels and observer raw.

This CPU-only copier neither imports accelerator packages nor evaluates test
results. It preserves available original bytes and materializes bound sources
and distribution records before excluding the disposable private venv.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

HASH = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".json",
    ".csv",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".sh",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
}
METADATA_NAMES = {"METADATA", "RECORD", "WHEEL", "INSTALLER", "PKG-INFO"}
SECRET_PARTS = {".git", ".ssh", ".aws", ".kube", ".docker", ".azure"}
SECRET_NAMES = {".netrc", ".gitconfig", "auth.json", "credentials", "credentials.json"}
JSON_CAP = 64 * 1024 * 1024


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def regular(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("Noncanonical source: " + str(path))
    if (
        any(part.lower() in SECRET_PARTS for part in path.parts)
        or path.name.lower() in SECRET_NAMES
    ):
        raise ValueError("Credential/configuration paths are not evidence inputs")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Evidence input must be a regular non-symlink file")
    return info


class Preserver:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.groups = {
            name: self.output / name
            for name in ("diagnostics", "observer-raw", "wheels")
        }
        for directory in self.groups.values():
            directory.mkdir()
        self.files = {}
        self.errors = []
        self.absent = []
        self.external_references = []
        self.reference_aliases = []
        self.json_files = []
        self.private_prefix = None

    def error(self, operation, path, error):
        self.errors.append(
            {
                "operation": operation,
                "path": str(path),
                "error": type(error).__name__ + ": " + str(error),
            }
        )

    def check(self, row, expected):
        if expected is None:
            return
        if "bytes" in expected and (
            type(expected["bytes"]) is not int or expected["bytes"] != row["bytes"]
        ):
            raise ValueError("Bound source size changed: " + row["original_path"])
        for key in ("sha256", "sha256_LF"):
            if key in expected and (
                not isinstance(expected[key], str)
                or not HASH.fullmatch(expected[key])
                or expected[key] != row.get(key)
            ):
                raise ValueError("Bound source digest changed: " + row["original_path"])

    def copy(self, source, group, relative, expected=None):
        source = Path(source)
        key = str(source)
        if key in self.files:
            self.check(self.files[key], expected)
            return self.files[key]
        before = regular(source)
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe preservation destination")
        destination = self.groups[group] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        lf_digest = (
            hashlib.sha256()
            if source.suffix in {".py", ".pyi"} or expected and "sha256_LF" in expected
            else None
        )
        count, carry = 0, b""
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(source, flags)
        with os.fdopen(fd, "rb") as incoming, destination.open("xb") as outgoing:
            if identity(os.fstat(incoming.fileno())) != identity(before):
                raise ValueError("Source changed before preservation")
            while chunk := incoming.read(1024 * 1024):
                outgoing.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if lf_digest is not None:
                    block = carry + chunk
                    carry = b"\r" if block.endswith(b"\r") else b""
                    lf_digest.update(
                        (block[:-1] if carry else block).replace(b"\r\n", b"\n")
                    )
            if lf_digest is not None:
                lf_digest.update(carry)
            if identity(os.fstat(incoming.fileno())) != identity(before):
                raise ValueError("Source changed during preservation")
        if count != before.st_size or identity(source.lstat()) != identity(before):
            raise ValueError("Source changed after preservation")
        row = {
            "original_path": key,
            "artifact_group": group,
            "artifact_path": relative.as_posix(),
            "bytes": count,
            "sha256": digest.hexdigest(),
        }
        if lf_digest is not None:
            row["sha256_LF"] = lf_digest.hexdigest()
        self.files[key] = row
        self.check(row, expected)
        if group == "diagnostics" and source.suffix == ".json":
            self.json_files.append((source, destination))
        return row

    def tree(self, source, prefix, gate=False, required=True):
        source = Path(source)
        if not source.exists():
            self.absent.append(str(source))
            if required:
                self.error(
                    "tree",
                    source,
                    FileNotFoundError("Explicit evidence root is absent"),
                )
            return
        if source.resolve(strict=True) != source or not source.is_dir():
            self.error("tree", source, ValueError("Noncanonical evidence directory"))
            return
        for directory, names, filenames in os.walk(
            source,
            followlinks=False,
            onerror=lambda error: self.error("tree", error.filename, error),
        ):
            for name in list(names):
                child = Path(directory) / name
                if child.is_symlink():
                    names.remove(name)
                    self.error("tree", child, ValueError("Evidence symlink directory"))
            for name in sorted(filenames):
                path = Path(directory) / name
                relative = path.relative_to(source)
                group = (
                    "observer-raw"
                    if gate and "observer-traces" in relative.parts
                    else "diagnostics"
                )
                try:
                    self.copy(path, group, Path(prefix) / relative)
                except Exception as error:
                    self.error("copy", path, error)

    def installation(self, source):
        if source is None:
            self.absent.append("MUSA_CI_INSTALL_EVIDENCE was not set")
            self.error(
                "installation",
                "MUSA_CI_INSTALL_EVIDENCE",
                FileNotFoundError("Installation evidence root was not set"),
            )
            return
        source = Path(source)
        if not source.exists():
            self.absent.append(str(source))
            self.error(
                "installation",
                source,
                FileNotFoundError("Installation evidence root is absent"),
            )
            return
        if source.resolve(strict=True) != source or not source.is_dir():
            self.error(
                "installation", source, ValueError("Noncanonical installation root")
            )
            return
        self.private_prefix = source / "venv"
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in {".json", ".raw", ".txt", ".exit"}:
                try:
                    self.copy(path, "diagnostics", Path("installation") / path.name)
                except Exception as error:
                    self.error("installation", path, error)
        for subdirectory in ("wheels", "test-wheels"):
            directory = source / subdirectory
            if directory.is_symlink():
                self.error(
                    "wheels", directory, ValueError("Wheel directory is a symlink")
                )
                continue
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix != ".whl":
                    self.error(
                        "wheels", path, ValueError("Unexpected wheelhouse member")
                    )
                    continue
                small = path.name.startswith(("vllm_plugin_fl-", "musa_gate_observer-"))
                try:
                    self.copy(
                        path,
                        "diagnostics" if small else "wheels",
                        Path("installation") / subdirectory / path.name,
                    )
                except Exception as error:
                    self.error("wheels", path, error)

    def metadata_snapshot(self, item):
        # setup.sh snapshot() records these two original file hashes directly,
        # without an individual {path, sha256} object for either file.
        if "metadata_directory" not in item or "files" not in item:
            return
        files = item["files"]
        # The native-probe variant already contains ordinary fingerprint dicts
        # and is processed by the explicit path/hash walker below.
        if (
            isinstance(files, dict)
            and files
            and all(isinstance(value, dict) for value in files.values())
        ):
            return
        if not isinstance(files, dict) or set(files) != {"METADATA", "RECORD"}:
            raise ValueError("Invalid setup metadata snapshot file schema")
        if not isinstance(item["metadata_directory"], str):
            raise ValueError("Metadata snapshot directory must be a string")
        directory = Path(item["metadata_directory"])
        if (
            not directory.is_absolute()
            or directory.resolve(strict=True) != directory
            or not directory.is_dir()
        ):
            raise ValueError("Metadata snapshot directory must be canonical")
        for name in ("METADATA", "RECORD"):
            digest = files[name]
            if not isinstance(digest, str) or not HASH.fullmatch(digest):
                raise ValueError("Invalid setup metadata snapshot digest")
            path = directory / name
            relative = (
                Path("bound-files")
                / hashlib.sha256(str(path).encode()).hexdigest()[:20]
                / name
            )
            self.copy(path, "diagnostics", relative, {"sha256": digest})

    def bound_sources(self):
        # Parse only copied diagnostic JSON. Never scan/import the private venv.
        pending = list(self.json_files)
        for original_report, copied_report in pending:
            try:
                if copied_report.stat().st_size > JSON_CAP:
                    raise ValueError("Diagnostic JSON exceeds preservation parser cap")
                stack = [json.loads(copied_report.read_bytes())]
                nodes = 0
                while stack:
                    item = stack.pop()
                    nodes += 1
                    if nodes > 1000000:
                        raise ValueError(
                            "Diagnostic JSON exceeds preservation node cap"
                        )
                    if isinstance(item, list):
                        stack.extend(item)
                    elif isinstance(item, dict):
                        stack.extend(item.values())
                        self.metadata_snapshot(item)
                        if "path" not in item or not any(
                            key in item for key in ("sha256", "sha256_LF")
                        ):
                            continue
                        if not isinstance(item["path"], str):
                            raise ValueError("Bound fingerprint path must be a string")
                        path = Path(item["path"])
                        if not path.is_absolute():
                            raise ValueError("Bound fingerprint path must be absolute")
                        if (
                            any(part.lower() in SECRET_PARTS for part in path.parts)
                            or path.name.lower() in SECRET_NAMES
                        ):
                            raise ValueError(
                                "Credential/configuration paths are not evidence inputs"
                            )
                        expected = {
                            key: item[key]
                            for key in ("bytes", "sha256", "sha256_LF")
                            if key in item
                        }
                        if "bytes" in expected and (
                            type(expected["bytes"]) is not int or expected["bytes"] < 0
                        ):
                            raise ValueError("Invalid bound byte count")
                        for key in ("sha256", "sha256_LF"):
                            if key in expected and (
                                not isinstance(expected[key], str)
                                or not HASH.fullmatch(expected[key])
                            ):
                                raise ValueError("Invalid bound digest")
                        if str(path) in self.files:
                            self.check(self.files[str(path)], expected)
                        elif path.suffix == ".whl":
                            # The installer already staged the exact whole wheel.
                            matches = [
                                row
                                for row in self.files.values()
                                if row["artifact_path"].endswith(".whl")
                                and row["sha256"] == expected.get("sha256")
                                and (
                                    "bytes" not in expected
                                    or row["bytes"] == expected["bytes"]
                                )
                            ]
                            if not matches:
                                self.copy(
                                    path,
                                    "wheels",
                                    Path("bound-wheels")
                                    / (
                                        hashlib.sha256(str(path).encode()).hexdigest()[
                                            :16
                                        ]
                                        + "-"
                                        + path.name
                                    ),
                                    expected,
                                )
                            else:
                                self.check(matches[0], expected)
                                self.reference_aliases.append(
                                    {
                                        "report": str(original_report),
                                        "original_path": str(path),
                                        "expected": expected,
                                        "preserved_file": matches[0],
                                        "scope": "The same complete staged wheel is already preserved; this is not a second source read",
                                    }
                                )
                        elif self.private_prefix is not None and path.is_relative_to(
                            self.private_prefix
                        ):
                            relative = (
                                Path("bound-files")
                                / hashlib.sha256(str(path).encode()).hexdigest()[:20]
                                / path.name
                            )
                            source_file = (
                                path.name in METADATA_NAMES
                                or path.suffix in SOURCE_SUFFIXES
                            )
                            self.copy(
                                path,
                                "diagnostics" if source_file else "wheels",
                                relative,
                                expected,
                            )
                        elif (
                            path.name in METADATA_NAMES
                            or path.suffix in SOURCE_SUFFIXES
                        ):
                            relative = (
                                Path("bound-files")
                                / hashlib.sha256(str(path).encode()).hexdigest()[:20]
                                / path.name
                            )
                            self.copy(path, "diagnostics", relative, expected)
                        else:
                            # Model assets/binaries were never part of the private
                            # venv upload. Retain their original declared pins;
                            # do not quietly turn this copier into a model export.
                            self.external_references.append(
                                {
                                    "report": str(original_report),
                                    "reference": dict(path=str(path), **expected),
                                    "materialized": False,
                                    "scope": "External non-source reference; no new certification",
                                }
                            )
            except Exception as error:
                self.error("bound-source-snapshot", original_report, error)

    def finish(self):
        common = {
            "schema_version": 1,
            "github_identity": {
                key: os.environ.get(key)
                for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
            },
            "preservation_passed": not self.errors,
            "errors": self.errors,
            "absent_inputs": self.absent,
            "full_musa_acceptance": False,
            "scope": "Original-byte evidence preservation only; not test or GPU acceptance",
        }
        manifests = {}
        for group in ("observer-raw", "wheels", "diagnostics"):
            report = dict(
                common,
                artifact_group=group,
                files=[
                    row for row in self.files.values() if row["artifact_group"] == group
                ],
            )
            if group == "diagnostics":
                report.update(
                    external_non_source_references=self.external_references,
                    same_whole_wheel_references=self.reference_aliases,
                    companion_manifests=manifests,
                )
            path = self.groups[group] / "preservation-manifest.json"
            raw = (
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            with path.open("xb") as output:
                output.write(raw)
            manifests[group] = {
                "path": group + "/" + path.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        return common


def preserve(candidate_root, suite_root, install_root, gate_root, output_root):
    result = Preserver(output_root)
    for label, operation in (
        ("suite", lambda: result.tree(suite_root, "suite")),
        ("installation", lambda: result.installation(install_root)),
        ("gate", lambda: result.tree(gate_root, "gate", gate=True)),
    ):
        try:
            operation()
        except Exception as error:
            result.error(label, label, error)
    candidate_root = Path(candidate_root)
    for name in ("coverage-musa.json", "report-musa.json"):
        path = candidate_root / name
        if path.exists():
            try:
                result.copy(path, "diagnostics", Path("repository-reports") / name)
            except Exception as error:
                result.error("repository-report", path, error)
    result.tree(
        candidate_root / "test-results/logs",
        "repository-reports/test-results/logs",
        required=False,
    )
    result.bound_sources()
    return result.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--install-root", default="")
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = preserve(
        args.candidate_root,
        args.suite_root,
        Path(args.install_root) if args.install_root else None,
        args.gate_root,
        args.output_root,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("preservation_passed", "errors", "absent_inputs", "scope")
            }
        )
    )
    return 0 if report["preservation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
